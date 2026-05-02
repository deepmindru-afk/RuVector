//! `ruvector-hailo-cluster-bench` — sustained-load harness for the
//! cluster dispatch path.
//!
//! Spawns N concurrent threads, each calling `embed_one_blocking` in a
//! tight loop for `--duration-secs` seconds. At the end, prints
//! throughput + latency percentiles (p50, p90, p99, max, min) computed
//! from every observed sample.
//!
//! Usage:
//!
//!   ruvector-hailo-cluster-bench --workers 127.0.0.1:50071,127.0.0.1:50072 \
//!     --concurrency 8 --duration-secs 10 --dim 384
//!
//! Output is plain-text on stdout; designed for `tee` + manual reading.

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant};

use ruvector_hailo_cluster::{
    Discovery, FileDiscovery, GrpcTransport, HailoClusterEmbedder, StaticDiscovery,
    TailscaleDiscovery,
};
use ruvector_hailo_cluster::transport::WorkerEndpoint;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    let mut workers_arg: Option<String> = None;
    let mut workers_file_arg: Option<String> = None;
    // ADR-172 §1c iter-107 — see embed.rs for the rationale.
    let mut workers_file_sig: Option<String> = None;
    let mut workers_file_pubkey: Option<String> = None;
    let mut tag_arg: Option<String> = None;
    let mut port_arg: u16 = 50051;
    let mut dim: usize = 384;
    let mut concurrency: usize = 4;
    let mut duration_secs: u64 = 5;
    let mut prom_path: Option<String> = None;
    let mut cache_cap: usize = 0;
    let mut cache_ttl_secs: u64 = 0;
    let mut cache_keyspace: usize = 0;
    let mut request_id: String = String::new();
    // Quiet mode suppresses the human-readable stdout summary (config,
    // throughput, percentiles, cache stats). Useful when you only care
    // about the --prom file artifact + exit code.
    let mut quiet = false;
    let mut fingerprint: String = String::new();
    let mut auto_fingerprint = false;
    // ADR-172 §2b iter-102: quorum threshold for auto-fingerprint. 0 =
    // smart default (1 for solo fleet, 2 for ≥2 workers).
    let mut auto_fingerprint_quorum: usize = 0;
    // ADR-172 §2a iter-101 gate — see embed.rs for the rationale; same
    // refusal applies here because bench drives the same cluster code.
    let mut allow_empty_fingerprint = false;
    let mut validate_fleet = false;
    // 0 = no background health-checker. >0 = probe every N seconds in
    // a background tokio task; mismatched fingerprints get hard-ejected
    // and the cache is auto-cleared via the cluster's wired callback.
    let mut health_check_secs: u64 = 0;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--workers" => { workers_arg = args.get(i + 1).cloned(); i += 2; }
            "--workers-file" => { workers_file_arg = args.get(i + 1).cloned(); i += 2; }
            "--workers-file-sig" => { workers_file_sig = args.get(i + 1).cloned(); i += 2; }
            "--workers-file-pubkey" => { workers_file_pubkey = args.get(i + 1).cloned(); i += 2; }
            "--tailscale-tag" => { tag_arg = args.get(i + 1).cloned(); i += 2; }
            "--port" => { port_arg = args.get(i + 1).and_then(|s| s.parse().ok()).unwrap_or(50051); i += 2; }
            "--dim" => { dim = args.get(i + 1).and_then(|s| s.parse().ok()).unwrap_or(384); i += 2; }
            "--concurrency" => { concurrency = args.get(i + 1).and_then(|s| s.parse().ok()).unwrap_or(4); i += 2; }
            "--duration-secs" => { duration_secs = args.get(i + 1).and_then(|s| s.parse().ok()).unwrap_or(5); i += 2; }
            "--prom" => { prom_path = args.get(i + 1).cloned(); i += 2; }
            "--cache" => { cache_cap = args.get(i + 1).and_then(|s| s.parse().ok()).unwrap_or(0); i += 2; }
            "--cache-ttl" => { cache_ttl_secs = args.get(i + 1).and_then(|s| s.parse().ok()).unwrap_or(0); i += 2; }
            "--cache-keyspace" => { cache_keyspace = args.get(i + 1).and_then(|s| s.parse().ok()).unwrap_or(0); i += 2; }
            "--request-id" => { request_id = args.get(i + 1).cloned().unwrap_or_default(); i += 2; }
            "--quiet" => { quiet = true; i += 1; }
            "--fingerprint" => { fingerprint = args.get(i + 1).cloned().unwrap_or_default(); i += 2; }
            "--auto-fingerprint" => { auto_fingerprint = true; i += 1; }
            "--auto-fingerprint-quorum" => {
                auto_fingerprint_quorum =
                    args.get(i + 1).and_then(|s| s.parse().ok()).unwrap_or(0);
                i += 2;
            }
            "--allow-empty-fingerprint" => { allow_empty_fingerprint = true; i += 1; }
            "--validate-fleet" => { validate_fleet = true; i += 1; }
            "--health-check" => {
                health_check_secs = args.get(i + 1).and_then(|s| s.parse().ok()).unwrap_or(0);
                i += 2;
            }
            "--help" | "-h" => { print_help(); return Ok(()); }
            "--version" | "-V" => {
                println!("{} {}", env!("CARGO_PKG_NAME"), env!("CARGO_PKG_VERSION"));
                return Ok(());
            }
            other => return Err(format!("unknown arg: {}", other).into()),
        }
    }

    let discovery: Box<dyn Discovery> = match (workers_arg, workers_file_arg, tag_arg) {
        (Some(csv), None, None) => {
            let workers: Vec<WorkerEndpoint> = csv
                .split(',')
                .filter(|s| !s.is_empty())
                .enumerate()
                .map(|(i, addr)| {
                    WorkerEndpoint::new(format!("static-{}", i), addr.trim().to_string())
                })
                .collect();
            Box::new(StaticDiscovery::new(workers))
        }
        (None, Some(path), None) => {
            let mut fd = FileDiscovery::new(path);
            match (&workers_file_sig, &workers_file_pubkey) {
                (Some(s), Some(p)) => fd = fd.with_signature(s, p),
                (Some(_), None) | (None, Some(_)) => {
                    return Err(
                        "--workers-file-sig and --workers-file-pubkey must both be set or both unset (ADR-172 §1c)"
                            .into(),
                    );
                }
                (None, None) => {}
            }
            Box::new(fd)
        }
        (None, None, Some(tag)) => Box::new(TailscaleDiscovery::new(tag, port_arg)),
        (None, None, None) => {
            return Err(
                "must pass exactly one of --workers / --workers-file / --tailscale-tag".into(),
            );
        }
        _ => {
            return Err(
                "discovery flags are mutually exclusive: pick one of --workers, --workers-file, --tailscale-tag".into(),
            );
        }
    };

    let workers = discovery.discover()?;
    if workers.is_empty() { return Err("0 workers discovered".into()); }
    if !quiet {
        println!("# bench config: workers={} dim={} concurrency={} duration={}s",
            workers.len(), dim, concurrency, duration_secs);
    }

    // Trait-object Arc so we can clone-and-share between the
    // optional `auto_fingerprint` probe cluster and the real cluster
    // below. `Arc::clone` requires precise type match — implicit
    // unsizing only happens at construction-time, not on clone.
    let transport: Arc<dyn ruvector_hailo_cluster::transport::EmbeddingTransport + Send + Sync> =
        Arc::new(GrpcTransport::new()?);

    // Auto-discover with quorum (ADR-172 §2b iter 102). Smart default:
    // quorum=2 when fleet has ≥2 workers, quorum=1 for solo dev fleets.
    if auto_fingerprint {
        let resolved_quorum: usize = if auto_fingerprint_quorum > 0 {
            auto_fingerprint_quorum
        } else if workers.len() >= 2 {
            2
        } else {
            1
        };
        let probe = HailoClusterEmbedder::new(
            workers.clone(),
            Arc::clone(&transport),
            dim,
            "".to_string(),
        )?;
        match probe.discover_fingerprint_with_quorum(resolved_quorum) {
            Ok(fp) if !fp.is_empty() => {
                if !quiet {
                    eprintln!(
                        "ruvector-hailo-cluster-bench: --auto-fingerprint (quorum={}) discovered fp={:?}",
                        resolved_quorum, fp
                    );
                }
                fingerprint = fp;
            }
            Ok(_) => {
                if !quiet {
                    eprintln!(
                        "ruvector-hailo-cluster-bench: --auto-fingerprint: empty fingerprint reported"
                    );
                }
                fingerprint.clear();
            }
            Err(e) => {
                eprintln!(
                    "ruvector-hailo-cluster-bench: --auto-fingerprint failed: {} (continuing without enforcement)",
                    e
                );
                fingerprint.clear();
            }
        }
    }

    // ADR-172 §2a mitigation (iter 101): same gate as embed.rs — refuse
    // to enable cache without a fingerprint binding it.
    if cache_cap > 0 && fingerprint.is_empty() && !allow_empty_fingerprint {
        return Err(
            "refusing --cache > 0 with empty fingerprint (ADR-172 §2a); pass \
             --fingerprint <hex> or --auto-fingerprint, or opt out with \
             --allow-empty-fingerprint"
                .into(),
        );
    }

    let cluster = Arc::new({
        let c = HailoClusterEmbedder::new(workers, transport, dim, fingerprint)?;
        match (cache_cap, cache_ttl_secs) {
            (0, _) => c,
            (cap, 0) => c.with_cache(cap),
            (cap, ttl) => c.with_cache_ttl(cap, Duration::from_secs(ttl)),
        }
    });

    if validate_fleet {
        // Mirror of embed's validation logic — refuse to bench against
        // a drifted/unhealthy fleet because numbers would be meaningless.
        match cluster.validate_fleet() {
            Ok(report) => {
                if !quiet {
                    eprintln!(
                        "ruvector-hailo-cluster-bench: fleet validation: {} healthy, {} mismatched fp, {} not ready, {} unreachable",
                        report.healthy.len(),
                        report.fingerprint_mismatched.len(),
                        report.not_ready.len(),
                        report.unreachable.len(),
                    );
                    for m in &report.fingerprint_mismatched {
                        eprintln!(
                            "  EJECTED {}: expected fp={:?}, actual fp={:?}",
                            m.worker, m.expected, m.actual
                        );
                    }
                }
            }
            Err(e) => {
                // Fail-fast — same exit code as embed for CI consistency.
                eprintln!("ruvector-hailo-cluster-bench: fleet validation FAILED: {}", e);
                std::process::exit(2);
            }
        }
    }

    // Background health-checker — when --health-check N is set, spawn
    // a tokio runtime for the lifetime of main. Same shape as embed's
    // wiring (iter 66). Bound to a name-prefixed local so the runtime
    // + checker live until main returns.
    let _health_keepalive = if health_check_secs > 0 {
        let rt = tokio::runtime::Builder::new_multi_thread()
            .worker_threads(1)
            .enable_all()
            .thread_name("health-check")
            .build()
            .map_err(|e| format!("health-check runtime: {}", e))?;
        let cfg = ruvector_hailo_cluster::HealthCheckerConfig {
            interval: Duration::from_secs(health_check_secs),
            ..cluster.health_checker_config()
        };
        let checker = cluster.spawn_health_checker(rt.handle(), cfg);
        if !quiet {
            eprintln!(
                "ruvector-hailo-cluster-bench: --health-check spawned, interval={}s",
                health_check_secs
            );
        }
        Some((rt, checker))
    } else {
        None
    };

    let stop = Arc::new(AtomicBool::new(false));
    let total_ok = Arc::new(AtomicU64::new(0));
    let total_err = Arc::new(AtomicU64::new(0));
    // Each thread accumulates samples in its own Vec (no lock contention),
    // then we collect them after stop. Pre-allocate a generous estimate
    // (10k samples/thread/sec at 1ms/embed).
    let cap = (duration_secs as usize) * 10_000;

    let start = Instant::now();
    let mut handles = Vec::new();
    for tid in 0..concurrency {
        let cluster = Arc::clone(&cluster);
        let stop = Arc::clone(&stop);
        let total_ok = Arc::clone(&total_ok);
        let total_err = Arc::clone(&total_err);
        let cache_keyspace = cache_keyspace;
        // Per-thread clone so the closure can format ids without locks.
        let request_id = request_id.clone();
        let h = thread::Builder::new()
            .name(format!("bench-{}", tid))
            .spawn(move || {
                let mut samples: Vec<u64> = Vec::with_capacity(cap);
                let mut counter: u64 = 0;
                while !stop.load(Ordering::Relaxed) {
                    let t0 = Instant::now();
                    // When cache_keyspace>0, all threads share the same
                    // bounded keyspace so requests overlap and the cache
                    // sees real hits. With keyspace=0 (default), every
                    // request gets a unique key — useful for measuring
                    // cold dispatch latency.
                    let key = if cache_keyspace > 0 {
                        format!("bench-{}", counter % (cache_keyspace as u64))
                    } else {
                        format!("bench-{}-{}", tid, counter)
                    };
                    // When --request-id is set, suffix tid+counter so
                    // every RPC in the run gets a unique-but-correlated
                    // id (`<run-token>.t3.c42`). Lets ops grep by run
                    // prefix in worker logs.
                    let r = if request_id.is_empty() {
                        cluster.embed_one_blocking(&key)
                    } else {
                        let id = format!("{}.t{}.c{}", request_id, tid, counter);
                        cluster.embed_one_blocking_with_request_id(&key, &id)
                    };
                    let elapsed_us = t0.elapsed().as_micros() as u64;
                    match r {
                        Ok(_) => {
                            total_ok.fetch_add(1, Ordering::Relaxed);
                            samples.push(elapsed_us);
                        }
                        Err(_) => { total_err.fetch_add(1, Ordering::Relaxed); }
                    }
                    counter += 1;
                }
                samples
            })
            .expect("spawn bench thread");
        handles.push(h);
    }

    thread::sleep(Duration::from_secs(duration_secs));
    stop.store(true, Ordering::Relaxed);

    let mut all_samples: Vec<u64> = Vec::with_capacity(cap * concurrency);
    for h in handles {
        let mut s = h.join().expect("bench thread join");
        all_samples.append(&mut s);
    }
    let wall = start.elapsed();
    let ok = total_ok.load(Ordering::Relaxed);
    let err = total_err.load(Ordering::Relaxed);
    let total = ok + err;

    all_samples.sort_unstable();
    let p = |q: f64| {
        if all_samples.is_empty() { return 0u64; }
        let idx = ((all_samples.len() as f64) * q) as usize;
        let idx = idx.min(all_samples.len() - 1);
        all_samples[idx]
    };
    let avg_us = if !all_samples.is_empty() {
        all_samples.iter().sum::<u64>() / (all_samples.len() as u64)
    } else { 0 };
    let throughput = (ok as f64) / wall.as_secs_f64();

    if !quiet {
        println!("---");
        println!("wall_seconds      : {:.3}", wall.as_secs_f64());
        println!("requests_total    : {}", total);
        println!("requests_ok       : {}", ok);
        println!("requests_err      : {}", err);
        println!("throughput_per_s  : {:.1}", throughput);
        println!("latency_us:");
        println!("  min             : {}", p(0.0));
        println!("  p50             : {}", p(0.50));
        println!("  p90             : {}", p(0.90));
        println!("  p99             : {}", p(0.99));
        println!("  max             : {}", p(1.0));
        println!("  avg             : {}", avg_us);
        println!("samples_collected : {}", all_samples.len());
    }

    // Cache stats are needed by both the stdout block (suppressed in
    // quiet) and the Prom file (written regardless), so compute once
    // and conditionally print.
    let cache_stats = if cache_cap > 0 {
        let s = cluster.cache_stats();
        if !quiet {
            println!(
                "cache: cap={} size={} hits={} misses={} evictions={} hit_rate={:.3}",
                s.capacity,
                s.size,
                s.hits,
                s.misses,
                s.evictions,
                if s.hits + s.misses > 0 {
                    (s.hits as f64) / ((s.hits + s.misses) as f64)
                } else {
                    0.0
                },
            );
        }
        Some(s)
    } else {
        None
    };

    if let Some(path) = prom_path.as_deref() {
        write_prom_textfile(path, &BenchSummary {
            wall, ok, err, throughput, avg_us,
            min_us: p(0.0), p50_us: p(0.50), p90_us: p(0.90), p99_us: p(0.99), max_us: p(1.0),
            samples: all_samples.len(),
            concurrency,
            cache: cache_stats,
        })?;
        if !quiet {
            println!("# wrote prometheus textfile: {}", path);
        }
    }
    Ok(())
}

/// Aggregated bench result, written to Prometheus textfile when `--prom`
/// is set. Field order = print order, so adding a metric is a one-line
/// touch in `write_prom_textfile`.
struct BenchSummary {
    wall: Duration,
    ok: u64,
    err: u64,
    throughput: f64,
    avg_us: u64,
    min_us: u64,
    p50_us: u64,
    p90_us: u64,
    p99_us: u64,
    max_us: u64,
    samples: usize,
    concurrency: usize,
    /// `None` when --cache 0; otherwise carries hit/miss/eviction counts
    /// so the Prom output reflects what actually happened on the cache.
    cache: Option<ruvector_hailo_cluster::cache::CacheStats>,
}

/// Emit Prometheus textfile-collector format. node_exporter's textfile
/// collector picks this up if dropped under `--collector.textfile.directory`.
/// Stable metric names so a CI scrape can alert on regressions across runs.
fn write_prom_textfile(path: &str, s: &BenchSummary) -> std::io::Result<()> {
    use std::io::Write as _;
    // Atomic write — write to <path>.tmp then rename, so a scraper that
    // races us never sees a half-written file.
    let tmp = format!("{}.tmp", path);
    let mut f = std::fs::File::create(&tmp)?;
    let labels = format!("concurrency=\"{}\"", s.concurrency);
    writeln!(f, "# HELP ruvector_hailo_bench_wall_seconds Wall-clock duration of the benchmark run.")?;
    writeln!(f, "# TYPE ruvector_hailo_bench_wall_seconds gauge")?;
    writeln!(f, "ruvector_hailo_bench_wall_seconds{{{}}} {:.3}", labels, s.wall.as_secs_f64())?;
    writeln!(f, "# HELP ruvector_hailo_bench_requests_total Total embed requests issued.")?;
    writeln!(f, "# TYPE ruvector_hailo_bench_requests_total counter")?;
    writeln!(f, "ruvector_hailo_bench_requests_total{{{},outcome=\"ok\"}} {}", labels, s.ok)?;
    writeln!(f, "ruvector_hailo_bench_requests_total{{{},outcome=\"err\"}} {}", labels, s.err)?;
    writeln!(f, "# HELP ruvector_hailo_bench_throughput_per_second Successful embeds per wall-second.")?;
    writeln!(f, "# TYPE ruvector_hailo_bench_throughput_per_second gauge")?;
    writeln!(f, "ruvector_hailo_bench_throughput_per_second{{{}}} {:.3}", labels, s.throughput)?;
    writeln!(f, "# HELP ruvector_hailo_bench_latency_microseconds End-to-end embed latency observed by the bench client.")?;
    writeln!(f, "# TYPE ruvector_hailo_bench_latency_microseconds gauge")?;
    writeln!(f, "ruvector_hailo_bench_latency_microseconds{{{},quantile=\"0\"}} {}", labels, s.min_us)?;
    writeln!(f, "ruvector_hailo_bench_latency_microseconds{{{},quantile=\"0.5\"}} {}", labels, s.p50_us)?;
    writeln!(f, "ruvector_hailo_bench_latency_microseconds{{{},quantile=\"0.9\"}} {}", labels, s.p90_us)?;
    writeln!(f, "ruvector_hailo_bench_latency_microseconds{{{},quantile=\"0.99\"}} {}", labels, s.p99_us)?;
    writeln!(f, "ruvector_hailo_bench_latency_microseconds{{{},quantile=\"1\"}} {}", labels, s.max_us)?;
    writeln!(f, "# HELP ruvector_hailo_bench_latency_avg_microseconds Mean observed embed latency.")?;
    writeln!(f, "# TYPE ruvector_hailo_bench_latency_avg_microseconds gauge")?;
    writeln!(f, "ruvector_hailo_bench_latency_avg_microseconds{{{}}} {}", labels, s.avg_us)?;
    writeln!(f, "# HELP ruvector_hailo_bench_samples Latency samples collected during the run.")?;
    writeln!(f, "# TYPE ruvector_hailo_bench_samples gauge")?;
    writeln!(f, "ruvector_hailo_bench_samples{{{}}} {}", labels, s.samples)?;
    if let Some(c) = &s.cache {
        writeln!(f, "# HELP ruvector_hailo_bench_cache_hits_total Cache hits during the bench run.")?;
        writeln!(f, "# TYPE ruvector_hailo_bench_cache_hits_total counter")?;
        writeln!(f, "ruvector_hailo_bench_cache_hits_total{{{}}} {}", labels, c.hits)?;
        writeln!(f, "# HELP ruvector_hailo_bench_cache_misses_total Cache misses during the bench run.")?;
        writeln!(f, "# TYPE ruvector_hailo_bench_cache_misses_total counter")?;
        writeln!(f, "ruvector_hailo_bench_cache_misses_total{{{}}} {}", labels, c.misses)?;
        writeln!(f, "# HELP ruvector_hailo_bench_cache_evictions_total Cache evictions during the bench run.")?;
        writeln!(f, "# TYPE ruvector_hailo_bench_cache_evictions_total counter")?;
        writeln!(f, "ruvector_hailo_bench_cache_evictions_total{{{}}} {}", labels, c.evictions)?;
        writeln!(f, "# HELP ruvector_hailo_bench_cache_size Final cache size at end of run.")?;
        writeln!(f, "# TYPE ruvector_hailo_bench_cache_size gauge")?;
        writeln!(f, "ruvector_hailo_bench_cache_size{{{}}} {}", labels, c.size)?;
        writeln!(f, "# HELP ruvector_hailo_bench_cache_hit_rate Hits / (hits + misses); 0 if no requests.")?;
        writeln!(f, "# TYPE ruvector_hailo_bench_cache_hit_rate gauge")?;
        let hit_rate = if c.hits + c.misses > 0 {
            (c.hits as f64) / ((c.hits + c.misses) as f64)
        } else { 0.0 };
        writeln!(f, "ruvector_hailo_bench_cache_hit_rate{{{}}} {:.4}", labels, hit_rate)?;
    }
    f.sync_all()?;
    drop(f);
    std::fs::rename(&tmp, path)?;
    Ok(())
}

fn print_help() {
    eprintln!(
        "ruvector-hailo-cluster-bench — sustained-load harness

USAGE:
    ruvector-hailo-cluster-bench [OPTIONS]

DISCOVERY (exactly one):
    --workers <addr1,addr2,...>     Static worker list (csv).
    --workers-file <path>           Manifest file: one host:port or
                                     `name = host:port` per line.
    --tailscale-tag <tag> [--port N]  Discover via tailscale.

OPTIONS:
    --concurrency <N>               Concurrent client threads (default 4).
    --duration-secs <N>             Run length seconds (default 5).
    --dim <N>                       Expected embedding dim (default 384).
    --prom <path>                   Write Prometheus textfile-collector
                                     output to <path> after the run, for
                                     CI regression alerts. Atomic write.
    --cache <N>                     Enable LRU cache of size N on the
                                     cluster coordinator. 0 = disabled.
    --cache-ttl <secs>              TTL for cached entries (used if
                                     --cache > 0). 0 = no TTL.
    --cache-keyspace <N>             Bound the unique key count to N so
                                     all bench threads share the same
                                     keyspace and the cache sees real
                                     hit traffic. 0 = each request gets
                                     a unique key (cold-path benchmark).
    --request-id <token>            Tracing token suffixed with .t<tid>.
                                     c<counter> per RPC. Lets ops grep
                                     a whole bench run from worker logs
                                     by the shared <token> prefix.
    --quiet                         Suppress the human-readable stdout
                                     summary. Pair with --prom to write
                                     a metrics file silently.
    --fingerprint <hex>             Reject workers reporting different
                                     fingerprints. Empty = no enforcement.
    --auto-fingerprint              Probe the fleet for its fingerprint
                                     and use that as the expected value.
    --auto-fingerprint-quorum <N>   Workers that must agree on the fp
                                     (ADR-172 §2b). Default: 2 if fleet
                                     has ≥2 workers, 1 otherwise.
    --allow-empty-fingerprint       Opt out of the ADR-172 §2a safety gate
                                     that refuses --cache > 0 with empty fp.
                                     Risks silent stale-serve from drift.
    --validate-fleet                Probe every worker on startup;
                                     refuse to bench (exit 2) if fleet
                                     has 0 healthy workers. Pairs with
                                     --auto-fingerprint to discover-then-
                                     enforce in one CI step.
    --health-check <secs>           Spawn a background health-checker
                                     that probes every <secs> seconds
                                     during the bench. Mismatched
                                     fingerprints get hard-ejected from
                                     dispatch + auto-clear the cache.
                                     0 = disabled.
    --help, -h                      Print this help.
    --version, -V                   Print the binary name + version and exit.
"
    );
}
