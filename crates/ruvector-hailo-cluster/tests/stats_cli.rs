//! End-to-end integration tests for the `ruvector-hailo-stats` binary.
//!
//! Mirrors `embed_cli.rs` (iter 70) — spawn the real binary, drive it
//! via `std::process::Command`, assert on stdout / exit code / stderr.
//! Catches CLI-level regressions when refactoring `src/bin/stats.rs`.

use std::process::Command;

mod common;
use common::{free_port, spawn_fakeworker};

const STATS: &str = env!("CARGO_BIN_EXE_ruvector-hailo-stats");

#[test]
fn stats_cli_list_workers_does_not_require_live_workers() {
    // --list-workers short-circuits before any RPC, so it works against
    // arbitrary addresses with no actual server. Verifies the discovery
    // → print path doesn't accidentally regress to needing live workers.
    let out = Command::new(STATS)
        .args([
            "--workers", "10.255.255.1:50051,10.255.255.2:50051",
            "--list-workers",
        ])
        .output()
        .expect("run stats");
    assert!(out.status.success(), "stats --list-workers exited {:?}", out.status);

    let stdout = String::from_utf8_lossy(&out.stdout);
    let lines: Vec<&str> = stdout.lines().collect();
    assert_eq!(lines.len(), 3, "header + 2 workers, got: {}", stdout);
    assert!(lines[0].starts_with("worker\taddress"));
    assert!(lines[1].contains("10.255.255.1:50051"));
    assert!(lines[2].contains("10.255.255.2:50051"));
}

#[test]
fn stats_cli_default_tsv_against_live_worker() {
    let port = free_port();
    let mut worker = spawn_fakeworker(port, 384, "fp:test");

    let out = Command::new(STATS)
        .args(["--workers", &format!("127.0.0.1:{}", port)])
        .output()
        .expect("run stats");

    let _ = worker.kill();
    let _ = worker.wait();

    assert!(out.status.success(), "stats exited {:?}, stderr: {}",
        out.status, String::from_utf8_lossy(&out.stderr));
    let stdout = String::from_utf8_lossy(&out.stdout);
    let lines: Vec<&str> = stdout.lines().collect();
    // Header + 1 worker row.
    assert_eq!(lines.len(), 2, "expected header+1 row, got: {}", stdout);
    assert!(lines[0].starts_with("worker\taddress\tfingerprint"),
        "unexpected header: {}", lines[0]);
    assert!(lines[1].contains("fp:test"),
        "fingerprint should appear in row: {}", lines[1]);
}

#[test]
fn stats_cli_json_output_includes_fingerprint_field() {
    let port = free_port();
    let mut worker = spawn_fakeworker(port, 384, "fp:json-test");

    let out = Command::new(STATS)
        .args(["--workers", &format!("127.0.0.1:{}", port), "--json"])
        .output()
        .expect("run stats");

    let _ = worker.kill();
    let _ = worker.wait();

    assert!(out.status.success());
    let stdout = String::from_utf8_lossy(&out.stdout);
    let line = stdout.trim();
    assert!(line.contains("\"fingerprint\":\"fp:json-test\""),
        "JSON should include fingerprint, got: {}", line);
    assert!(line.contains("\"stats\":"), "JSON should include stats");
}

#[test]
fn stats_cli_tsv_includes_rate_limit_columns() {
    // Iter-105 (ADR-172 §3b follow-up): rl_denials + rl_peers must
    // surface in the default TSV. Fakeworker reports 0 for both
    // (limiter not exercised), but the columns are present.
    let port = free_port();
    let mut worker = spawn_fakeworker(port, 4, "fp:rl");

    let out = Command::new(STATS)
        .args(["--workers", &format!("127.0.0.1:{}", port)])
        .output()
        .expect("run stats");

    let _ = worker.kill();
    let _ = worker.wait();

    assert!(out.status.success(), "stderr: {}", String::from_utf8_lossy(&out.stderr));
    let stdout = String::from_utf8_lossy(&out.stdout);
    let lines: Vec<&str> = stdout.lines().collect();
    assert!(lines[0].contains("rl_denials"), "header should include rl_denials: {}", lines[0]);
    assert!(lines[0].contains("rl_peers"), "header should include rl_peers: {}", lines[0]);
    let cols: Vec<&str> = lines[1].split('\t').collect();
    assert_eq!(cols.len(), 12, "expected 12 columns in TSV row, got {}: {:?}", cols.len(), cols);
    // Last two columns are u64 — must parse cleanly.
    assert!(cols[10].parse::<u64>().is_ok(), "rl_denials should be u64: {:?}", cols[10]);
    assert!(cols[11].parse::<u64>().is_ok(), "rl_peers should be u64: {:?}", cols[11]);
}

#[test]
fn stats_cli_prom_output_includes_rate_limit_metrics() {
    let port = free_port();
    let mut worker = spawn_fakeworker(port, 4, "fp:rl-prom");

    let out = Command::new(STATS)
        .args(["--workers", &format!("127.0.0.1:{}", port), "--prom"])
        .output()
        .expect("run stats");

    let _ = worker.kill();
    let _ = worker.wait();

    assert!(out.status.success(), "stderr: {}", String::from_utf8_lossy(&out.stderr));
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(
        stdout.contains("ruvector_rate_limit_denials_total"),
        "prom output missing denials metric: {}",
        stdout
    );
    assert!(
        stdout.contains("ruvector_rate_limit_tracked_peers"),
        "prom output missing tracked_peers metric: {}",
        stdout
    );
    // HELP/TYPE sections should also document them.
    assert!(stdout.contains("# HELP ruvector_rate_limit_denials_total"));
    assert!(stdout.contains("# TYPE ruvector_rate_limit_tracked_peers gauge"));
}

#[test]
fn stats_cli_strict_homogeneous_with_drift_exits_three() {
    // Two workers, different fingerprints — drift detected.
    // --strict-homogeneous turns drift into exit 3.
    let port_a = free_port();
    let port_b = free_port();
    let mut wa = spawn_fakeworker(port_a, 384, "fp:current");
    let mut wb = spawn_fakeworker(port_b, 384, "fp:stale");

    let out = Command::new(STATS)
        .args([
            "--workers",
            &format!("127.0.0.1:{},127.0.0.1:{}", port_a, port_b),
            "--strict-homogeneous",
        ])
        .output()
        .expect("run stats");

    let _ = wa.kill();
    let _ = wa.wait();
    let _ = wb.kill();
    let _ = wb.wait();

    assert_eq!(out.status.code(), Some(3),
        "drift + --strict-homogeneous should exit 3, got {:?}", out.status);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(stderr.contains("DRIFT"),
        "stderr should mention DRIFT, got: {}", stderr);
}

#[test]
fn stats_cli_version_flag_prints_pkg_name_and_version() {
    for arg in &["--version", "-V"] {
        let out = Command::new(STATS).arg(arg).output().expect("run stats");
        assert!(out.status.success());
        let line = String::from_utf8_lossy(&out.stdout).trim().to_string();
        assert!(line.starts_with("ruvector-hailo-cluster"), "got: {}", line);
        assert_eq!(line.split_whitespace().count(), 2);
    }
}

#[test]
fn stats_cli_strict_homogeneous_with_no_drift_exits_zero() {
    // Same fingerprint on both workers → no drift → exit 0.
    let port_a = free_port();
    let port_b = free_port();
    let mut wa = spawn_fakeworker(port_a, 384, "fp:same");
    let mut wb = spawn_fakeworker(port_b, 384, "fp:same");

    let out = Command::new(STATS)
        .args([
            "--workers",
            &format!("127.0.0.1:{},127.0.0.1:{}", port_a, port_b),
            "--strict-homogeneous",
        ])
        .output()
        .expect("run stats");

    let _ = wa.kill();
    let _ = wa.wait();
    let _ = wb.kill();
    let _ = wb.wait();

    assert!(out.status.success(),
        "homogeneous fleet should exit 0, got {:?}", out.status);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(!stderr.contains("DRIFT"),
        "stderr must NOT mention DRIFT for homogeneous fleet, got: {}", stderr);
}
