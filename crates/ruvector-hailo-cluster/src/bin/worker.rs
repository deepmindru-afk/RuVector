//! `ruvector-hailo-worker` — gRPC server that serves text embedding via
//! a Hailo-8 NPU.
//!
//! ADR-167 §5 step 10 (worker side of the cluster). On the Pi 5 + AI HAT+
//! this wraps the local `HailoEmbedder` and serves the same
//! `embedding_server::Embedding` trait that `ruvector-hailo-cluster`'s
//! `GrpcTransport` calls into.
//!
//! Env vars:
//!   RUVECTOR_WORKER_BIND   socket addr to listen on   (default 0.0.0.0:50051)
//!   RUVECTOR_MODEL_DIR     dir holding model.hef + vocab.txt
//!                          (default ./models/all-minilm-l6-v2)
//!
//! Without the `hailo` feature, `HailoEmbedder::open()` returns
//! `FeatureDisabled` and the worker exits with a clear message — useful
//! to validate the binary builds + arg parsing without a Pi attached.

use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

use std::sync::atomic::{AtomicU64, Ordering};

use std::pin::Pin;

use ruvector_hailo::HailoEmbedder;
use ruvector_hailo_cluster::compute_fingerprint;
use ruvector_hailo_cluster::proto::embedding_server::{Embedding, EmbeddingServer};
use ruvector_hailo_cluster::proto::{
    EmbedBatchRequest, EmbedRequest, EmbedResponse, EmbedStreamResponse, HealthRequest,
    HealthResponse, StatsRequest, StatsResponse,
};
use tokio_stream::wrappers::ReceiverStream;
use tonic::{transport::Server, Request, Response, Status};
use tracing::{error, info, instrument, warn};

/// The actual gRPC service. Holds a thread-safe HailoEmbedder.
struct WorkerService {
    embedder: Arc<HailoEmbedder>,
    /// Server-reported version string — used by the coordinator's health
    /// check and surfaced in logs.
    version: String,
    /// PCIe BDF of the device the embedder opened — surfaces to clients
    /// for debugging which fleet member served them.
    device_id: String,
    /// sha256(HEF || vocab.txt) — coordinator refuses to mix workers with
    /// different fingerprints (ADR-167 §8.3 fleet integrity guard).
    /// Phase 1 ships an empty string until step 6 (HEF) lands; coordinator
    /// treats empty fingerprint as "skip the check".
    fingerprint: String,
    /// Process start time, for uptime reporting in GetStats.
    start: Instant,
    /// Atomic counters surfaced via GetStats.
    embed_ok: AtomicU64,
    embed_err: AtomicU64,
    health_count: AtomicU64,
    latency_sum_us: AtomicU64,
    latency_min_us: AtomicU64,
    latency_max_us: AtomicU64,
}

#[tonic::async_trait]
impl Embedding for WorkerService {
    #[instrument(skip(self, request), fields(text_len, latency_us, dim, request_id))]
    async fn embed(
        &self,
        request: Request<EmbedRequest>,
    ) -> Result<Response<EmbedResponse>, Status> {
        // Prefer the gRPC metadata header (canonical) over the proto
        // field (back-compat). Defaults to "-" when neither is set.
        let req_id_owned = ruvector_hailo_cluster::proto::extract_request_id(
            &request,
            &request.get_ref().request_id,
        );
        let req = request.into_inner();
        let req_id_field: &str = if req_id_owned.is_empty() { "-" } else { &req_id_owned };
        tracing::Span::current()
            .record("text_len", req.text.len())
            .record("request_id", req_id_field);

        let start = Instant::now();
        match self.embedder.embed(&req.text) {
            Ok(v) => {
                let dim = self.embedder.dimensions() as u32;
                let latency_us = start.elapsed().as_micros() as u64;
                tracing::Span::current()
                    .record("latency_us", latency_us)
                    .record("dim", dim);
                self.embed_ok.fetch_add(1, Ordering::Relaxed);
                self.latency_sum_us.fetch_add(latency_us, Ordering::Relaxed);
                update_min(&self.latency_min_us, latency_us);
                update_max(&self.latency_max_us, latency_us);
                info!("embed ok");
                Ok(Response::new(EmbedResponse {
                    vector: v,
                    dim,
                    latency_us: latency_us as i64,
                }))
            }
            Err(e) => {
                self.embed_err.fetch_add(1, Ordering::Relaxed);
                warn!(error = %e, "embed failed");
                Err(Status::internal(format!("embed: {}", e)))
            }
        }
    }

    #[instrument(skip_all)]
    async fn health(
        &self,
        _request: Request<HealthRequest>,
    ) -> Result<Response<HealthResponse>, Status> {
        self.health_count.fetch_add(1, Ordering::Relaxed);
        Ok(Response::new(HealthResponse {
            version: self.version.clone(),
            device_id: self.device_id.clone(),
            model_fingerprint: self.fingerprint.clone(),
            // Worker is "ready" iff embedder.dimensions() returned a real
            // dim. Iter 87+: open() pre-declares MINI_LM_DIM = 384 so the
            // worker reports ready=true and the coordinator dispatches
            // even before the .hef lands (FNV-1a placeholder vectors).
            // When HEF wiring lands the dim will come from the loaded
            // network group's output shape instead.
            ready: self.embedder.dimensions() > 0,
        }))
    }

    type EmbedStreamStream =
        Pin<Box<dyn futures_core::Stream<Item = Result<EmbedStreamResponse, Status>> + Send + 'static>>;

    #[instrument(skip(self, request), fields(batch_size, request_id))]
    async fn embed_stream(
        &self,
        request: Request<EmbedBatchRequest>,
    ) -> Result<Response<Self::EmbedStreamStream>, Status> {
        let req_id_owned = ruvector_hailo_cluster::proto::extract_request_id(
            &request,
            &request.get_ref().request_id,
        );
        let req = request.into_inner();
        let n = req.texts.len();
        let req_id_field: &str = if req_id_owned.is_empty() { "-" } else { &req_id_owned };
        tracing::Span::current()
            .record("batch_size", n)
            .record("request_id", req_id_field);
        info!("worker embed_stream");

        let embedder = Arc::clone(&self.embedder);
        let dim = embedder.dimensions() as u32;
        let (tx, rx) = tokio::sync::mpsc::channel::<Result<EmbedStreamResponse, Status>>(n.max(1));

        // Spawn the embed work — order-preserving sequential issue (the
        // current HailoEmbedder is single-threaded; later iterations can
        // pipeline once the NPU is hooked up). `index` field guards the
        // contract that consumers reorder if needed.
        tokio::task::spawn(async move {
            for (i, text) in req.texts.into_iter().enumerate() {
                let start = Instant::now();
                let item = match embedder.embed(&text) {
                    Ok(v) => Ok(EmbedStreamResponse {
                        index: i as u32,
                        vector: v,
                        dim,
                        latency_us: start.elapsed().as_micros() as i64,
                    }),
                    Err(e) => Err(Status::internal(format!("embed[{}]: {}", i, e))),
                };
                if tx.send(item).await.is_err() {
                    // Client cancelled mid-stream — bail out.
                    break;
                }
            }
        });

        let stream = ReceiverStream::new(rx);
        Ok(Response::new(Box::pin(stream)))
    }

    #[instrument(skip_all)]
    async fn get_stats(
        &self,
        _request: Request<StatsRequest>,
    ) -> Result<Response<StatsResponse>, Status> {
        let min = self.latency_min_us.load(Ordering::Relaxed);
        Ok(Response::new(StatsResponse {
            embed_count: self.embed_ok.load(Ordering::Relaxed),
            error_count: self.embed_err.load(Ordering::Relaxed),
            health_count: self.health_count.load(Ordering::Relaxed),
            latency_us_sum: self.latency_sum_us.load(Ordering::Relaxed),
            latency_us_min: if min == u64::MAX { 0 } else { min },
            latency_us_max: self.latency_max_us.load(Ordering::Relaxed),
            uptime_seconds: self.start.elapsed().as_secs(),
        }))
    }
}

fn update_min(slot: &AtomicU64, v: u64) {
    let mut cur = slot.load(Ordering::Relaxed);
    while v < cur {
        match slot.compare_exchange_weak(cur, v, Ordering::Relaxed, Ordering::Relaxed) {
            Ok(_) => break,
            Err(actual) => cur = actual,
        }
    }
}
fn update_max(slot: &AtomicU64, v: u64) {
    let mut cur = slot.load(Ordering::Relaxed);
    while v > cur {
        match slot.compare_exchange_weak(cur, v, Ordering::Relaxed, Ordering::Relaxed) {
            Ok(_) => break,
            Err(actual) => cur = actual,
        }
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Tracing init — env-driven (`RUST_LOG=info` etc.). Writes to stderr
    // so stdout stays clean for any future structured-output piping.
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .with_writer(std::io::stderr)
        .init();

    let bind: std::net::SocketAddr = std::env::var("RUVECTOR_WORKER_BIND")
        .unwrap_or_else(|_| "0.0.0.0:50051".into())
        .parse()?;
    let model_dir: PathBuf = std::env::var("RUVECTOR_MODEL_DIR")
        .unwrap_or_else(|_| "./models/all-minilm-l6-v2".into())
        .into();

    info!(bind = %bind, model_dir = %model_dir.display(), "ruvector-hailo-worker starting");

    // Compute the model fingerprint *before* opening the device, so
    // even if HailoEmbedder::open fails we've recorded the artifact
    // identity (or its absence — empty string).
    let fingerprint = compute_fingerprint(&model_dir);
    if fingerprint.is_empty() {
        warn!(
            "model_dir {} has no model.hef / vocab.txt — fingerprint empty; \
             coordinators will skip the integrity check",
            model_dir.display()
        );
    } else {
        info!(fingerprint = %fingerprint, "model fingerprint computed");
    }

    // Open the NPU + load the model. This is the only place that can
    // surface FeatureDisabled / NotYetImplemented today; the binary exits
    // cleanly with the underlying error message so operators see what's
    // missing.
    let embedder = HailoEmbedder::open(&model_dir).map_err(|e| {
        error!(error = %e, model_dir = %model_dir.display(), "HailoEmbedder::open failed");
        format!(
            "failed to open HailoEmbedder at {}: {} \
             (rebuild with --features hailo on a Pi 5 + AI HAT+, \
              and ensure model.hef + vocab.txt exist)",
            model_dir.display(),
            e
        )
    })?;
    let device_id = embedder.device_id().to_string();

    // Iter 95 (ADR-174 §93): log the NPU on-die temperature once at
    // startup so operators see baseline thermal state without polling.
    // Hailo-8 has two thermal sensors per die; we log both. None means
    // the read failed (firmware unsupported on this board variant); the
    // `ruvector-hailo-stats` integration in iter 96 surfaces it
    // continuously via the Health RPC.
    match embedder.chip_temperature() {
        Some((ts0, ts1)) => {
            info!(
                ts0_celsius = ts0,
                ts1_celsius = ts1,
                "Hailo-8 NPU on-die temperature at startup"
            );
        }
        None => {
            // Soft warn — older Hailo firmware doesn't expose the
            // temperature opcode; not a startup-blocking issue.
            tracing::warn!("Hailo-8 NPU temperature read returned None (firmware may not support the opcode)");
        }
    }

    let svc = WorkerService {
        embedder: Arc::new(embedder),
        version: format!("ruvector-hailo-worker {}", env!("CARGO_PKG_VERSION")),
        device_id,
        fingerprint,
        start: Instant::now(),
        embed_ok: AtomicU64::new(0),
        embed_err: AtomicU64::new(0),
        health_count: AtomicU64::new(0),
        latency_sum_us: AtomicU64::new(0),
        latency_min_us: AtomicU64::new(u64::MAX),
        latency_max_us: AtomicU64::new(0),
    };

    let rt = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(2)
        .enable_all()
        .build()?;

    rt.block_on(async move {
        info!(addr = %bind, "ruvector-hailo-worker serving");
        Server::builder()
            .add_service(EmbeddingServer::new(svc))
            .serve_with_shutdown(bind, shutdown_signal())
            .await
    })?;

    Ok(())
}

/// Future that resolves when SIGINT or SIGTERM arrives — graceful exit.
async fn shutdown_signal() {
    use tokio::signal::unix::{signal, SignalKind};
    let mut sigterm = signal(SignalKind::terminate())
        .expect("install SIGTERM handler");
    let mut sigint = signal(SignalKind::interrupt())
        .expect("install SIGINT handler");
    tokio::select! {
        _ = sigterm.recv() => info!("SIGTERM received, shutting down"),
        _ = sigint.recv()  => info!("SIGINT received, shutting down"),
    }
}
