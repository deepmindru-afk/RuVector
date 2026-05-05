//! `ruview-mcp-brain-mini` — minimal HTTP brain for the cognitum
//! cluster (ADR-183 Tier 2 iter 8).
//!
//! Wire-compatible with the existing `mcp-brain-serve` REST shape so
//! both [`ruview_vitals_worker::brain::BrainClient`] and RuView's own
//! `brain_bridge.rs` post into it without code change:
//!
//! ```text
//!   POST /memories    {category, content}
//!     -> 201 {id, category, content, content_hash, created_at}
//!
//!   GET /memories?category=X&limit=N
//!     -> 200 {count, total, offset, memories: [...]}
//! ```
//!
//! Storage: in-memory `Vec<Memory>` with optional JSONL append-only
//! persistence behind `RUVIEW_BRAIN_STORE_PATH`. Restart-load is
//! best-effort; corrupt lines are skipped with a WARN. Concurrency:
//! one `tokio::sync::RwLock<Vec<Memory>>` — fine for the cluster's
//! peak rate (~4 hosts × 1 POST/30 s).
//!
//! This is intentionally a tiny brain. Pluging in the full
//! `mcp-brain-server-local` (HNSW vector search, AIDefence, etc.) is
//! a future iter; the workers don't need vector recall, just durable
//! memory ingest.

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use axum::{
    extract::{Query, State},
    http::StatusCode,
    response::IntoResponse,
    routing::get,
    Json, Router,
};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use tokio::net::TcpListener;
use tokio::sync::RwLock;
use tracing_subscriber::EnvFilter;

#[derive(Debug, Clone, Serialize, Deserialize)]
struct Memory {
    id: String,
    category: String,
    content: String,
    content_hash: String,
    created_at: u64,
}

#[derive(Debug, Deserialize)]
struct PostBody {
    category: String,
    content: String,
}

#[derive(Debug, Deserialize)]
struct ListQuery {
    category: Option<String>,
    limit: Option<usize>,
    offset: Option<usize>,
}

#[derive(Default)]
struct Store {
    /// All memories, append order = insertion order.
    memories: Vec<Memory>,
    /// Optional JSONL append-only file for restart durability.
    store_path: Option<PathBuf>,
}

impl Store {
    fn load(path: Option<PathBuf>) -> Self {
        let mut store = Self {
            memories: Vec::new(),
            store_path: path.clone(),
        };
        if let Some(p) = path {
            if p.exists() {
                if let Ok(contents) = std::fs::read_to_string(&p) {
                    let mut loaded = 0usize;
                    let mut skipped = 0usize;
                    for (i, line) in contents.lines().enumerate() {
                        if line.trim().is_empty() {
                            continue;
                        }
                        match serde_json::from_str::<Memory>(line) {
                            Ok(m) => {
                                store.memories.push(m);
                                loaded += 1;
                            }
                            Err(e) => {
                                tracing::warn!(line = i + 1, error = %e, "skip corrupt line");
                                skipped += 1;
                            }
                        }
                    }
                    tracing::info!(loaded, skipped, path = %p.display(), "restored from JSONL");
                }
            }
        }
        store
    }
}

#[tokio::main(flavor = "multi_thread", worker_threads = 2)]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    init_tracing();
    let bind = std::env::var("RUVIEW_BRAIN_BIND")
        .unwrap_or_else(|_| "0.0.0.0:9876".to_string());
    let store_path = std::env::var("RUVIEW_BRAIN_STORE_PATH")
        .ok()
        .map(PathBuf::from);
    let store = Arc::new(RwLock::new(Store::load(store_path)));

    let app = Router::new()
        .route("/memories", get(list_memories).post(post_memory))
        .route("/health", get(health))
        .with_state(store.clone());

    let listener = TcpListener::bind(&bind).await?;
    tracing::info!(
        addr = %listener.local_addr()?,
        memories = store.read().await.memories.len(),
        "ruview-mcp-brain-mini up"
    );
    axum::serve(listener, app).await?;
    Ok(())
}

async fn post_memory(
    State(store): State<Arc<RwLock<Store>>>,
    Json(body): Json<PostBody>,
) -> Result<(StatusCode, Json<Memory>), (StatusCode, Json<HashMap<String, String>>)> {
    if body.category.is_empty() || body.content.is_empty() {
        let mut err = HashMap::new();
        err.insert("error".into(), "category and content must be non-empty".into());
        return Err((StatusCode::BAD_REQUEST, Json(err)));
    }
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);

    let content_hash = {
        let mut h = Sha256::new();
        h.update(body.content.as_bytes());
        format!("{:x}", h.finalize())
    };
    // 32-char id derived from (timestamp, category, content). Stable
    // for distinct inputs, collision-resistant for the cluster's
    // post rate.
    let id = {
        let mut h = Sha256::new();
        h.update(now.to_be_bytes());
        h.update(body.category.as_bytes());
        h.update(body.content.as_bytes());
        let hex = format!("{:x}", h.finalize());
        hex.chars().take(32).collect::<String>()
    };

    let memory = Memory {
        id,
        category: body.category,
        content: body.content,
        content_hash,
        created_at: now,
    };

    let mut g = store.write().await;
    if let Some(path) = &g.store_path {
        if let Ok(line) = serde_json::to_string(&memory) {
            // Best-effort append; ignore I/O errors (POST should not fail
            // because the disk hiccupped).
            let _ = append_line(path, &line);
        }
    }
    g.memories.push(memory.clone());
    drop(g);
    tracing::debug!(category = %memory.category, "POST /memories ok");
    Ok((StatusCode::CREATED, Json(memory)))
}

fn append_line(path: &std::path::Path, line: &str) -> std::io::Result<()> {
    use std::io::Write;
    let mut f = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)?;
    writeln!(f, "{line}")
}

async fn list_memories(
    State(store): State<Arc<RwLock<Store>>>,
    Query(q): Query<ListQuery>,
) -> Json<serde_json::Value> {
    let g = store.read().await;
    let limit = q.limit.unwrap_or(50).min(1000);
    let offset = q.offset.unwrap_or(0);
    let filtered: Vec<&Memory> = g
        .memories
        .iter()
        .rev()
        .filter(|m| q.category.as_ref().map_or(true, |c| &m.category == c))
        .skip(offset)
        .take(limit)
        .collect();
    Json(serde_json::json!({
        "count": filtered.len(),
        "total": g.memories.len(),
        "offset": offset,
        "memories": filtered,
    }))
}

async fn health() -> impl IntoResponse {
    (StatusCode::OK, "ok")
}

fn init_tracing() {
    let filter = EnvFilter::try_from_env("RUVIEW_BRAIN_LOG")
        .or_else(|_| EnvFilter::try_new("info"))
        .expect("default tracing filter");
    tracing_subscriber::fmt()
        .with_env_filter(filter)
        .with_target(true)
        .with_writer(std::io::stderr)
        .init();
}
