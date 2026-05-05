//! `ruview-cluster-sdk` — gRPC client for the cognitum ruview-vitals-worker
//! cluster (ADR-183). Provides typed access to all four nodes' vitals streams
//! with concurrent fan-out and health aggregation.
//!
//! ## Quick start
//!
//! ```no_run
//! use ruview_cluster_sdk::{ClusterClient, NodeAddr};
//!
//! # tokio_test::block_on(async {
//! let nodes = vec![
//!     NodeAddr::new("cognitum-cluster-1", "http://100.80.54.16:50055"),
//!     NodeAddr::new("cognitum-cluster-2", "http://100.77.220.24:50055"),
//!     NodeAddr::new("cognitum-cluster-3", "http://100.73.75.53:50055"),
//!     NodeAddr::new("cognitum-v0",        "http://100.77.59.83:50054"),
//! ];
//! let client = ClusterClient::new(nodes);
//! let snapshot = client.snapshot().await.unwrap();
//! for (name, reading) in &snapshot.readings {
//!     println!("{name}: breathing {:.1} bpm", reading.breathing.as_ref().map_or(0.0, |e| e.value_bpm));
//! }
//! # });
//! ```

pub mod client;
pub mod cluster;
pub mod error;

pub use client::VitalsClient;
pub use cluster::{ClusterClient, ClusterSnapshot, NodeAddr, NodeHealth};
pub use error::{Error, Result};

/// Generated tonic stubs (client-side only; server disabled in build.rs).
pub mod proto {
    tonic::include_proto!("cognitum.ruview.vitals.v1");
}
