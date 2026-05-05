//! Brain POST shim — fan vital-sign summaries to the cognitum-v0
//! brain at `http://cognitum-v0:9876/memories`.
//!
//! Reuses RuView's `brain_bridge.rs` shape (`{category, content}`)
//! verbatim — no new schema. ADR-183 §"Open questions" #2.

use std::sync::atomic::Ordering;
use std::sync::Arc;
use std::time::Duration;

use serde::Serialize;

use crate::error::Result;
use crate::state::WorkerState;
use crate::types::{VitalReading, VitalStatus};

/// JSON body POSTed to `/memories`.
#[derive(Debug, Clone, Serialize)]
pub struct MemoryPost {
    pub category: String,
    pub content: String,
}

/// Reqwest-backed client. Cheap to clone (`reqwest::Client` is `Arc`-
/// like internally).
#[derive(Debug, Clone)]
pub struct BrainClient {
    http: reqwest::Client,
    base_url: String,
    node_name: String,
}

impl BrainClient {
    pub fn new(base_url: String, node_name: String) -> Result<Self> {
        let http = reqwest::Client::builder()
            .timeout(Duration::from_secs(5))
            .user_agent(concat!(
                "ruview-vitals-worker/",
                env!("CARGO_PKG_VERSION")
            ))
            .build()?;
        Ok(Self {
            http,
            base_url,
            node_name,
        })
    }

    #[must_use]
    pub fn base_url(&self) -> &str {
        &self.base_url
    }

    #[must_use]
    pub fn node_name(&self) -> &str {
        &self.node_name
    }

    /// POST `{category, content}` to `<base_url>/memories`. 5 s
    /// timeout; surfaces non-2xx responses as [`crate::Error::Http`].
    pub async fn post_memory(&self, category: &str, content: &str) -> Result<()> {
        let payload = MemoryPost {
            category: category.to_string(),
            content: content.to_string(),
        };
        self.http
            .post(format!("{}/memories", self.base_url))
            .json(&payload)
            .send()
            .await?
            .error_for_status()?;
        Ok(())
    }
}

/// Build the natural-language summary for one reading. Format mirrors
/// the iter-123 telemetry bridge's pattern so cluster-side cosine
/// search can treat both bridges' outputs uniformly.
#[must_use]
pub fn format_vitals_summary(reading: &VitalReading, node_name: &str) -> String {
    if reading.status == VitalStatus::Unavailable {
        return format!(
            "wifi vitals node {} on {}: pipeline warmup or no signal (status unavailable, snr {:.1} dB)",
            reading.node_id, node_name, reading.snr_db
        );
    }
    format!(
        "wifi vitals node {} on {}: breathing {:.1} bpm (conf {:.0}%) heart rate {:.1} bpm \
         (conf {:.0}%) snr {:.1} dB status {}",
        reading.node_id,
        node_name,
        reading.breathing.value_bpm,
        reading.breathing.confidence * 100.0,
        reading.heart_rate.value_bpm,
        reading.heart_rate.confidence * 100.0,
        reading.snr_db,
        status_label(reading.status),
    )
}

/// Lowercase, hyphen-free label — keeps the embed text stable.
const fn status_label(s: VitalStatus) -> &'static str {
    match s {
        VitalStatus::Valid => "valid",
        VitalStatus::Degraded => "degraded",
        VitalStatus::Unreliable => "unreliable",
        VitalStatus::Unavailable => "unavailable",
    }
}

/// Periodic loop: every `interval`, snapshot the latest readings and
/// POST a memory per node. Runs until cancelled (i.e. forever for the
/// worker; used as `tokio::spawn(run_brain_loop(...))`).
///
/// The loop never panics — POST failures are counted in
/// [`crate::state::WorkerStats::brain_posts_failed`] and surfaced via
/// `GetStats`.
pub async fn run_brain_loop(client: BrainClient, state: Arc<WorkerState>, interval: Duration) {
    tracing::info!(
        url = client.base_url(),
        node = client.node_name(),
        interval_secs = interval.as_secs(),
        "brain loop starting"
    );
    let mut tick = tokio::time::interval(interval);
    // Skip the immediate first tick — let the pipeline collect at
    // least one full window before we POST.
    tick.tick().await;

    loop {
        tick.tick().await;
        let readings = state.latest_snapshot().await;
        tracing::debug!(
            count = readings.len(),
            "brain tick: snapshotting latest readings"
        );
        if readings.is_empty() {
            continue;
        }
        for reading in readings {
            let summary = format_vitals_summary(&reading, &state.config.node_name);
            match client.post_memory("spatial-vitals", &summary).await {
                Ok(()) => {
                    state.stats.brain_posts_ok.fetch_add(1, Ordering::Relaxed);
                    tracing::info!(
                        node_id = reading.node_id,
                        breathing_bpm = reading.breathing.value_bpm,
                        heart_rate_bpm = reading.heart_rate.value_bpm,
                        "POST /memories ok"
                    );
                }
                Err(e) => {
                    state.stats.brain_posts_failed.fetch_add(1, Ordering::Relaxed);
                    tracing::warn!(error = %e, node_id = reading.node_id, "POST /memories failed");
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pipeline::unavailable_reading;
    use crate::types::{VitalEstimate, VitalStatus};

    #[test]
    fn unavailable_reading_summary_mentions_warmup() {
        let r = unavailable_reading(5, 0);
        let s = format_vitals_summary(&r, "cognitum-cluster-1");
        assert!(s.contains("warmup"));
        assert!(s.contains("node 5"));
        assert!(s.contains("cognitum-cluster-1"));
    }

    #[test]
    fn valid_reading_summary_includes_bpm_and_status() {
        let r = VitalReading {
            node_id: 9,
            timestamp_us: 0,
            breathing: VitalEstimate {
                value_bpm: 14.5,
                confidence: 0.85,
                status: VitalStatus::Valid,
            },
            heart_rate: VitalEstimate {
                value_bpm: 72.0,
                confidence: 0.7,
                status: VitalStatus::Valid,
            },
            snr_db: 32.0,
            subcarrier_count: 56,
            window_frames: 900,
            status: VitalStatus::Valid,
        };
        let s = format_vitals_summary(&r, "cognitum-cluster-2");
        assert!(s.contains("breathing 14.5 bpm"));
        assert!(s.contains("heart rate 72.0 bpm"));
        assert!(s.contains("snr 32.0 dB"));
        assert!(s.contains("status valid"));
        assert!(s.contains("conf 85%"));
    }

    #[test]
    fn memory_post_serialises() {
        let p = MemoryPost {
            category: "spatial-vitals".into(),
            content: "test".into(),
        };
        let json = serde_json::to_string(&p).unwrap();
        assert!(json.contains("\"category\":\"spatial-vitals\""));
        assert!(json.contains("\"content\":\"test\""));
    }
}
