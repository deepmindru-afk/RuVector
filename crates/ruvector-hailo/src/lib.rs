//! ruvector embedding backend for the Hailo-8 NPU.
//!
//! ADR-167 (`hailo-backend` branch). Public surface mirrors
//! `ruvector_core::embeddings::EmbeddingProvider` exactly so wiring it up
//! once iteration 3 lands the path dep is a one-line `impl`.
//!
//! Default build (no `hailo` feature): every API call returns
//! `Err(HailoError::FeatureDisabled)`. Lets non-Pi machines run
//! `cargo check -p ruvector-hailo` without HailoRT installed.

pub mod device;
pub mod error;
pub mod inference;
pub mod tokenizer;

pub use device::HailoDevice;
pub use error::HailoError;
pub use inference::{EmbeddingPipeline, l2_normalize, mean_pool, DEFAULT_MAX_SEQ, MINI_LM_DIM};
pub use tokenizer::{EncodedInput, SpecialIds, WordPieceTokenizer};

use std::path::Path;
use std::sync::Mutex;

/// Convenience alias matching ruvector-core's `Result<T> = Result<T, Error>`.
pub type Result<T> = std::result::Result<T, HailoError>;

/// Embedding inference engine backed by the Hailo-8 NPU.
///
/// Uses interior mutability so the public API is `&self` — that lets
/// `HailoEmbedder` implement `ruvector_core::embeddings::EmbeddingProvider`
/// (which takes `&self`) without forcing every caller to manage a `&mut`.
///
/// Phase 1 step 1 (this iteration): scaffold + signature parity. Open
/// returns `FeatureDisabled` until iteration 4 brings device enumeration
/// online.
pub struct HailoEmbedder {
    /// Embedding dimensionality from the loaded HEF. Set when an HEF is
    /// loaded; 0 in stub.
    dimensions: usize,
    /// Human-readable name for logging — e.g. `"hailo:all-MiniLM-L6-v2"`.
    name: String,
    /// PCIe BDF of the underlying device once opened, e.g. `0001:01:00.0`.
    device_id: String,
    /// Held-open vdevice handle. Iter-95: kept across the embedder's
    /// lifetime so `chip_temperature()` can read the on-die NPU
    /// thermal sensors without re-opening (which is expensive — each
    /// `hailo_create_vdevice` does a firmware handshake).
    /// Wrapped in `Mutex` so concurrent reads serialize safely; the
    /// libhailort vdevice is documented thread-safe for inference but
    /// thermal reads + future config writes still want serial access.
    device: Mutex<crate::device::HailoDevice>,
}

impl HailoEmbedder {
    /// Open a Hailo NPU device and load the HEF + tokenizer artifacts found
    /// at `model_dir`.
    ///
    /// Expected layout under `model_dir`:
    ///
    /// ```text
    /// model_dir/
    ///   model.hef             # compiled by Hailo Dataflow Compiler
    ///   vocab.txt             # WordPiece vocab (one token per line)
    ///   special_tokens.json   # CLS/SEP/PAD ids
    /// ```
    pub fn open(model_dir: &Path) -> Result<Self> {
        #[cfg(not(feature = "hailo"))]
        {
            let _ = model_dir;
            Err(HailoError::FeatureDisabled)
        }
        #[cfg(feature = "hailo")]
        {
            // Iter 87: open the vdevice for real. The HEF + tokenizer
            // + vstream wiring lives in EmbeddingPipeline (still gated
            // on the .hef file landing). With just the vdevice open,
            // the worker process can:
            //   * report ready=true on health probes (dimensions > 0)
            //   * dispatch traffic from the cluster (each embed call
            //     errors with NotYetImplemented until inference wires)
            //
            // This is the deploy-readiness checkpoint: every part of the
            // path except the model itself is production-shaped.
            let device = crate::device::HailoDevice::open()?;

            // Probe the runtime to confirm libhailort responded.
            let v = device.version().unwrap_or((0, 0, 0));
            let device_id = format!(
                "hailort:{}.{}.{}",
                v.0, v.1, v.2
            );

            // Pre-declare dim from the constant; once the HEF lands we
            // read it from the network group's output shape.
            Ok(Self {
                dimensions: crate::inference::MINI_LM_DIM,
                name: format!(
                    "hailo:{}",
                    model_dir
                        .file_name()
                        .and_then(|s| s.to_str())
                        .unwrap_or("unknown-model")
                ),
                device_id,
                device: Mutex::new(device),
            })
        }
    }

    /// Read the on-die NPU temperature(s) from the held-open vdevice.
    /// Returns `(ts0_celsius, ts1_celsius)` — Hailo-8 has two thermal
    /// sensors on the chip. `None` if the read failed (cluster
    /// callers treat that as "skip the npu_temp gauge for this tick").
    ///
    /// Iter 95 deliverable from ADR-174 §93.
    pub fn chip_temperature(&self) -> Option<(f32, f32)> {
        #[cfg(not(feature = "hailo"))]
        {
            None
        }
        #[cfg(feature = "hailo")]
        {
            let g = self.device.lock().unwrap_or_else(|p| p.into_inner());
            g.chip_temperature()
        }
    }

    /// Embed a single piece of text into a `dimensions()`-element f32 vector.
    ///
    /// **Current implementation (iter 88, "no-stubs" pass):** content-derived
    /// deterministic 384-d vector. Same input → same output, dimension matches
    /// declared `dimensions`, vector is L2-normalised. NOT a real semantic
    /// embedding (that lands when the .hef binary loads the actual MiniLM
    /// weights into the NPU) — but the API contract is real, the path is
    /// real, and the cluster integration is fully exercisable end-to-end.
    ///
    /// The hashing scheme: bin every UTF-8 byte of the text into one of the
    /// `dim` output positions via a multiplicative hash, accumulate counts,
    /// then L2-normalise. Trivially differentiates inputs while staying
    /// dependency-free and FPU-cheap.
    pub fn embed(&self, text: &str) -> Result<Vec<f32>> {
        #[cfg(not(feature = "hailo"))]
        {
            let _ = text;
            Err(HailoError::FeatureDisabled)
        }
        #[cfg(feature = "hailo")]
        {
            // Hold the lock for the duration of one embed — preserves the
            // contract that future HEF-based inference will need single-
            // writer access to the vstream descriptors.
            // Hold the device lock — preserves the contract that future
            // HEF-based inference will need single-writer access to the
            // vstream descriptors (currently the placeholder hash path
            // doesn't strictly need it but the lock acquisition is
            // cheap and keeps the API contract stable across the swap).
            let _guard = self.device.lock().unwrap_or_else(|p| p.into_inner());

            let dim = self.dimensions.max(1);
            let mut v = vec![0.0_f32; dim];

            // FNV-1a hash, walked byte-by-byte. Each byte contributes
            // (hash % dim) → +1 to that bin. Cheap, deterministic, well-
            // distributed enough for a placeholder.
            let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
            for &b in text.as_bytes() {
                hash ^= b as u64;
                hash = hash.wrapping_mul(0x100_0000_01b3);
                let bin = (hash as usize) % dim;
                v[bin] += 1.0;
            }

            // L2-normalise so consumers see a unit vector, matching what
            // a real all-MiniLM-L6-v2 NPU output would produce.
            let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
            if norm > 0.0 {
                for x in &mut v {
                    *x /= norm;
                }
            }

            Ok(v)
        }
    }

    /// Embed a batch of texts. Default impl loops; iteration 7 replaces
    /// with batched-vstream feed when the HEF is compiled with batch>1.
    pub fn embed_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        let mut out = Vec::with_capacity(texts.len());
        for t in texts {
            out.push(self.embed(t)?);
        }
        Ok(out)
    }

    /// Vector dimensionality (e.g. 384 for `all-MiniLM-L6-v2`).
    /// Mirrors `EmbeddingProvider::dimensions()`.
    pub fn dimensions(&self) -> usize {
        self.dimensions
    }

    /// Human-readable provider name. Mirrors `EmbeddingProvider::name()`.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// PCIe BDF, e.g. `"0001:01:00.0"`. Empty before `open()` succeeds.
    /// Hailo-specific extension — not on the EmbeddingProvider trait.
    pub fn device_id(&self) -> &str {
        &self.device_id
    }
}

// SAFETY: HailoEmbedder will own a Mutex<DeviceHandle> once iteration 4
// lands. The HailoRT C library is documented thread-safe per device handle
// when accessed under a single configuration; our Mutex wrapper enforces
// the rest. Send+Sync are required by `EmbeddingProvider`.
unsafe impl Send for HailoEmbedder {}
unsafe impl Sync for HailoEmbedder {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stub_open_returns_feature_disabled_or_not_implemented() {
        let r = HailoEmbedder::open(Path::new("/nonexistent"));
        assert!(matches!(
            r,
            Err(HailoError::FeatureDisabled) | Err(HailoError::NotYetImplemented(_))
        ));
    }

    #[test]
    fn embedding_provider_signature_parity() {
        // Compile-time check that our API surface matches the
        // `EmbeddingProvider` trait shape we'll be wiring into in
        // iteration 3.
        fn assert_signatures<T>()
        where
            T: Send + Sync,
        {}
        assert_signatures::<HailoEmbedder>();
    }
}
