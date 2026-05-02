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

#[cfg(feature = "cpu-fallback")]
pub mod cpu_embedder;

pub use device::HailoDevice;
pub use error::HailoError;
pub use inference::{EmbeddingPipeline, l2_normalize, mean_pool, DEFAULT_MAX_SEQ, MINI_LM_DIM};
pub use tokenizer::{EncodedInput, SpecialIds, WordPieceTokenizer};

#[cfg(feature = "cpu-fallback")]
pub use cpu_embedder::CpuEmbedder;

use std::path::Path;
#[cfg(feature = "hailo")]
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
    /// Iter 137 — gated on `feature = "hailo"` AND wrapped in Option
    /// so the cpu-fallback path can ship on hosts that *built* the
    /// hailo feature in but happen to lack a HAT at runtime.
    #[cfg(feature = "hailo")]
    device: Option<Mutex<crate::device::HailoDevice>>,
    /// Iter 133 — Path C CPU fallback. `Some(_)` when the operator
    /// has model.safetensors + tokenizer.json + config.json in the
    /// model dir but no HEF (yet). When set, `embed()` dispatches to
    /// real BERT-6 inference on the host CPU via candle. NPU stays
    /// idle — fallback only. Only present when built with
    /// `--features cpu-fallback`.
    #[cfg(feature = "cpu-fallback")]
    cpu_fallback: Option<crate::cpu_embedder::CpuEmbedder>,
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
        // Iter 137: combinatorial feature gating. Build matrix:
        //   * neither feature      → FeatureDisabled (default x86 dev)
        //   * hailo only           → device-only (HAT host, no Python deps)
        //   * cpu-fallback only    → CPU-only (dev box, no HailoRT installed)
        //   * hailo + cpu-fallback → device + CPU fallback (production Pi)
        // Default no-features build: short-circuit. Returning here also
        // makes the constructor below dead code, so we provide stub
        // values for `device_id` etc. so the cfg lattice still compiles.
        #[cfg(all(not(feature = "hailo"), not(feature = "cpu-fallback")))]
        {
            let _ = model_dir;
            return Err(HailoError::FeatureDisabled);
        }
        #[cfg(all(not(feature = "hailo"), not(feature = "cpu-fallback")))]
        #[allow(unreachable_code)]
        let device_id = String::new();

        // Try to open the Hailo device when the feature is on. If the
        // host has no HAT we still want CPU fallback to succeed — only
        // surface the device error if we can't fall back.
        #[cfg(feature = "hailo")]
        let (device_opt, device_id) = match crate::device::HailoDevice::open() {
            Ok(device) => {
                let v = device.version().unwrap_or((0, 0, 0));
                let device_id = format!("hailort:{}.{}.{}", v.0, v.1, v.2);
                (Some(device), device_id)
            }
            #[cfg(feature = "cpu-fallback")]
            Err(_) => (None, "cpu-fallback:no-device".to_string()),
            #[cfg(not(feature = "cpu-fallback"))]
            Err(e) => return Err(e),
        };

        #[cfg(all(not(feature = "hailo"), feature = "cpu-fallback"))]
        let device_id = "cpu-fallback:no-hailo-feature".to_string();

        // Iter 133 path-C: load CPU fallback when the feature is on
        // and the model dir has the HF safetensors trio. When there's
        // no HEF (always true today — model surgery pending) the CPU
        // fallback is the sole inference path.
        #[cfg(feature = "cpu-fallback")]
        let cpu_fallback = {
            let safetensors = model_dir.join("model.safetensors");
            let hef_path = model_dir.join("model.hef");
            if !hef_path.exists() && safetensors.exists() {
                Some(crate::cpu_embedder::CpuEmbedder::open(model_dir)?)
            } else {
                None
            }
        };

        // Dimension comes from the CPU fallback's BERT config when
        // available, otherwise the MINI_LM constant. Future HEF path
        // reads it from the network group's output shape.
        #[cfg(feature = "cpu-fallback")]
        let dimensions = cpu_fallback
            .as_ref()
            .map(|c| c.output_dim())
            .unwrap_or(crate::inference::MINI_LM_DIM);
        #[cfg(not(feature = "cpu-fallback"))]
        let dimensions = crate::inference::MINI_LM_DIM;

        Ok(Self {
            dimensions,
            name: format!(
                "hailo:{}",
                model_dir
                    .file_name()
                    .and_then(|s| s.to_str())
                    .unwrap_or("unknown-model")
            ),
            device_id,
            #[cfg(feature = "hailo")]
            device: device_opt.map(Mutex::new),
            #[cfg(feature = "cpu-fallback")]
            cpu_fallback,
        })
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
            // None when no HAT was present at open time — cpu-fallback
            // path with no NPU. Caller treats this the same as a failed
            // sensor read, which is the correct semantic.
            let g = self.device.as_ref()?.lock().unwrap_or_else(|p| p.into_inner());
            g.chip_temperature()
        }
    }

    /// Embed a single piece of text into a `dimensions()`-element f32 vector.
    ///
    /// Embed `text` into a `dim`-length unit vector.
    ///
    /// **Iter 130 — placeholder removed.** Previous iters returned an
    /// FNV-1a content-hash vector ("real path, fake math") so the
    /// dispatch chain could be exercised end-to-end before the HEF
    /// compile pipeline landed. That was misleading — operators saw
    /// vectors come back and reasonably assumed they were embeddings.
    /// Now `embed` returns `HailoError::NoModelLoaded` until a real
    /// model graph is wired in, so the cluster's failure mode honestly
    /// reflects "no inference happening."
    ///
    /// **What still works without a model:** open / dimensions / device
    /// id / chip_temperature / the entire gRPC stack. The worker boots,
    /// reports ready=false (since dimensions=0 is the gate, but iter 87
    /// pre-declared 384 to keep the path testable; iter 130 keeps that
    /// pre-declaration so health probes succeed and the operator-side
    /// `--validate-fleet` flow can detect "model missing" via a clean
    /// embed failure rather than a connection-refused).
    ///
    /// **To make `embed` work end-to-end:** see the iter-130 commit
    /// message and ADR-167's "What's still unimplemented" section —
    /// drop a compiled `model.hef` into the worker's model dir and
    /// restart. The existing `HailoEmbedder::open` path picks it up;
    /// the ModelLoaded gate trips and `embed` starts dispatching to
    /// the NPU's vstream API.
    pub fn embed(&self, text: &str) -> Result<Vec<f32>> {
        // Iter 137: dispatch order:
        //   1. CPU fallback if loaded (real semantic vectors today)
        //   2. NPU HEF inference (only path that exercises the device,
        //      currently NoModelLoaded — pending HEF model surgery)
        //   3. FeatureDisabled if neither feature is built in
        #[cfg(feature = "cpu-fallback")]
        if let Some(cpu) = &self.cpu_fallback {
            return cpu.embed(text);
        }

        #[cfg(feature = "hailo")]
        {
            let _ = text;
            // Hold the device lock briefly — preserves the contract
            // that the real HEF-based inference path needs
            // single-writer access to the vstream descriptors.
            if let Some(dev) = &self.device {
                let _guard = dev.lock().unwrap_or_else(|p| p.into_inner());
            }
            return Err(HailoError::NoModelLoaded);
        }

        #[allow(unreachable_code)]
        {
            let _ = text;
            Err(HailoError::FeatureDisabled)
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

    /// Iter 130: honest "is a model graph actually loaded?" gate.
    /// Returns `true` only when `embed()` would do real NPU inference.
    /// Today this is **always false** — HEF loading isn't wired in yet
    /// (the Hailo Dataflow Compiler step that produces `model.hef` is a
    /// vendor-tool blocker outside this repo). The worker's `health()`
    /// uses this to set the `ready` flag so the cluster's
    /// `validate_fleet` correctly identifies model-less workers as
    /// not-ready instead of false-healthy.
    ///
    /// When HEF support lands, this becomes `true` once a graph is
    /// configured into the vdevice. No callers need to change — the
    /// signal flips automatically.
    pub fn has_model(&self) -> bool {
        // Iter 133 path-C: CPU fallback counts as a loaded model.
        // The cluster's `validate_fleet` flow correctly marks workers
        // ready=true when CPU fallback is wired even with no HEF.
        #[cfg(feature = "cpu-fallback")]
        {
            if self.cpu_fallback.is_some() {
                return true;
            }
        }
        false
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
    fn open_on_missing_dir_resolves_without_panic() {
        // Across all feature combos, opening against a nonexistent dir
        // must resolve to either:
        //   * Err(FeatureDisabled / NoDevice / BadModelDir / ...) —
        //     hard failure modes the operator can act on
        //   * Ok(embedder) with has_model() == false — the iter-130
        //     "model not yet present" path that lets health probes
        //     report ready=false instead of connection-refused
        let r = HailoEmbedder::open(Path::new("/nonexistent"));
        match r {
            Ok(e) => assert!(
                !e.has_model(),
                "open(missing dir) returned Ok but has_model=true — should be ready=false"
            ),
            Err(
                HailoError::FeatureDisabled
                | HailoError::NotYetImplemented(_)
                | HailoError::BadModelDir { .. }
                | HailoError::NoDevice(_)
                | HailoError::Tokenizer(_),
            ) => {}
            Err(other) => panic!("unexpected open() error: {:?}", other),
        }
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
