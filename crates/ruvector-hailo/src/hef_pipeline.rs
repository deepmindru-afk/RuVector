//! HEF inference pipeline — push embeddings through the Hailo-8 NPU.
//!
//! ADR-176 P1 (`hailo-backend`, iter 158-159). Reads a compiled
//! `model.hef` (produced by `deploy/compile-encoder-hef.py`),
//! configures it on an existing `HailoDevice` vdevice, opens
//! input + output vstreams, and exposes a `forward()` that takes
//! FP32 `[1, seq, hidden]` embeddings and returns FP32
//! `[1, seq, hidden]` post-encoder hidden states.
//!
//! The HEF compiled in iter 156b has these vstream shapes:
//!
//! ```text
//!   Input  minilm_encoder/input_layer1   UINT8, FCR(1x128x384)
//!   Output minilm_encoder/normalization12 UINT8, FCR(1x128x384)
//! ```
//!
//! Quantization scale + zero-point come from `hailo_vstream_info_t`
//! at HEF-load time. We dequantize on read so callers see FP32.
//!
//! **Phase boundary**: this module owns NPU forward pass only. The
//! tokenize → host-side embedding lookup → NPU forward → mean-pool →
//! L2-normalize chain lives in `HefEmbedder` (P3, iter 161).

#![cfg(feature = "hailo")]
// Iter 158 scaffold: several fields are populated by iter-159's
// open_inner + forward bodies. Dead-code warnings would mask
// real-progress signals during the EPIC roll-out.
#![allow(dead_code)]

use crate::device::HailoDevice;
use crate::error::HailoError;
use std::path::Path;
use std::ptr;

/// Quantization parameters for an INT8/UINT8 vstream tensor.
#[derive(Clone, Copy, Debug)]
pub struct QuantInfo {
    /// `dequantized = scale * (raw - zero_point)`.
    pub scale: f32,
    pub zero_point: f32,
}

/// HEF-driven NPU forward pass for the all-MiniLM-L6-v2 encoder.
///
/// Held by `HailoEmbedder` when `model.hef` exists in the model dir.
/// Single-input, single-output: input is the post-embedding hidden
/// states (host-computed via candle's `BertEmbeddings`); output is
/// `last_hidden_state` (the encoder's final LayerNorm output).
///
/// **Lifetime contract**: the underlying HailoRT handles
/// (`hailo_hef`, configured network group, vstreams) are released in
/// the `Drop` impl in this order: vstreams → network group → HEF.
/// Reverse-order release is what the C API expects.
pub struct HefPipeline {
    /// Loaded HEF artifact. Owned; released on drop.
    hef: hailort_sys::hailo_hef,
    /// Configured network group bound to a vdevice. The vdevice itself
    /// is owned by `HailoDevice` higher up the call stack, not here.
    network_group: hailort_sys::hailo_configured_network_group,
    /// Single input vstream (`hidden_states`). UINT8 over the wire.
    input_vstream: hailort_sys::hailo_input_vstream,
    /// Single output vstream (`last_hidden_state`). UINT8 over the wire.
    output_vstream: hailort_sys::hailo_output_vstream,

    /// Quant for the input — host computes float embeddings then we
    /// quantize before `vstream_write`.
    input_quant: QuantInfo,
    /// Quant for the output — NPU returns UINT8 then we dequantize
    /// back to FP32 for the host-side mean-pool.
    output_quant: QuantInfo,

    /// Logical input shape `[batch, seq, hidden]`. Iter 156b: `[1, 128, 384]`.
    input_shape: [usize; 3],
    /// Logical output shape `[batch, seq, hidden]`. Iter 156b: `[1, 128, 384]`.
    output_shape: [usize; 3],

    /// Raw input buffer size in bytes (UINT8). Cached so `forward()`
    /// doesn't recompute per call.
    input_frame_bytes: usize,
    /// Raw output buffer size in bytes.
    output_frame_bytes: usize,
}

impl HefPipeline {
    /// Open `hef_path` and configure it onto `device`'s vdevice.
    ///
    /// The HEF must contain exactly one network group with exactly
    /// one input and one output vstream — the iter-156b compile
    /// produces this shape. Multi-network HEFs are out of scope for
    /// this iteration.
    pub fn open(
        device: &HailoDevice,
        hef_path: &Path,
    ) -> Result<Self, HailoError> {
        let path_c = std::ffi::CString::new(
            hef_path
                .to_str()
                .ok_or_else(|| HailoError::BadModelDir {
                    path: hef_path.display().to_string(),
                    what: "non-UTF8 HEF path",
                })?,
        )
        .map_err(|_| HailoError::BadModelDir {
            path: hef_path.display().to_string(),
            what: "HEF path contains nul byte",
        })?;

        // 1. Load HEF from disk.
        let mut hef: hailort_sys::hailo_hef = ptr::null_mut();
        // SAFETY: path is valid CString; HailoRT writes through `&mut hef`.
        let status = unsafe {
            hailort_sys::hailo_create_hef_file(&mut hef as *mut _, path_c.as_ptr())
        };
        if status != 0 {
            return Err(HailoError::Hailort {
                status: status as i32,
                where_: "hailo_create_hef_file",
            });
        }

        // From here on we own `hef`; release it on any error path
        // before propagating.
        let result =
            Self::open_inner(device, hef, hef_path).map_err(|e| {
                // SAFETY: `hef` was returned by hailo_create_hef_file
                // and hasn't been transferred elsewhere yet.
                unsafe {
                    hailort_sys::hailo_release_hef(hef);
                }
                e
            });

        result
    }

    fn open_inner(
        _device: &HailoDevice,
        _hef: hailort_sys::hailo_hef,
        _hef_path: &Path,
    ) -> Result<Self, HailoError> {
        // Iter 158 scaffold: HEF is loaded; the configure_vdevice +
        // vstream creation lands in iter 159. For now return a typed
        // sentinel error so calling code (HailoEmbedder::open) can
        // distinguish "HEF found but not yet wired" from "HEF missing".
        //
        // The iter-159 follow-up replaces this body with:
        //   * hailo_init_configure_params_by_vdevice
        //   * hailo_configure_vdevice → network_group
        //   * hailo_make_input_vstream_params + hailo_create_input_vstreams
        //   * hailo_make_output_vstream_params + hailo_create_output_vstreams
        //   * hailo_get_input_vstream_info / output → quant + shape
        Err(HailoError::NotYetImplemented(
            "HefPipeline::open_inner — iter 159 wires configure_vdevice + vstreams",
        ))
    }

    /// FP32 forward pass. Takes a flat `[batch * seq * hidden]` input
    /// in row-major order, returns the same shape post-encoder.
    ///
    /// Iter 159 fills this in. Iter 158 returns NotYetImplemented.
    pub fn forward(&mut self, _input: &[f32]) -> Result<Vec<f32>, HailoError> {
        Err(HailoError::NotYetImplemented(
            "HefPipeline::forward — iter 159 fills in vstream write/read + quant",
        ))
    }

    pub fn input_shape(&self) -> [usize; 3] {
        self.input_shape
    }

    pub fn output_shape(&self) -> [usize; 3] {
        self.output_shape
    }

    pub fn input_quant(&self) -> QuantInfo {
        self.input_quant
    }

    pub fn output_quant(&self) -> QuantInfo {
        self.output_quant
    }
}

impl Drop for HefPipeline {
    fn drop(&mut self) {
        // SAFETY: each handle was returned by HailoRT and hasn't been
        // released yet. Release order is reverse of acquisition:
        // vstreams first (they hold refs into the network group), then
        // the network group, then the HEF.
        unsafe {
            // Iter 159 fills in real release calls — for now the fields
            // are never populated (open_inner returns NotYetImplemented
            // before constructing Self) so Drop is a no-op.
            //
            // hailort_sys::hailo_release_input_vstreams(&mut self.input_vstream as *mut _, 1);
            // hailort_sys::hailo_release_output_vstreams(&mut self.output_vstream as *mut _, 1);
            // hailort_sys::hailo_release_configured_network_group(self.network_group);
            if !self.hef.is_null() {
                hailort_sys::hailo_release_hef(self.hef);
            }
        }
    }
}

// SAFETY: HailoRT documents handles as thread-safe for inference
// when external serialisation prevents config changes during traffic.
// `HefPipeline` is held behind `Mutex` in `HailoEmbedder`.
unsafe impl Send for HefPipeline {}
unsafe impl Sync for HefPipeline {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn open_returns_not_yet_implemented_until_iter_159() {
        // Open HailoDevice would fail without a real /dev/hailo0
        // present, so we can't even reach HefPipeline::open here on
        // a dev box. The test exists to assert the public type
        // signatures compile.
        let _ = std::mem::size_of::<HefPipeline>();
        let _ = std::mem::size_of::<QuantInfo>();
    }
}
