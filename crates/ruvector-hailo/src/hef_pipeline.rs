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

        // Iter 173 — security defense in depth: verify the HEF magic
        // before handing the bytes to libhailort. The Hailo HEF format
        // starts with `0x01 0x48 0x45 0x46` (`\x01HEF`). Catches:
        //   * accidental file corruption / truncation
        //   * wrong-file mistakes (operator drops a .onnx where .hef
        //     was expected)
        //   * targeted substitution with a non-HEF payload
        // Costs ~4 bytes of read + a memcmp; sub-microsecond at boot.
        // The iter-143 fingerprint is the cluster-wide drift gate; this
        // is the per-worker "is this even a HEF" gate.
        //
        // Iter 174 — opt-in pre-pinned sha256 verification. Operators
        // set RUVECTOR_HEF_SHA256 in the env file (e.g. to the published
        // GitHub Release sha256 from iter 169). At boot, before passing
        // the path to libhailort, we hash the file and compare. Catches
        // a substituted HEF that satisfies the magic check but isn't
        // the artifact the operator expected to deploy.
        // sha256 on Pi 5 NEON ~1 GB/s; 15.7 MB HEF costs ~16 ms — well
        // within the iter-173 ~1s boot budget. Skipped when env var is
        // unset (back-compat: default deploys retain iter-173 behavior).
        const HEF_MAGIC: [u8; 4] = [0x01, b'H', b'E', b'F'];
        let pinned_sha256 = std::env::var("RUVECTOR_HEF_SHA256").ok();
        let mut header = [0u8; 4];
        use std::io::Read as _;
        let mut f = std::fs::File::open(hef_path).map_err(|e| {
            HailoError::Tokenizer(format!("open HEF: {}", e))
        })?;
        f.read_exact(&mut header).map_err(|e| {
            HailoError::Tokenizer(format!("read HEF header: {}", e))
        })?;
        if header != HEF_MAGIC {
            return Err(HailoError::BadModelDir {
                path: hef_path.display().to_string(),
                what: "model.hef magic mismatch — not a Hailo HEF",
            });
        }
        if let Some(want) = pinned_sha256 {
            // Re-open + stream-hash the whole file. We don't keep the
            // 15.7 MB in RAM — sha2 is updated chunk-by-chunk.
            use sha2::{Digest, Sha256};
            let mut f2 = std::fs::File::open(hef_path).map_err(|e| {
                HailoError::Tokenizer(format!("open HEF for sha256: {}", e))
            })?;
            let mut h = Sha256::new();
            let mut buf = [0u8; 64 * 1024];
            loop {
                let n = f2.read(&mut buf).map_err(|e| {
                    HailoError::Tokenizer(format!("read HEF for sha256: {}", e))
                })?;
                if n == 0 {
                    break;
                }
                h.update(&buf[..n]);
            }
            let got = format!("{:x}", h.finalize());
            let want_norm = want.trim().to_lowercase();
            if got != want_norm {
                return Err(HailoError::BadModelDir {
                    path: hef_path.display().to_string(),
                    what: "model.hef sha256 mismatch — RUVECTOR_HEF_SHA256 pin failed",
                });
            }
        }
        drop(f);

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
        Self::open_inner(device, hef, hef_path).inspect_err(|_| {
            // SAFETY: `hef` was returned by hailo_create_hef_file
            // and hasn't been transferred elsewhere yet.
            unsafe {
                hailort_sys::hailo_release_hef(hef);
            }
        })
    }

    fn open_inner(
        device: &HailoDevice,
        hef: hailort_sys::hailo_hef,
        _hef_path: &Path,
    ) -> Result<Self, HailoError> {
        let vdevice = device.raw_vdevice();

        // 1. Init default configure params for this HEF + vdevice.
        // SAFETY: hef + vdevice are valid handles; the SDK writes
        // through `&mut params`.
        let mut params: hailort_sys::hailo_configure_params_t =
            unsafe { std::mem::zeroed() };
        let status = unsafe {
            hailort_sys::hailo_init_configure_params_by_vdevice(
                hef,
                vdevice,
                &mut params as *mut _,
            )
        };
        if status != 0 {
            return Err(HailoError::Hailort {
                status: status as i32,
                where_: "hailo_init_configure_params_by_vdevice",
            });
        }

        // 2. Configure the vdevice with this HEF. Iter-156b's HEF
        // contains exactly one network group; n_ng >1 would mean a
        // different HEF and we surface the mismatch as an error.
        let mut n_ng: usize = 1;
        let mut network_group: hailort_sys::hailo_configured_network_group =
            ptr::null_mut();
        let status = unsafe {
            hailort_sys::hailo_configure_vdevice(
                vdevice,
                hef,
                &mut params as *mut _,
                &mut network_group as *mut _,
                &mut n_ng as *mut _,
            )
        };
        if status != 0 {
            return Err(HailoError::Hailort {
                status: status as i32,
                where_: "hailo_configure_vdevice",
            });
        }
        if n_ng != 1 {
            return Err(HailoError::Hailort {
                status: -1,
                where_: "hailo_configure_vdevice — expected 1 network group",
            });
        }

        // 3. Build input vstream params, format=FLOAT32 so HailoRT
        // does the quantize for us. iter-156b HEF has one input.
        let mut input_count: usize = 1;
        let mut input_params: hailort_sys::hailo_input_vstream_params_by_name_t =
            unsafe { std::mem::zeroed() };
        let status = unsafe {
            hailort_sys::hailo_make_input_vstream_params(
                network_group,
                false,
                hailort_sys::hailo_format_type_t_HAILO_FORMAT_TYPE_FLOAT32,
                &mut input_params as *mut _,
                &mut input_count as *mut _,
            )
        };
        if status != 0 {
            return Err(HailoError::Hailort {
                status: status as i32,
                where_: "hailo_make_input_vstream_params",
            });
        }
        if input_count != 1 {
            return Err(HailoError::Hailort {
                status: -1,
                where_: "expected 1 input vstream",
            });
        }

        // 4. Create the input vstream from the params.
        let mut input_vstream: hailort_sys::hailo_input_vstream =
            ptr::null_mut();
        let status = unsafe {
            hailort_sys::hailo_create_input_vstreams(
                network_group,
                &input_params as *const _,
                1,
                &mut input_vstream as *mut _,
            )
        };
        if status != 0 {
            return Err(HailoError::Hailort {
                status: status as i32,
                where_: "hailo_create_input_vstreams",
            });
        }

        // 5. Same for output vstream.
        let mut output_count: usize = 1;
        let mut output_params: hailort_sys::hailo_output_vstream_params_by_name_t =
            unsafe { std::mem::zeroed() };
        let status = unsafe {
            hailort_sys::hailo_make_output_vstream_params(
                network_group,
                false,
                hailort_sys::hailo_format_type_t_HAILO_FORMAT_TYPE_FLOAT32,
                &mut output_params as *mut _,
                &mut output_count as *mut _,
            )
        };
        if status != 0 {
            return Err(HailoError::Hailort {
                status: status as i32,
                where_: "hailo_make_output_vstream_params",
            });
        }

        let mut output_vstream: hailort_sys::hailo_output_vstream =
            ptr::null_mut();
        let status = unsafe {
            hailort_sys::hailo_create_output_vstreams(
                network_group,
                &output_params as *const _,
                1,
                &mut output_vstream as *mut _,
            )
        };
        if status != 0 {
            return Err(HailoError::Hailort {
                status: status as i32,
                where_: "hailo_create_output_vstreams",
            });
        }

        // 6. Read vstream metadata for shape + quant. We use FLOAT32
        // format so HailoRT does quant for us; we keep the quant info
        // for diagnostics only.
        let mut input_info: hailort_sys::hailo_vstream_info_t =
            unsafe { std::mem::zeroed() };
        let status = unsafe {
            hailort_sys::hailo_get_input_vstream_info(
                input_vstream,
                &mut input_info as *mut _,
            )
        };
        if status != 0 {
            return Err(HailoError::Hailort {
                status: status as i32,
                where_: "hailo_get_input_vstream_info",
            });
        }
        let mut output_info: hailort_sys::hailo_vstream_info_t =
            unsafe { std::mem::zeroed() };
        let status = unsafe {
            hailort_sys::hailo_get_output_vstream_info(
                output_vstream,
                &mut output_info as *mut _,
            )
        };
        if status != 0 {
            return Err(HailoError::Hailort {
                status: status as i32,
                where_: "hailo_get_output_vstream_info",
            });
        }

        // SAFETY: HEF compiled with rank-3 inputs, so the union holds
        // a `shape: hailo_3d_image_shape_t`. NMS shape doesn't apply.
        let in_shape = unsafe { input_info.__bindgen_anon_1.shape };
        let out_shape = unsafe { output_info.__bindgen_anon_1.shape };

        // Logical [batch=1, seq=128, hidden=384] maps to
        // (height=1, width=128, features=384) for our HEF. Buffer is
        // row-major over h×w×f. We use max(height, width) since the
        // mapping isn't strict — Hailo can route either axis to the
        // longer one based on its placement decisions.
        let input_shape = [
            1usize,
            in_shape.height.max(in_shape.width) as usize,
            in_shape.features as usize,
        ];
        let output_shape = [
            1usize,
            out_shape.height.max(out_shape.width) as usize,
            out_shape.features as usize,
        ];

        // FP32 frame size = sum of dims * 4 bytes. The vstream API
        // also exposes `hailo_get_input_vstream_frame_size` if we
        // want HailoRT to compute it; using the shape is equivalent
        // and avoids one more FFI hop.
        let input_frame_bytes =
            input_shape[0] * input_shape[1] * input_shape[2] * 4;
        let output_frame_bytes =
            output_shape[0] * output_shape[1] * output_shape[2] * 4;

        let input_quant = QuantInfo {
            scale: input_info.quant_info.qp_scale as f32,
            zero_point: input_info.quant_info.qp_zp as f32,
        };
        let output_quant = QuantInfo {
            scale: output_info.quant_info.qp_scale as f32,
            zero_point: output_info.quant_info.qp_zp as f32,
        };

        Ok(Self {
            hef,
            network_group,
            input_vstream,
            output_vstream,
            input_quant,
            output_quant,
            input_shape,
            output_shape,
            input_frame_bytes,
            output_frame_bytes,
        })
    }

    /// FP32 forward pass. Takes a flat `[batch * seq * hidden]` input
    /// in row-major order, returns the same shape post-encoder.
    ///
    /// HailoRT does the FP32 → INT8 quantize on write and INT8 → FP32
    /// dequantize on read because we configured both vstreams with
    /// `HAILO_FORMAT_TYPE_FLOAT32`. We pass FP32 bytes in, get FP32
    /// bytes out.
    pub fn forward(&mut self, input: &[f32]) -> Result<Vec<f32>, HailoError> {
        let expected_floats = self.input_frame_bytes / 4;
        if input.len() != expected_floats {
            return Err(HailoError::Shape {
                expected: expected_floats,
                actual: input.len(),
            });
        }

        // Push the FP32 input. HailoRT internally quantizes to UINT8
        // using the embedded scale + zero-point from the HEF.
        // SAFETY: input.as_ptr() points at input.len() * 4 valid bytes.
        let status = unsafe {
            hailort_sys::hailo_vstream_write_raw_buffer(
                self.input_vstream,
                input.as_ptr() as *const std::ffi::c_void,
                self.input_frame_bytes,
            )
        };
        if status != 0 {
            return Err(HailoError::Hailort {
                status: status as i32,
                where_: "hailo_vstream_write_raw_buffer",
            });
        }

        // Pull the FP32 output. HailoRT dequantizes for us.
        let mut out = vec![0.0f32; self.output_frame_bytes / 4];
        // SAFETY: out.as_mut_ptr() points at out.len() * 4 writable bytes.
        let status = unsafe {
            hailort_sys::hailo_vstream_read_raw_buffer(
                self.output_vstream,
                out.as_mut_ptr() as *mut std::ffi::c_void,
                self.output_frame_bytes,
            )
        };
        if status != 0 {
            return Err(HailoError::Hailort {
                status: status as i32,
                where_: "hailo_vstream_read_raw_buffer",
            });
        }

        Ok(out)
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
        // the HEF (the configured network group is owned by the
        // vdevice and released when the vdevice is — HailoRT C API
        // doesn't expose a separate release for it).
        unsafe {
            if !self.input_vstream.is_null() {
                hailort_sys::hailo_release_input_vstreams(
                    &mut self.input_vstream as *mut _,
                    1,
                );
            }
            if !self.output_vstream.is_null() {
                hailort_sys::hailo_release_output_vstreams(
                    &mut self.output_vstream as *mut _,
                    1,
                );
            }
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
