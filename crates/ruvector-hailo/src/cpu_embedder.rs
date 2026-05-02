//! Pure-Rust CPU fallback for sentence-transformers/all-MiniLM-L6-v2
//! (iter 133, ADR-167 path C).
//!
//! Runs real BERT-6 inference on the host CPU (Cortex-A76 NEON on the
//! Pi 5, AVX2 on x86 dev hosts) via candle-transformers. The Hailo NPU
//! stays idle — this is a fallback, not the primary path. Use when
//! the operator has the model weights but not (yet) a compiled HEF.
//!
//! # Artifacts expected in `model_dir`
//!
//! ```text
//!   model_dir/
//!     model.safetensors       # ~90 MB BERT-6 weights from HF
//!     tokenizer.json          # HF tokenizers JSON (not the WordPiece text vocab)
//!     config.json             # BERT config — hidden_size, layers, heads, etc.
//! ```
//!
//! These are the standard HuggingFace artifacts for
//! `sentence-transformers/all-MiniLM-L6-v2`. No HEF / Hailo Dataflow
//! Compiler dependency.
//!
//! # Realistic latency
//!
//! Single-thread BERT-6 forward on Cortex-A76 at 2.4 GHz, 128-token
//! sequence: ~50-150 ms per embed. AVX2 x86 hosts run ~10-30 ms.
//! Slow vs Hailo's 1-3 ms NPU target, but real semantic vectors today.

#![cfg(feature = "cpu-fallback")]

use crate::error::HailoError;
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::bert::{BertModel, Config};
use std::path::Path;
use std::sync::Mutex;
use tokenizers::Tokenizer;

/// CPU-side BERT-6 embedder. Held in `HailoEmbedder` as a fallback
/// when no HEF is loaded. Thread-safe: forward inference is mutexed
/// because candle's BERT impl is not Sync (holds owned Tensors).
pub struct CpuEmbedder {
    inner: Mutex<Inner>,
    output_dim: usize,
    /// 128-token sequence cap matches all-MiniLM-L6-v2's training-time
    /// max. Raising this breaks RoPE/positional baked into the weights.
    max_seq: usize,
}

struct Inner {
    model: BertModel,
    tokenizer: Tokenizer,
    device: Device,
}

impl CpuEmbedder {
    /// Load the model from `model_dir`. Errors if the three required
    /// files (model.safetensors, tokenizer.json, config.json) aren't
    /// all present + parseable.
    pub fn open(model_dir: &Path) -> Result<Self, HailoError> {
        let weights_path = model_dir.join("model.safetensors");
        let tokenizer_path = model_dir.join("tokenizer.json");
        let config_path = model_dir.join("config.json");

        if !weights_path.exists() {
            return Err(HailoError::BadModelDir {
                path: model_dir.display().to_string(),
                what: "model.safetensors",
            });
        }
        if !tokenizer_path.exists() {
            return Err(HailoError::BadModelDir {
                path: model_dir.display().to_string(),
                what: "tokenizer.json",
            });
        }
        if !config_path.exists() {
            return Err(HailoError::BadModelDir {
                path: model_dir.display().to_string(),
                what: "config.json",
            });
        }

        // CPU device — NPU is dormant in this fallback. Future iter
        // could pick up GPU via candle's CUDA/Metal backends, but the
        // Pi 5 + AI HAT+ deploy doesn't have one.
        let device = Device::Cpu;

        // BERT config drives the model topology. all-MiniLM-L6-v2
        // ships hidden_size=384, num_hidden_layers=6, num_heads=12.
        let config_str = std::fs::read_to_string(&config_path)
            .map_err(|e| HailoError::Tokenizer(format!("read config.json: {}", e)))?;
        let config: Config = serde_json::from_str(&config_str)
            .map_err(|e| HailoError::Tokenizer(format!("parse config.json: {}", e)))?;
        let output_dim = config.hidden_size;

        // Load weights via candle's safetensors mmap helper.
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&[&weights_path], DType::F32, &device)
                .map_err(|e| HailoError::Tokenizer(format!("load safetensors: {}", e)))?
        };
        let model = BertModel::load(vb, &config)
            .map_err(|e| HailoError::Tokenizer(format!("BertModel::load: {}", e)))?;

        // HF tokenizers — handles padding + truncation; faster than
        // our own WordPiece walk.
        let tokenizer = Tokenizer::from_file(&tokenizer_path)
            .map_err(|e| HailoError::Tokenizer(format!("Tokenizer::from_file: {}", e)))?;

        Ok(Self {
            inner: Mutex::new(Inner {
                model,
                tokenizer,
                device,
            }),
            output_dim,
            max_seq: 128,
        })
    }

    pub fn output_dim(&self) -> usize {
        self.output_dim
    }

    /// Embed `text` into a unit-norm `output_dim`-length f32 vector.
    /// Mean-pools the BERT-6 output across the masked sequence and
    /// L2-normalises — matches what `sentence-transformers/all-MiniLM-L6-v2`
    /// produces from its native Python pipeline.
    pub fn embed(&self, text: &str) -> Result<Vec<f32>, HailoError> {
        let mut g = self.inner.lock().unwrap_or_else(|p| p.into_inner());
        let inner = &mut *g;

        let mut encoding = inner
            .tokenizer
            .encode(text, true)
            .map_err(|e| HailoError::Tokenizer(format!("encode: {}", e)))?;
        encoding.truncate(self.max_seq, 1, tokenizers::TruncationDirection::Right);

        let token_ids: Vec<i64> = encoding.get_ids().iter().map(|&x| x as i64).collect();
        let attention: Vec<i64> = encoding
            .get_attention_mask()
            .iter()
            .map(|&x| x as i64)
            .collect();

        let token_t = Tensor::new(token_ids.as_slice(), &inner.device)
            .map_err(|e| HailoError::Tokenizer(format!("token tensor: {}", e)))?
            .unsqueeze(0)
            .map_err(|e| HailoError::Tokenizer(format!("token unsqueeze: {}", e)))?;
        let token_type = Tensor::zeros((1, token_ids.len()), DType::I64, &inner.device)
            .map_err(|e| HailoError::Tokenizer(format!("token type: {}", e)))?;
        let attention_t = Tensor::new(attention.as_slice(), &inner.device)
            .map_err(|e| HailoError::Tokenizer(format!("attention tensor: {}", e)))?
            .unsqueeze(0)
            .map_err(|e| HailoError::Tokenizer(format!("attention unsqueeze: {}", e)))?;

        // Forward pass — returns (1, seq_len, hidden_size).
        let output = inner
            .model
            .forward(&token_t, &token_type, Some(&attention_t))
            .map_err(|e| HailoError::Tokenizer(format!("BertModel::forward: {}", e)))?;

        // Mean-pool over the sequence dim, weighted by the attention
        // mask. Standard sentence-transformers operation.
        let attention_f = attention_t
            .to_dtype(DType::F32)
            .map_err(|e| HailoError::Tokenizer(format!("mask f32: {}", e)))?;
        let mask = attention_f
            .unsqueeze(2)
            .map_err(|e| HailoError::Tokenizer(format!("mask unsqueeze: {}", e)))?
            .broadcast_as(output.shape())
            .map_err(|e| HailoError::Tokenizer(format!("mask broadcast: {}", e)))?;
        let masked = output
            .broadcast_mul(&mask)
            .map_err(|e| HailoError::Tokenizer(format!("masked mul: {}", e)))?;
        let summed = masked
            .sum(1)
            .map_err(|e| HailoError::Tokenizer(format!("sum: {}", e)))?;
        let denom = mask
            .sum(1)
            .map_err(|e| HailoError::Tokenizer(format!("denom sum: {}", e)))?;
        let pooled = summed
            .broadcast_div(&denom)
            .map_err(|e| HailoError::Tokenizer(format!("div: {}", e)))?;

        // Squeeze batch + read out as Vec<f32>.
        let v: Vec<f32> = pooled
            .squeeze(0)
            .map_err(|e| HailoError::Tokenizer(format!("squeeze: {}", e)))?
            .to_vec1()
            .map_err(|e| HailoError::Tokenizer(format!("to_vec1: {}", e)))?;

        // L2-normalise — matches sentence-transformers convention.
        let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        let v = if norm > 0.0 {
            v.iter().map(|x| x / norm).collect()
        } else {
            v
        };
        Ok(v)
    }
}
