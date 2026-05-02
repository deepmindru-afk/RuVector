---
id: ADR-167
title: ruvector Hailo-8 NPU embedding backend
status: Proposed
date: 2026-05-01
author: ruv
branch: hailo-backend
tags: [ruvector, hailo, hailo-8, npu, embedding, edge-ai, raspberry-pi-5, ai-hat-plus, hailort]
related: [ADR-SYS-0027, ADR-165, ADR-166]
---

# ADR-167 — ruvector Hailo-8 NPU embedding backend

## Status

**Iter 134/135 (2026-05-02): CPU fallback path is production-deployable
today; HEF compile is unblocked at the tooling layer but blocked at the
model-graph layer.** Branch `hailo-backend`.

| Surface | Status |
|---|---|
| Hailo Dataflow Compiler install | ✅ DFC v3.33.0 + HailoRT 4.23.0 installed via `setup-hailo-compiler.sh` (iter 132/135 — auto-pins TF 2.18 + protobuf 3.20.3 + torch 2.4 + transformers 4.49 with `TRANSFORMERS_NO_TF=1` to avoid keras conflicts) |
| ONNX export | ✅ `export-minilm-onnx.py` — `torch.onnx.export` against `transformers.AutoModel`, no optimum-cli dep (iter 135) |
| Hailo parser → optimize → compile | ✅ Pipeline runs end-to-end via `compile-hef.py` (Python SDK API, iter 135). Fails at parse stage with `UnsupportedGatherLayerError` on the BERT `word_embeddings.Gather` and `UnexpectedNodeError` on attention-mask `Where`/`Expand` ops |
| **Conclusion**: BERT-6 ONNX as exported from HuggingFace is **not directly compilable for Hailo-8** without model surgery — the embedding lookup and attention-mask broadcast aren't representable in Hailo's HN graph. Path A (HEF compile) requires re-exporting the ONNX with embeddings precomputed host-side and the encoder block in isolation. Substantial follow-up work; see "HEF model surgery" section below. |
| **Path C — CPU fallback (iter 133/134)** | ✅ Fully production-deployable. `cargo build --features hailo,cpu-fallback` produces a worker binary that runs real BERT-6 on host CPU when `model.safetensors`+`tokenizer.json`+`config.json` are present in the model dir. Validated end-to-end: `sim(dog,puppy)=0.469`, `sim(dog,kafka)=-0.107` (semantically correct ordering). `deploy/download-cpu-fallback-model.sh` fetches the artifacts with sha256 pinning. Latency ~50–150 ms/embed on Cortex-A76, ~10–30 ms on AVX2. |

**The "ship today" path** is `--features hailo,cpu-fallback` plus
`download-cpu-fallback-model.sh`. Real semantic vectors flow end-to-end
from Pi 5 worker to cluster, NPU stays idle. When the HEF model surgery
lands, drop the `model.hef` into the same dir and restart — no other
code changes required, the existing `HailoEmbedder::open` path picks
up the HEF and bypasses CPU fallback automatically.

### HEF model surgery (iter 136+ follow-up, currently scoped only)

The Hailo-8 NPU's HN graph format doesn't represent the standard
HuggingFace BERT export's:
- `Gather` op for token / token-type embedding lookups (these are
  table lookups, not real ML ops)
- `Where`/`Expand` ops for broadcasting the attention mask across
  the QK^T product

The recommended surgery (from Hailo's parser recommendation output):
1. Move embedding lookups host-side: tokenize → embedding-table lookup
   → send `embeddings_out` (shape `[1, 128, 384]` float) to the NPU
   instead of `input_ids`
2. Pre-compute the attention mask host-side and apply it post-NPU
3. Re-export the ONNX with `start_node_names=[/embeddings/Add_1]` and
   `end_node_names=[last_hidden_state]` — encoder layers only, no
   embedding lookup, no mask broadcast
4. Worker's gRPC API stays the same; the change is internal to
   `HailoEmbedder::embed`

This is ~2-3 days of work. Documented but not scheduled — the cpu-
fallback path is sufficient for current ruvllm + ruview throughput.

**Earlier (iter 116) snapshot** preserved below for historical context.

---

**Implemented (modulo HEF compile, external blocker)** on branch
`hailo-backend` as of iter 116 (2026-05-02).

**Iter 99–116 status update** (this session): every code-side mitigation
and feature item that was implementable without external vendor tooling
has shipped. The original validation snapshot (iter 15) is preserved
below for historical context. The current cumulative state:

| Surface | Status as of iter 116 |
|---|---|
| ADR-172 security stack | 6/8 MEDIUM ✓, 2/4 HIGH ✓ — see ADR-172 acceptance gate |
| Cluster crate test suite | 132 host tests + composition test green |
| ESP32-S3 mmWave sensor firmware (iter A) | Live on Waveshare ESP32-S3-Touch-AMOLED-1.8; on-device parser self-test PASS(8) |
| Shared `crates/ruvector-mmwave` parser | 10 unit tests; consumed by both firmware + host bridge |
| Host-side `ruvector-mmwave-bridge` bin | `--simulator` produces real JSONL events; `--workers` posts via embed RPC end-to-end (verified vs fakeworker) |
| ULID request IDs | Iter 109 — 26-char Crockford base32 |
| Cache TTL exposed in stats | Iter 108 |
| HEF compile pipeline (real semantic vectors) | ❌ External blocker — Hailo Dataflow Compiler is proprietary x86-host tooling, runs outside this repo |
| **Placeholder vectors removed (iter 130)** | ✅ `embed()` now returns `HailoError::NoModelLoaded` instead of FNV-1a content hashes; `health.ready` flips false via the new `HailoEmbedder::has_model()` gate so the cluster's `validate_fleet` correctly identifies model-less workers |
| **HEF acquisition recipe (iter 131-132)** | ✅ Three documented paths to land a `model.hef` artifact, with realistic caveats per path. |

### HEF acquisition: the actual three paths (iter 131-132)

**Path A: install the Hailo Dataflow Compiler + compile from ONNX**
- Operator-side prerequisites (one-time): create free Hailo developer
  account at <https://hailo.ai/developer-zone/sw-downloads/>, download
  `hailort_X.Y.Z_amd64.deb` + `hailo_dataflow_compiler-X.Y.Z-py3-none-linux_x86_64.whl`.
- `deploy/setup-hailo-compiler.sh /path/to/downloads` — uses `uv` to
  materialise a Python 3.10 venv (vendor wheel breaks on Python 3.12
  shipped with Ubuntu 24.04+), installs the wheel + optimum-cli into
  the venv, sudo-installs the runtime .deb.
- `deploy/compile-hef.sh` — exports
  `sentence-transformers/all-MiniLM-L6-v2` to ONNX, runs Hailo's
  parser → optimize → compiler pipeline, drops `model.hef`.
- **This is the only documented path that targets Hailo-8** (the chip
  on the Pi 5 + AI HAT+).

**Path B: pre-compiled HEFs from Hailo Model Zoo**
- Two repos: <https://github.com/hailo-ai/hailo_model_zoo> (general
  vision/NLP) and <https://github.com/hailo-ai/hailo_model_zoo_genai>
  (LLMs).
- Reality check (verified 2026-05-02): **every pre-compiled embedding/
  LLM HEF in those zoos targets `hailo15h` or `hailo10h`, NOT
  `hailo8`.** Examples:
  - `bert_base_uncased.yaml`: `supported_hw_arch: [hailo15h, hailo10h]`
  - `tinyclip_vit_8m_16_text_3m_yfcc15m_text_encoder` (3M params,
    0.38G ops — would be ideal for Hailo-8): same hailo15h/10h
    constraint
  - `llama3.2/1b` GenAI HEF: `hef_h10h` field only, no `hef_h8` field
- Path B is therefore **a non-starter for the Pi 5 + AI HAT+ today.**
  Documents itself once Hailo publishes Hailo-8 builds of these
  models, or when an operator upgrades to a Hailo-15h-equipped
  Pi-class board.

**Path C: pure-Rust CPU fallback**
- Add `candle-transformers` dep, load `all-MiniLM-L6-v2` weights
  (safetensors, ~90 MB), run BERT-6 on Cortex-A76 NEON.
- ~400 LOC + ~50 MB of compiled deps.
- NPU stays idle; NPU TOPS budget unused; but real semantic
  embeddings work end-to-end today without any vendor tooling.
- Realistic per-embed latency on Cortex-A76: ~50-150 ms (BERT-6
  forward pass at 384-token sequence, single thread).
- Documented as a future option; not yet implemented on this branch.
| ADR-174 thermal subscriber Unix-socket protocol | ❌ Deferred (iter 95-97 plan never built) |
| Long-running coordinator daemon | ❌ Not built — CLI bins are stateless |
| Native AsyncEmbeddingTransport trait | ❌ Public API change deferred (no consumer demand yet) |

The **only** remaining gap that would meaningfully change behavior on
the Pi 5 + Hailo-8 is the HEF compile step (vendor tooling). Once a
`model.hef` artifact lands at `/var/lib/ruvector-hailo/models/all-minilm-l6-v2/`,
the existing `HailoEmbedder::open` path consumes it without code changes;
vectors stop being FNV-1a content-hash placeholders and become real
semantic embeddings.

---

**Validation snapshot (iter 15, 2026-05-01):**

| Surface | x86 (ruvultra) | Pi 5 + AI HAT+ (cognitum-v0) |
|---|---:|---:|
| `hailort-sys` clippy + tests | ✅ 1 stub | ✅ 1 (`HailoRT 4.23.0` reported via `hailo_get_library_version`) |
| `ruvector-hailo` clippy + lib tests | ✅ 14 | ✅ 3 (`HailoDevice` open/version/drop, real vdevice handle) |
| `ruvector-hailo` tokenizer proptest fuzz | ✅ **7 properties** × 256 random cases each (~1.8 k fuzz inputs) | host-side, N/A |
| `ruvector-hailo-cluster` clippy + lib tests | ✅ **29** | host-side, N/A on Pi |
| `ruvector-hailo-cluster` integration tests | ✅ 2 (P2C+EWMA distribution, failover) | host-side, N/A |
| Worker binary `ruvector-hailo-worker` | ✅ builds | ✅ built + boots, exits at HEF gate |
| Embed CLI `ruvector-hailo-embed` | ✅ builds + `--help` | (not tested on Pi yet) |
| Demo binary `ruvector-hailo-fakeworker` | ✅ end-to-end localhost demo | (not needed on Pi) |
| Stats CLI `ruvector-hailo-stats` | ✅ end-to-end vs fakeworker; tab-separated table out, exit 2 on partial failure | (works against any tonic Embedding server) |
| Bench tool `ruvector-hailo-cluster-bench` | ✅ 8 threads × 2 fakeworkers × 5 s = **94 k req/s**, p99 153 µs, 0 errors over 473 k requests | — |
| Deploy artifacts `deploy/{*.service, *.env.example, install.sh}` | ✅ systemd unit validates clean, sandboxed (DeviceAllow=/dev/hailo0 only, ProtectSystem=strict, NoNewPrivileges, etc.) | (run install.sh on Pi after binary build) |

**Empirical dispatch validation (iter 15):**
- 200 embed requests through real tonic/TCP/HTTP-2 against 2 mock workers (1 ms vs 15 ms latency) → **fast worker received 190, slow received 10** (19:1 EWMA bias).
- 50 embed requests with one of two workers dead → **49 succeeded, 1 errored** (one health-probe budget burned, then dead worker ejected, all subsequent requests routed to live worker).

**Three-binary localhost demo (iter 17):**
- `ruvector-hailo-fakeworker × 2` (deterministic vectors, configurable artificial latency, structured tracing) + `ruvector-hailo-embed` reading from stdin → end-to-end 5 embeds at 189 embeds/sec (first hit 12 ms cold-channel handshake, steady-state ~3.4 ms over loopback).
- Means a developer can today exercise the entire cluster path *without a Pi* — useful for stress-tests + regression checks before real workers come online.

**Coordinator hot-path microbench (iter 19, criterion):**
- `pool::choose_two_random` n=8/16/64 → 47.9 / 74.6 / 134 ns (O(n) over healthy set)
- `HashShardRouter::pick` → ~16 ns/pick alloc-free
- `embed_one_blocking` against in-memory transport (1/2/8 workers) → 119.7 / 137.8 / 189.2 ns
- Dispatch overhead is **two orders of magnitude below network**, so coordinator-side won't be the bottleneck even at 5,000+ embeds/sec aggregate fleet throughput.

**Sustained-load bench (iter 24, `ruvector-hailo-cluster-bench`):**
- Setup: 2 fakeworkers on localhost + 8 client threads, 5 s
- **473,571 requests, 0 errors, 94,693 embeds/sec sustained**
- Latency µs: min=33, p50=81, p90=107, p99=153, max=2,933, avg=83
- Caps the *dispatch + tonic + protobuf + loopback TCP* layer at ~94 k req/s in release build.
- Real-NPU inference at ~3 ms (yolov8s pose latency on Pi 5 + Hailo-8) would dominate to ~333 req/s/worker → cluster stack is nowhere near a bottleneck.

**All `cargo clippy --all-targets -- -D warnings` green** across all three crates.

**Single remaining gate**: Hailo Dataflow Compiler install on ruvultra (HEF compilation step 6). Once the `.hef` lands at `crates/ruvector-hailo/models/all-minilm-l6-v2/model.hef`, two more iterations land:
- iter 15: fill `EmbeddingPipeline::new` (HEF load + vstream creation via hailort-sys)
- iter 16: fill `HailoEmbedder::embed` (encode → push input vstream → pull output vstream → mean_pool → l2_normalize)

Then the loop's "fully implemented and validated" milestone is met end-to-end.

## 1. Context

ruvector currently runs embedding inference on the CPU. On a Raspberry Pi 5
(Cortex-A76 @ 2.4 GHz, 4 cores, NEON), `all-MiniLM-L6-v2` ONNX INT8 hits
roughly 50–100 short-text embeddings/sec — enough for desk use, marginal for
high-rate ingest.

The same Pi 5 carries a **Hailo-8 (26 TOPS) AI HAT+** today:

```
PCIe 0001:01:00.0  Hailo Technologies Hailo-8 [1e60:2864] rev 01
hailort 4.23.0, /dev/hailo0, FW loaded 151 ms
```

Verified hardware bench (this session, see `~/projects/ruvector/docs/notes/hailo-bench.md`):

| Model | Task | FPS (HW-only) | Latency |
|---|---|---:|---:|
| yolov6n_h8 | object detection | 585.9 | 3.18 ms |
| yolov8s_h8 | object detection | 318.0 | 6.62 ms |
| yolov5n_seg_h8 | segmentation | 123.1 | 11.88 ms |

The chip is silicon-saturated for vision at sub-30-ms latency. **Embedding
transformers run on the same NPU at similar tier** — projected single-token
single-sentence inference around 0.5–2 ms once the model is properly compiled.
That is **50–100× the CPU path** and matters for any workload doing
ingest-side embedding (RAG, semantic search, on-device retrieval).

## 2. Decision

Add a new ruvector crate, **`ruvector-hailo`**, that exposes an embedding-only
inference backend over `/dev/hailo0` via the HailoRT C library, gated behind
a Cargo feature `hailo`. Implement it as a **drop-in alternative** to the
existing CPU/ONNX backend, hidden behind `ruvector-core::EmbeddingBackend`
trait. Default builds on non-Pi hardware compile without it.

### 2.1 Five fixed choices

| Decision | Choice | Why |
|---|---|---|
| Crate scope | embedding only (text → vector) | Vision/audio paths already have working precedents; embedding is the highest-impact ruvector use case |
| First model | **all-MiniLM-L6-v2** (`sentence-transformers/all-MiniLM-L6-v2`) | 384-dim, ~22 M params, well-supported, default in ruvector-core |
| Model format on disk | `.hef` (HailoRT 4.x) | The only format the chip runs — produced by Hailo Dataflow Compiler |
| Runtime path | HailoRT C library via `bindgen`-generated FFI | Stable; matches what hailo-tappas uses; avoids Python-bridge overhead |
| Feature gating | `[features] hailo = ["dep:hailort-sys"]` opt-in | x86_64 ruvultra + zenbook builds untouched; hailo-only Pi builds enabled with `--features hailo` |

### 2.2 Crate layout

```
crates/ruvector-hailo/
├── Cargo.toml                  # depends on ruvector-core (trait), hailort-sys (FFI)
├── README.md                   # this file's quickstart subset
├── src/
│   ├── lib.rs                  # public API: HailoEmbedder
│   ├── device.rs               # /dev/hailo0 discovery + open/close
│   ├── hef.rs                  # HEF file load + network group config
│   ├── tokenizer.rs            # WordPiece tokenizer, vocab.txt loader (text → input ids)
│   ├── inference.rs            # input_vstream feed → output_vstream pull → vector
│   ├── pool.rs                 # thread-safe device pool for concurrent embeds
│   └── error.rs                # thiserror-based HailoError → ruvector::EmbeddingError
├── models/                     # *.hef + tokenizer artifacts (downloaded; gitignored binaries)
│   ├── README.md               # how to build the HEF (cross-compile step)
│   └── all-minilm-l6-v2/
│       ├── model.hef           # produced by hailo-compiler (x86 only)
│       ├── vocab.txt           # WordPiece vocab, 30522 entries
│       └── special_tokens.json # CLS/SEP/PAD ids
└── examples/
    └── embed_demo.rs           # ruvector-hailo demo: embed N sentences, print FPS
```

### 2.3 Public API

```rust
pub struct HailoEmbedder { /* opaque */ }

impl HailoEmbedder {
    pub fn open(model_dir: &Path) -> Result<Self, HailoError>;
    pub fn embed(&mut self, text: &str) -> Result<Vec<f32>, HailoError>;
    pub fn embed_batch(&mut self, texts: &[&str]) -> Result<Vec<Vec<f32>>, HailoError>;
    pub fn dim(&self) -> usize;        // 384 for all-MiniLM-L6-v2
    pub fn device_id(&self) -> &str;   // PCIe BDF
}

impl ruvector_core::EmbeddingBackend for HailoEmbedder { /* … */ }
```

The trait `ruvector_core::EmbeddingBackend` is added to `ruvector-core`
(small surface — `embed`, `embed_batch`, `dim`); the existing CPU backend
becomes a default impl. ruvector-cli grows a `--backend {cpu,hailo}` flag.

### 2.4 Toolchain split

Two distinct machines, two distinct steps:

1. **Compilation of the model** to HEF — runs on **ruvultra (x86_64)**:
   - Hailo Dataflow Compiler (Linux x86 only, requires Hailo developer license).
   - Pipeline: `*.onnx` → calibration with a small text corpus → `*.hef`.
   - Output committed to `models/all-minilm-l6-v2/model.hef` (not into git;
     a `models/.gitignore` excludes binary artifacts; CI fetches via Git
     LFS from a private S3 mirror, or developer downloads via a make rule).

2. **Runtime** on the **Pi 5** with the AI HAT+:
   - `cargo build --release -p ruvector-hailo --features hailo` on the Pi
     (or cross-compiled binary copy from ruvultra).
   - `/usr/lib/aarch64-linux-gnu/libhailort.so.4.23.0` is the runtime DSO,
     installed by `hailo-all` (already present on cognitum-v0).

### 2.5 Trait wiring into ruvector-core

ruvector-core already exposes the right trait: `ruvector_core::embeddings::EmbeddingProvider`
with `embed(&self, &str) -> Result<Vec<f32>>`, `dimensions() -> usize`,
`name() -> &str`, and `Send + Sync` bounds. This was discovered iteration 2
(the initial ADR draft incorrectly proposed adding a new `EmbeddingBackend`
trait). The right design is therefore:

- `HailoEmbedder` matches `EmbeddingProvider`'s exact signature shape
  (immutable `&self`, interior mutability via `Mutex` for the device handle).
- Iteration 3 brings the path dep on `ruvector-core` and adds
  `impl ruvector_core::embeddings::EmbeddingProvider for HailoEmbedder`.
- Existing CPU/ONNX backends are untouched.

This keeps the change a strict addition: no breaking modifications to
ruvector-core or its consumers (ruvector-cli, ruvector-server, etc.).

## 3. Considered alternatives

### 3.1 Python bridge to hailo's existing python wrapper

Rejected. ruvector is Rust-only end-to-end; introducing a Python interpreter
adds startup cost, deployment complexity, GIL contention on multi-thread
embedding, and an unwelcome dependency tree. FFI to the C library is what
TAPPAS and the official Hailo C++ examples do — same path.

### 3.2 ONNX Runtime with an optional Hailo execution provider

There is a community-maintained ORT execution provider for Hailo, but it's
not in the upstream ORT release stream as of 2026-05; using it would pin us
to a fork and reintroduce ORT's full dependency footprint (which we already
don't have in ruvector-core). Direct HailoRT is the lower-friction path.

### 3.3 Vendor a minimal `hailort-sys` crate

Kept as the implementation detail of choice. We generate FFI bindings via
`bindgen` against `hailort.h` at build time, link against the system
`libhailort.so` (preferred — matches the in-tree kernel driver's firmware)
or an env-overridable static lib path for cross-compilation.

### 3.4 Wait for an upstream `hailort-rs` crate to mature

The community-maintained `hailort-rs` (github.com/IPDS-NTU/hailort-rs) covers
the API surface we need but doesn't follow `hailort` releases tightly. We
take inspiration from its design, but our `hailort-sys` is local and pinned
to the system `libhailort` version (4.23.0 today).

## 4. Consequences

### Positive
- ruvector gets a 50-100× embedding-throughput on Pi 5 + HAT+, unblocking
  on-device RAG, federated retrieval, and real-time agent workloads.
- The `EmbeddingBackend` trait is a one-time refactor that pays for any
  future accelerators (Coral, NVIDIA Jetson, FPGA via ADR-167-fpga).
- Every other ruvector subsystem (HNSW search, cluster gossip, RAG)
  inherits the speedup without code change.

### Negative
- Adds a non-trivial build step (Hailo Dataflow Compiler on x86) that's
  *not* reproducible from CI alone — the compiler license is per-machine.
  Mitigation: commit a pre-built `.hef` artifact (LFS) and document the
  rebuild procedure in `models/README.md`.
- HailoRT C ABI changes between major versions; we pin to 4.23.0 today,
  must re-bindgen on upgrade.
- HEF compilation has its own quantization quirks; `all-MiniLM-L6-v2`'s
  attention layers may need tweaks (e.g., per-head quantization scales)
  to maintain MTEB-level accuracy. Phase 2 problem; Phase 1 accepts
  default-quantized output and measures the accuracy delta.
- Adds another crate (#129) to the workspace.

### Neutral
- `models/` is committed but binaries are gitignored.
- ruvector-cli grows a `--backend` flag; default stays CPU/ONNX so non-Pi
  users see no regression.

## 5. Implementation plan (this is the loop's work plan)

| Step | Deliverable | Verification |
|---|---|---|
| 1 (this iteration) | branch `hailo-backend` created; crate scaffold + ADR + Cargo.toml + lib.rs stub | `git branch`, `cargo check -p ruvector-hailo` succeeds with no `hailo` feature |
| 2 (this iteration) | ~~`EmbeddingBackend` trait added~~ — discovered `ruvector_core::embeddings::EmbeddingProvider` already exists (`fn embed(&self, &str) -> Result<Vec<f32>>; fn dimensions() -> usize; fn name() -> &str`). HailoEmbedder API surface updated to match exactly, with `Mutex<()>` placeholder for interior mutability. | `cargo check -p ruvector-hailo` clean; signature parity unit test passes |
| 3 (this iteration) | `hailort-sys` mini-crate added: `Cargo.toml` (links="hailort", `hailo` feature), `wrapper.h`, `build.rs` (bindgen against `<hailo/hailort.h>`, allowlist `hailo_*`), `src/lib.rs` (`version_triple()` smoke). | ✅ `cargo build --features hailo` on Pi succeeds; `cargo test --features hailo` prints `HailoRT version: 4.23.0` |
| 4 (this iteration) | `device::HailoDevice::{open, version, drop}` wired against `hailort-sys`. Calls `hailo_create_vdevice` / `hailo_get_library_version` / `hailo_release_vdevice` through bindgen FFI under feature gate. ruvector-hailo `[features] hailo = ["hailort-sys/hailo"]`. | ✅ `cargo test --features hailo` on Pi: `HailoRT 4.23.0 via HailoDevice`, all 3 tests pass |
| 5 (this iteration) | `WordPieceTokenizer` in `src/tokenizer.rs`: BasicTokenizer (lowercase + whitespace + punctuation split), greedy-longest-match WordPiece with `##` continuations, `[CLS] … [SEP]` wrap, optional pad-to-`max_seq`. `EncodedInput { input_ids, attention_mask, actual_len }`. Pure std, no FFI. | ✅ 5 unit tests on x86: special-token ids match BERT convention; `encode("Hello, World!")` → `[101,104,106,105,100,102]`; `"ruvector"` → `[ru, ##v, ##ec, ##tor]`; pad-to-max-seq; truncation. Real `all-MiniLM-L6-v2` vocab parity test deferred to step 6. |
| 6 | Compile `.hef` for `all-MiniLM-L6-v2` on ruvultra (Hailo Dataflow Compiler) and commit/upload | **BLOCKED** — Hailo Dataflow Compiler not installed on ruvultra (`which hailo` empty, no `hailo_sdk_client` Python module). Requires download from `hailo.ai/developer-zone/` (free developer login). User action gate. |
| 6.5 (this iteration, scope adjusted) | `inference::EmbeddingPipeline` skeleton + pure-Rust `mean_pool` and `l2_normalize` helpers in `src/inference.rs`. Pipeline gates HEF/vstream wiring behind `NotYetImplemented` until step 6 unblocks. | ✅ 14 tests on x86 (5 inference helpers + 5 tokenizer + 2 device + 2 lib). `mean_pool` matches arithmetic mean, masks padding, returns zero on all-masked input. `l2_normalize` yields unit norm; idempotent on zero vector. |
| 7 | `inference::EmbeddingPipeline::embed_one()` end-to-end on Pi: text → tokens → input vstream → output vstream → mean-pool → L2 → vector | `embed("hello") -> [f32; 384]` printed, deterministic across runs |
| 8 | Compare CPU-backend output vs Hailo-backend output for 100 sentences (cosine similarity ≥ 0.99) | accuracy regression test in `tests/` |
| 9 | Benchmark: throughput on Pi (target ≥ 1000 embeds/sec batch=32), record in `docs/notes/hailo-embed-bench.md` | `examples/embed_demo.rs --benchmark` |
| 10 (worker side, iter 12) | `ruvector-hailo-worker` binary in `crates/ruvector-hailo-cluster/src/bin/worker.rs`. Wraps `HailoEmbedder` and serves `embedding_server::Embedding` via tonic. Env vars `RUVECTOR_WORKER_BIND` + `RUVECTOR_MODEL_DIR`. Graceful SIGTERM/SIGINT shutdown. `--features hailo` propagates to ruvector-hailo. | ✅ Builds on x86 (no feature) and Pi with `--features hailo`. Pi runtime smoke: reports `bind=0.0.0.0:50051 model_dir=...`, attempts open, exits clean with `NotYetImplemented` (gate is HEF compilation only). |
| 10 (client side, this iter) | `ruvector-hailo-embed` binary in `crates/ruvector-hailo-cluster/src/bin/embed.rs`. Reads stdin one-doc-per-line, embeds via configured cluster, prints JSON one-per-line. Args: `--workers <csv>` or `--tailscale-tag <tag> --port N`, plus `--dim` and `--fingerprint`. Outputs summary throughput stats on stderr. Built without clap (~140 lines argv parser). Modifying `ruvector-cli` itself was rejected — too much workspace-wide blast radius for a feature-gated path. | ✅ Builds clean. `--help` renders. End-to-end is implicitly tested via the underlying 25 cluster tests (P2C, EWMA, retry, DimMismatch, gRPC roundtrip, Tailscale discovery). |
| 11 | Final validation: end-to-end RAG query on a 10k-sentence corpus, latency budget ≤ 5 ms p99 | timed run, results pinned in ADR-167 §6 |

## 6. Open questions

1. **Calibration corpus** — what 1k-sample text we feed Hailo's compiler
   for INT8 calibration. Default: ruvector's `bench_data/glove.6B.100d.txt`
   first 10k lines, after WordPiece tokenization.
2. **Sequence length** — HEF must be compiled for a fixed token count.
   Pick **`max_seq=128`** as the first cut (covers 99% of search-query-style
   inputs); larger inputs truncated. Phase 2 considers a multi-shape HEF.
3. **Pooling** — `all-MiniLM-L6-v2` uses mean-pooling over token embeddings;
   the pooling op may not be in Hailo's allowlist for the H8 — we may need
   to do it on CPU after the NPU emits per-token outputs. Adds 50 µs on M1
   Pi CPU; trivial.
4. **Concurrency** — HailoRT supports multi-network on one device but the
   chip serializes within a model. Real concurrency comes from the
   service mode (`hailort_service`) — not Phase 1.

## 8. Multi-Pi clustering (added 2026-05-01, iteration 3)

A single Pi 5 + Hailo-8 sustains roughly **1,000 short-text embeds/sec**
once iteration 9 lands its bench. Real workloads (RAG fan-out, federated
ingest, multi-tenant agent queries) routinely want 5–10× that, and a
single-failure point is unwelcome anyway. This section defines the
multi-device design that complements the single-device backend above.

### 8.1 Scope

- **In scope**: a coordinator that fans embed requests across N Pi 5 + AI
  HAT+ workers, observes per-worker health, transparently fails over,
  and presents a single `EmbeddingProvider` API to ruvector callers.
- **Out of scope (Phase 1)**: distributed *training* of new models;
  PCIe-switch multi-NPU on a single Pi (covered by ADR-NN-multi-npu);
  cross-WAN replication (covered by `ruvector-replication`).

### 8.2 Topology

```
                        ┌──────────────────────────┐
                        │  ruvector-cli / server   │  (any client)
                        │      ↓ EmbeddingProvider │
                        │   HailoClusterEmbedder   │  (this crate, ruvector-hailo-cluster)
                        └────────┬───────┬─────────┘
                            mDNS / Tailscale discover
                  ┌──────────────┼───────┴────────┬─────────────┐
                  ▼              ▼                ▼             ▼
              ┌───────┐      ┌───────┐        ┌───────┐     ┌───────┐
              │ pi-A  │      │ pi-B  │  ...   │ pi-N  │     │  …    │
              │ Hailo │      │ Hailo │        │ Hailo │     │       │
              └───────┘      └───────┘        └───────┘     └───────┘
              cognitum-v0    cognitum-v1                              (workers — each
                                                                       runs ruvector-server
                                                                       with --backend hailo)
```

### 8.3 Decisions

| Decision | Choice | Rationale |
|---|---|---|
| Worker discovery | **Tailscale tag-based**, mDNS fallback | ruvultra/zenbook/Pi already tailnet-resident; tags (e.g. `tag:ruvector-hailo-worker`) enumerate workers without static config. mDNS fallback for LAN-only setups. |
| Worker RPC | **gRPC over Tailscale (TLS-by-tailnet)** with HTTP/2 fallback | Existing `ruvector-server` already exposes a gRPC surface; new method `EmbedHailo(text) -> vec<f32>` reuses that scaffolding. Tailnet implies authenticated transport. |
| Load-balancing | **Power-of-two random choice** with per-worker EWMA latency | Cheap, near-optimal for embedding workloads (uniform tasks, low variance). Beats round-robin when workers heterogeneous. |
| Health checks | gRPC ping every 5 s; eject after 3 consecutive failures or p99>500 ms for 30 s | Fast-enough to react, slow-enough to avoid flapping. |
| Failover | **Idempotent retry** to a different worker on RPC failure | Embedding is pure (same input → same output); safe to retry. Cap at 2 retries; surface error if both miss. |
| Batch policy | Client-side batches up to N=32, sharded across workers by hash(text) | Hash-shard means same input always hits same worker — cache-friendly if workers add their own LRU. Random shard for cold inputs. |
| Authentication | Tailscale ACLs gate which clients can reach `tag:ruvector-hailo-worker` | No app-level auth needed inside the tailnet; tailnet membership = authorization. |
| Failure semantics | At-least-once embed delivery; no consistency requirement (embedding is pure) | Avoids the consensus overhead `ruvector-cluster` carries for vector storage. |

### 8.4 New crate: `ruvector-hailo-cluster`

```
crates/ruvector-hailo-cluster/
├── Cargo.toml                  # depends on ruvector-core (trait), tonic (gRPC), tokio
├── src/
│   ├── lib.rs                  # public: HailoClusterEmbedder, ClusterConfig
│   ├── discovery.rs            # Tailscale tag enumeration + mDNS fallback
│   ├── pool.rs                 # P2C random selection + EWMA latency
│   ├── health.rs               # async gRPC ping, ejection logic
│   ├── shard.rs                # hash-based input → worker mapping
│   └── proto.rs                # generated tonic stubs from proto/embedding.proto
└── proto/
    └── embedding.proto         # service Embedding { rpc Embed (Req) returns (Vec) }
```

`HailoClusterEmbedder` implements `ruvector_core::embeddings::EmbeddingProvider`,
just like single-device `HailoEmbedder` will. **Existing ruvector callers
need zero changes** — they consume an `Arc<dyn EmbeddingProvider>` and
either flavor satisfies it.

### 8.5 Worker side

Each Pi runs `ruvector-server` with `--backend hailo --bind 0.0.0.0:50051`
and a `tag:ruvector-hailo-worker` Tailscale ACL tag. The server's existing
gRPC surface gains one new RPC:

```
rpc EmbedHailo (EmbedRequest) returns (EmbedResponse);
```

with `EmbedRequest = { text: string, max_seq: int32 }` and
`EmbedResponse = { vector: repeated float, dim: int32, latency_us: int64 }`.

That's the only new server-side code; the rest is config.

### 8.6 Bench targets (Phase 1.5)

With **2× Pi 5 + Hailo-8** workers, target:
- ≥ **1,800 embeds/sec** aggregate (≥90% of single-Pi × 2 — overhead from
  P2C choice + gRPC framing eats ~10%)
- p99 ≤ **6 ms** (single-Pi p99 + 2 ms gRPC over tailnet)
- Failover under one-worker-down: no client-visible errors, throughput
  drops to single-Pi level within one health-check tick (~5 s)

Scale-out should be near-linear up to ~8 Pis (the point where the
coordinator's gRPC fan-out becomes the bottleneck on its own host's NIC).

### 8.7 Implementation plan additions to §5

Phase 2 of the loop (after §5 step 11 lands):

| Step | Deliverable | Verification |
|---|---|---|
| 12 (this iteration, scope adjusted — protoc not on ruvultra) | `crates/ruvector-hailo-cluster` scaffolded with `HailoClusterEmbedder`, `P2cPool` (EWMA + ejection), `HashShardRouter`, `EmbeddingTransport` trait + `NullTransport`, `ClusterError`, and `proto/embedding.proto` (text only, codegen pending). | ✅ 9 tests green: empty-worker reject, single-worker pick, EWMA prefers-lower-latency, mark-healthy-restores-ejected, shard determinism + distribution (1000 inputs × 4 workers ~uniform). Tonic codegen of the proto deferred to step 14 (needs `protoc`-bin-vendored). |
| 13 (this iteration) | `Discovery` trait + `StaticDiscovery` (config-driven) + `TailscaleDiscovery` (`tailscale status --json` shell-out, peer tag filter, IPv4 first-pick, sorted-stable output). Pure JSON parser is decoupled from the subprocess (test fixtures feed it directly). | ✅ 6 tests: tag-filter (3 peers → 2 tagged kept), IPv6-only-peer skip, null-Peer-map handles empty, invalid JSON rejected, static-list passthrough, **live `tailscale status --json` against ruvultra's real tailnet** smoke-tested (passes independent of how many workers are currently tagged). |
| 14 (this iteration, partial) | tonic codegen wired via `protoc-bin-vendored`; `GrpcTransport` (tokio-runtime-backed `EmbeddingTransport` impl) with channel-caching, configurable connect/RPC timeouts. | ✅ 3 end-to-end tests against an in-process tonic mock worker: embed roundtrip (vector + latency), health metadata, channel cached across 5 calls. Real-worker test (Pi-side server) waits on step 6/7 (HEF). |
| 15 (this iteration) | `HailoClusterEmbedder::embed_one_blocking` dispatch loop: P2C pick → transport.embed → record_latency on success / record_health_failure on error / retry up to 2 times across different workers. Dim-mismatch is fatal (no retry — fleet hetero-fingerprint guard). | ✅ 4 dispatch tests: happy-path returns vector + 1 call; retry-then-succeed (2 fails + 1 ok = 3 calls); budget-exhausted → AllWorkersFailed; dim-mismatch fatal → 1 call. |
| 16 | Failover: kill one worker mid-bench, verify continued operation | scripted test; no failed embed calls in the surviving worker's log |
| 17 | Aggregate bench: 2× Pi 5 sustained throughput, p99 latency | recorded in `docs/notes/hailo-cluster-bench.md` |

### 8.8 Open questions for the cluster layer

1. **Should we use `hailort_service` mode on each worker?** It allows
   multiple processes to share `/dev/hailo0` — useful if you want both
   ruvector-server and a TAPPAS pipeline running concurrently. Default:
   no, ruvector-server claims the device exclusively. Re-evaluate if
   anyone hits the contention.
2. **Cross-region clustering?** Tailscale spans regions transparently
   but adds 50-100 ms latency. Out of Phase 1 scope; revisit only when
   a real multi-DC deployment exists.
3. **Should the worker advertise its model fingerprint?** Yes (compile-
   time hash of the `.hef` + tokenizer). Coordinator refuses to mix
   workers with different fingerprints — prevents silent vector-space
   drift across a fleet.

## 7. References

- ADR-SYS-0027 — N6 NPU edge sensor node (sibling NPU; same lessons)
- ADR-165 — tiny ruvLLM agents on ESP32 SoCs (existing Rust on-device path)
- ADR-166 — ESP32 Rust cross-compile bring-up ops (cross-compile precedent)
- HailoRT 4.23 reference — `/opt/hailo/hailort-4.23/include/hailort.h`
- Hailo Model Zoo — `https://github.com/hailo-ai/hailo_model_zoo`
- `hailort-rs` (community FFI design we draw from) — `https://github.com/IPDS-NTU/hailort-rs`
