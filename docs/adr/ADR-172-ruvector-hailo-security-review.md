---
id: ADR-172
title: ruvector-hailo deep security review
status: Proposed
date: 2026-05-02
author: ruv
branch: hailo-backend
tags: [ruvector, hailo, security, audit, mtls, supply-chain]
related: [ADR-167, ADR-168, ADR-169, ADR-170, ADR-171]
---

# ADR-172 — Deep security review

## Status

Proposed — companion ADR for PR #413. Each finding tagged with severity
+ proposed mitigation. Implementation lands as iterations 91-97 across
follow-up PRs.

## Threat model

Three operator scenarios drive the threat surface:

| Scenario | Trust assumption | Bad actor capability |
|---|---|---|
| **A. Single-tenant LAN** (home lab, R&D) | All workers trusted | None — internal threat only |
| **B. Multi-tenant tailnet** (small team, mixed trust) | Workers trusted; co-tenants might not be | Spoof a worker, observe traffic |
| **C. Public internet exposure** (don't do this, but plausible) | Nothing trusted | Full active MITM, DoS, supply-chain |

Pre-iter-91 the codebase implicitly targets scenario A. The findings
below scope what each scenario adds.

## Findings

### 1. Network attack surface (tonic gRPC) — HIGH

**1a. No TLS / no mTLS.** [✅ MITIGATED — iter 99]
All coordinator↔worker traffic was cleartext over WiFi/Tailnet/LAN.
Tailscale's WireGuard envelope mitigates Scenario B over the tailnet
proper but doesn't help anything off-tailnet. LAN deploys were wide open.

*Mitigation (shipped iter 99):* New `tls` cargo feature on
`ruvector-hailo-cluster` enables rustls-backed TLS via tonic's
`ServerTlsConfig` + `ClientTlsConfig`. Worker reads `RUVECTOR_TLS_CERT`
+ `RUVECTOR_TLS_KEY` env vars (and optional `RUVECTOR_TLS_CLIENT_CA`
for mTLS); coordinator constructs `GrpcTransport::with_tls(connect, rpc,
TlsClient)` to dial `https://`. Feature is off by default (back-compat);
recommended-on for Scenario B+. Tested via `tests/tls_roundtrip.rs` —
self-signed cert generated at runtime, full embed + health roundtrip
asserted, plus a negative test that plaintext clients fail cleanly
against TLS-only servers.

**1b. No client authentication.** [✅ MITIGATED — iter 100]
Any tonic client reaching a worker could saturate `/dev/hailo0`. NPU is
a shared limited resource — a single attacker could deny service to all.

*Mitigation (shipped iter 100):* Worker reads `RUVECTOR_TLS_CLIENT_CA`
env var (added iter 99) and applies `TlsServer::with_client_ca`. Combined
with tonic's default `client_auth_optional = false`, any client lacking
a CA-signed identity is rejected at handshake. Coordinator side gains
`TlsClient::with_client_identity_bytes` / `with_client_identity` to
present a CA-issued cert. Tested via `tests/mtls_roundtrip.rs` — 3 cases:
(1) valid CA-signed client succeeds, (2) anonymous client rejected,
(3) untrusted self-signed client rejected. Bearer-token interceptor
remains a future option for token-based deployments.

**1c. `--workers-file` accepts arbitrary host:port.**
Path-traversal (file injection) and SSRF via discovery file content.

*Mitigation:* Document that `--workers-file` must be operator-controlled.
Add a manifest signature option (`--workers-file-sig <path>`) that
verifies a detached Ed25519 signature before loading.

**1d. Tailscale tag spoofing.**
If an attacker controls a tagged peer they auto-join the fleet via
`--tailscale-tag`. Tailscale ACLs limit who can apply tags but
misconfigured tagOwners is a real risk.

*Mitigation:* Document tag governance prerequisites. Optionally add
`--require-fingerprint <fp>` so even auto-discovered peers must match
the expected model fingerprint to be dispatched to.

### 2. Cache integrity / poisoning — MEDIUM

**2a. Empty `expected_model_fingerprint` skips integrity check.**
Default-empty in CLI flags, tests, demos, and examples. Operator opting
into `--auto-fingerprint` is the only thing protecting them — and
auto-fingerprint trusts the first-reachable worker.

*Mitigation:* Make `--fingerprint <hex>` required when `--cache > 0`
(opt-out via explicit `--allow-empty-fingerprint`). Document the
"silently serve stale" failure mode in BENCHMARK.md.

**2b. Worker-reported fingerprint is trusted blindly.**
A malicious worker can claim any fingerprint. Cache key includes the
*coordinator's expected* fp, which mitigates if it's set — but iter
46's `--auto-fingerprint` flow asks the worker for the fp, so a hostile
worker can pollute.

*Mitigation:* When `--auto-fingerprint` is set, after discovering a fp,
require at least 2 workers in the fleet to report the same fp before
trusting it. Currently any single worker can establish "the" fp.

**2c. No cache encryption at rest.**
Cache lives in process RAM only — not strictly an issue today. Will
matter if a future iter persists the cache to disk for warm restarts.

*Mitigation:* Document the in-RAM-only invariant; add an audit gate to
CI that fails any PR introducing cache-on-disk paths.

### 3. Worker-side hardening — MEDIUM

**3a. libhailort runs as root by default.**
`/dev/hailo0` is `crw-rw-rw-` on the Pi 5 we tested, so root isn't
strictly required — but the systemd unit defaults to root. ProtectSystem
strict + NoNewPrivileges help; a dedicated `ruvector-hailo` user would
be safer.

*Mitigation:* Update `deploy/ruvector-hailo-worker.service`:
- `User=ruvector-hailo` / `Group=ruvector-hailo`
- `DynamicUser=true` as alternative
- udev rule `KERNEL=="hailo0", MODE="0660", GROUP="ruvector-hailo"`
- `install.sh` creates the user/group + drops the udev file

**3b. No rate limiting per peer.**
Single client can DoS by saturating NPU at line rate. Workers process
one embed at a time (Mutex), so concurrent attackers serialize — but
that's still 100% utilization.

*Mitigation:* Tonic interceptor that rate-limits per source IP /
client-cert fingerprint. `governor` crate (~50 LOC).

**3c. No audit log.**
Worker tracing logs embed text + request_id. Useful for ops but a
privacy concern at scale (text content in logs forever). No way to
opt out except `RUST_LOG=warn` which loses other diagnostics.

*Mitigation:* Add `--log-text-content {full|hash|none}` flag. Default
to `hash` (sha256 of input) so correlation works without leaking text.

### 4. Tracing / log injection — LOW

**4a. request_id from caller is propagated verbatim.**
Caller can inject control chars / ANSI escapes / newlines into worker
tracing spans. Could log-forge multi-line entries to confuse log
analysis.

*Mitigation:* Sanitize in `proto::extract_request_id`: strip control
chars, cap at 64 chars, fall back to random if hostile-shaped.

**4b. x-request-id metadata header has no length cap.**
Large values inflate log line size; no DoS but resource-burn.

*Mitigation:* Same fix as 4a — 64-char cap.

### 5. Build supply chain — MEDIUM

**5a. bindgen against /usr/include/hailo/hailort.h on Pi.**
We trust whatever's at that path. `dpkg verify hailort` would let CI
detect tampering.

*Mitigation:* `build.rs` records `dpkg -s hailort | sha256` into a
build-time const; runtime asserts on mismatch.

**5b. protoc-bin-vendored crate ships protoc binary.**
Pre-built binary in build-deps. Verify provenance.

*Mitigation:* Pin a specific version + sha256 in Cargo.lock (already
true). Add cargo-deny config to alert on protoc-bin-vendored version
bumps.

**5c. No cargo-audit / cargo-deny in CI.**
Vulnerable transitive deps would land silently.

*Mitigation:* Add `.github/workflows/audit.yml` running cargo-audit +
cargo-deny on every push.

### 6. HEF artifact pipeline (future, when HEF lands) — HIGH

**6a. HEF is ~MB binary loaded by libhailort firmware.**
Operator drops a file at `models/all-minilm-l6-v2/model.hef`; libhailort
trusts it. A swapped HEF can do anything the NPU firmware permits.

*Mitigation:* Worker startup verifies a detached signature
(`model.hef.sig`) against a baked-in operator pubkey. Cache fingerprint
includes the signature hash. Refuse to load unsigned HEFs unless
`--unsigned-ok` flag passed.

**6b. HEF origin chain.**
Who compiled it? Hailo Dataflow Compiler runs on x86; supply chain there
matters. Log the compiler version + ONNX source sha256 on every load.

### 7. ruview / brain integration (ADR-171 future) — MEDIUM

**7a. Brain `share` exfiltrates content to Cloud Run.**
By design — that's how the shared knowledge graph works. But telemetry
paths must not leak PII or query content.

*Mitigation:* `mcp-brain.service` runs with `--telemetry-only` flag
that strips text content from outbound messages. Cloud Run side
already has differential privacy ε=1.0 on embeddings (per CLAUDE.md);
extend to text fields.

**7b. LoRa transport plaintext over the air.**
ADR-171 §LoRa proposed encrypting payload with the model fingerprint
as the symmetric key. That's not a real key — it's a public hash.
Anyone who knows the fingerprint can decrypt.

*Mitigation:* Replace with X25519 ECDH session keys on the LoRa
transport handshake. Each gateway+sensor pair establishes a fresh
session key. Out-of-band key exchange via QR code at provisioning.

## Mitigation roadmap

| Iter | Severity | Item | Implementation |
|---|---|---|---|
| 91 | HIGH | 1a — TLS support | tonic ServerTlsConfig + ClientTlsConfig; docs (✅ shipped iter 99) |
| 91 | LOW | 4a/4b — request_id sanitisation | proto::extract_request_id 64-char cap + control-char strip |
| 92 | HIGH | 1b — mTLS client auth | --require-client-cert worker flag (✅ shipped iter 100 via RUVECTOR_TLS_CLIENT_CA) |
| 92 | MEDIUM | 5c — cargo-audit CI | new workflow + initial vuln triage |
| 93 | MEDIUM | 3a — drop root | new user + udev rule + install.sh update |
| 93 | MEDIUM | 2a — fp required with cache | CLI flag enforcement + docs |
| 94 | MEDIUM | 3b — per-peer rate limit | governor interceptor |
| 94 | MEDIUM | 2b — auto-fp quorum requirement | discover_fingerprint quorum mode |
| 95 | MEDIUM | 3c — log text hash mode | --log-text-content flag |
| 96 | HIGH (future) | 6a — HEF signature verification | sig file + pubkey on worker startup |
| 97 | MEDIUM | 7a/7b — brain + LoRa | telemetry-only flag + X25519 LoRa |

## Out of scope

* CVE triage of transitive deps — handled by 5c's cargo-audit workflow
* Hardware-level attacks (Hailo firmware vulns, PCIe DMA) — vendor's
  responsibility; we trust the firmware once `/dev/hailo0` exists
* Side-channel timing attacks against the cache — out of scope for an
  embedding cache; mitigation would be constant-time ops, expensive

## Acceptance criteria

ADR-172 considered "implemented" when:
- All 4 HIGH items have shipped with tests
- 2/3 MEDIUM items have shipped (7 of 11 total)
- A penetration-test pass against scenario B confirms no exploitable path
- cargo-audit + cargo-deny green on every commit
