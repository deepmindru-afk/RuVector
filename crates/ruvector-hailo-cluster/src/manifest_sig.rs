//! Ed25519 detached-signature verification for `--workers-file` manifests
//! (ADR-172 §1c MEDIUM mitigation, iter 107).
//!
//! A worker manifest is just a text file the operator commits to git.
//! Without a signature, an attacker who can write to that file gets to
//! point coordinator embed/bench/stats traffic at arbitrary host:port
//! pairs (worker substitution / SSRF). This module gives operators an
//! opt-in path: sign the manifest with an offline Ed25519 key, ship the
//! detached signature alongside it, point the CLI at the public key.
//!
//! # Wire format
//!
//! All three files are pure ASCII for `cat`-debuggability and to avoid
//! pulling in a PEM/PKCS8 parser:
//!
//!   manifest        : same as today (host:port per line + comments)
//!   `<manifest>.sig`: 128 lowercase hex chars (the 64-byte signature)
//!                     + an optional trailing newline
//!   pubkey file     : 64 lowercase hex chars (the 32-byte VerifyingKey)
//!                     + an optional trailing newline
//!
//! Generate a key pair offline (e.g. with `openssl genpkey
//! -algorithm Ed25519 ...` then convert, or with the rcgen / ed25519-dalek
//! CLI from the same crate this file depends on). Sign with:
//!
//!   ed25519_dalek::SigningKey::sign(&manifest_bytes)
//!
//! The signature is detached so the manifest stays a plain text file.

use crate::error::ClusterError;
use ed25519_dalek::{Signature, Verifier, VerifyingKey};
use std::path::Path;

/// Decode `s` as N bytes of lowercase ASCII hex, ignoring a single
/// trailing newline. Returns a clean `ClusterError::Transport` on any
/// problem so call sites get a uniform error type.
fn hex_decode(s: &str, want_bytes: usize, what: &str) -> Result<Vec<u8>, ClusterError> {
    let trimmed = s.trim_end_matches(['\n', '\r', ' ']);
    if trimmed.len() != want_bytes * 2 {
        return Err(ClusterError::Transport {
            worker: "<manifest_sig>".into(),
            reason: format!(
                "{}: expected {} hex chars, got {} ({:?}…)",
                what,
                want_bytes * 2,
                trimmed.len(),
                trimmed.get(..trimmed.len().min(8)).unwrap_or(""),
            ),
        });
    }
    let mut out = Vec::with_capacity(want_bytes);
    let bytes = trimmed.as_bytes();
    for i in 0..want_bytes {
        let hi = (bytes[i * 2] as char)
            .to_digit(16)
            .ok_or_else(|| ClusterError::Transport {
                worker: "<manifest_sig>".into(),
                reason: format!("{}: non-hex char at offset {}", what, i * 2),
            })?;
        let lo = (bytes[i * 2 + 1] as char)
            .to_digit(16)
            .ok_or_else(|| ClusterError::Transport {
                worker: "<manifest_sig>".into(),
                reason: format!("{}: non-hex char at offset {}", what, i * 2 + 1),
            })?;
        out.push((hi * 16 + lo) as u8);
    }
    Ok(out)
}

/// Verify `signature_hex` over `manifest_bytes` under `pubkey_hex`.
/// Returns `Ok(())` on success, `ClusterError::Transport` on any
/// failure (decode, length, signature mismatch).
pub fn verify_detached(
    manifest_bytes: &[u8],
    signature_hex: &str,
    pubkey_hex: &str,
) -> Result<(), ClusterError> {
    let pk_bytes = hex_decode(pubkey_hex, 32, "pubkey")?;
    let pk_arr: [u8; 32] = pk_bytes
        .as_slice()
        .try_into()
        .expect("hex_decode returned 32 bytes");
    let pk = VerifyingKey::from_bytes(&pk_arr).map_err(|e| ClusterError::Transport {
        worker: "<manifest_sig>".into(),
        reason: format!("pubkey decode: {}", e),
    })?;
    let sig_bytes = hex_decode(signature_hex, 64, "signature")?;
    let sig_arr: [u8; 64] = sig_bytes
        .as_slice()
        .try_into()
        .expect("hex_decode returned 64 bytes");
    let sig = Signature::from_bytes(&sig_arr);
    pk.verify(manifest_bytes, &sig).map_err(|e| ClusterError::Transport {
        worker: "<manifest_sig>".into(),
        reason: format!("signature verification failed: {}", e),
    })
}

/// File-based wrapper around [`verify_detached`]: reads manifest, sig,
/// and pubkey from disk and verifies. Use this from the discovery path.
pub fn verify_files(
    manifest_path: &Path,
    sig_path: &Path,
    pubkey_path: &Path,
) -> Result<(), ClusterError> {
    let manifest = std::fs::read(manifest_path).map_err(|e| ClusterError::Transport {
        worker: "<manifest_sig>".into(),
        reason: format!("read manifest {}: {}", manifest_path.display(), e),
    })?;
    let sig = std::fs::read_to_string(sig_path).map_err(|e| ClusterError::Transport {
        worker: "<manifest_sig>".into(),
        reason: format!("read signature {}: {}", sig_path.display(), e),
    })?;
    let pk = std::fs::read_to_string(pubkey_path).map_err(|e| ClusterError::Transport {
        worker: "<manifest_sig>".into(),
        reason: format!("read pubkey {}: {}", pubkey_path.display(), e),
    })?;
    verify_detached(&manifest, &sig, &pk)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ed25519_dalek::{Signer, SigningKey};

    fn fixture_keypair() -> (SigningKey, String) {
        // Deterministic test key — never use this in production; it's
        // committed in the test source. The 32-byte seed is arbitrary.
        let seed: [u8; 32] = [
            0xa1, 0xa2, 0xa3, 0xa4, 0xa5, 0xa6, 0xa7, 0xa8,
            0xb1, 0xb2, 0xb3, 0xb4, 0xb5, 0xb6, 0xb7, 0xb8,
            0xc1, 0xc2, 0xc3, 0xc4, 0xc5, 0xc6, 0xc7, 0xc8,
            0xd1, 0xd2, 0xd3, 0xd4, 0xd5, 0xd6, 0xd7, 0xd8,
        ];
        let sk = SigningKey::from_bytes(&seed);
        let pk = sk.verifying_key().to_bytes();
        let mut hex = String::with_capacity(64);
        for b in pk {
            use std::fmt::Write as _;
            write!(&mut hex, "{:02x}", b).unwrap();
        }
        (sk, hex)
    }

    fn sig_hex(sk: &SigningKey, msg: &[u8]) -> String {
        let bytes = sk.sign(msg).to_bytes();
        let mut hex = String::with_capacity(128);
        for b in bytes {
            use std::fmt::Write as _;
            write!(&mut hex, "{:02x}", b).unwrap();
        }
        hex
    }

    #[test]
    fn verify_detached_accepts_valid_signature() {
        let (sk, pk_hex) = fixture_keypair();
        let manifest = b"pi-0 = 10.0.0.1:50051\npi-1 = 10.0.0.2:50051\n";
        let sig = sig_hex(&sk, manifest);
        verify_detached(manifest, &sig, &pk_hex).expect("good sig must verify");
    }

    #[test]
    fn verify_detached_accepts_trailing_newlines_in_hex() {
        let (sk, pk_hex) = fixture_keypair();
        let manifest = b"pi-0 = 10.0.0.1:50051\n";
        let mut sig = sig_hex(&sk, manifest);
        sig.push('\n');
        let pk_with_newline = format!("{}\r\n", pk_hex);
        verify_detached(manifest, &sig, &pk_with_newline).expect("trailing newlines tolerated");
    }

    #[test]
    fn verify_detached_rejects_tampered_manifest() {
        let (sk, pk_hex) = fixture_keypair();
        let original = b"pi-0 = 10.0.0.1:50051\n";
        let sig = sig_hex(&sk, original);
        // Operator-side attacker swaps the address.
        let tampered = b"pi-0 = 10.0.0.99:50051\n";
        let err = verify_detached(tampered, &sig, &pk_hex).expect_err("tamper must fail");
        match err {
            ClusterError::Transport { reason, .. } => {
                assert!(
                    reason.contains("signature verification failed"),
                    "unexpected reason: {}",
                    reason
                );
            }
            other => panic!("expected Transport, got {:?}", other),
        }
    }

    #[test]
    fn verify_detached_rejects_wrong_pubkey() {
        let (sk, _pk_hex) = fixture_keypair();
        let manifest = b"pi-0 = 10.0.0.1:50051\n";
        let sig = sig_hex(&sk, manifest);
        // Different pubkey, all zeros — won't match.
        let bad_pk = "00".repeat(32);
        let err = verify_detached(manifest, &sig, &bad_pk).expect_err("wrong key must fail");
        match err {
            ClusterError::Transport { .. } => {}
            other => panic!("expected Transport, got {:?}", other),
        }
    }

    #[test]
    fn verify_detached_rejects_short_signature() {
        let (_sk, pk_hex) = fixture_keypair();
        // 127 hex chars instead of 128 — operator typo / truncated file.
        let short = "ab".repeat(63) + "a";
        let err = verify_detached(b"x", &short, &pk_hex).expect_err("short sig must fail");
        match err {
            ClusterError::Transport { reason, .. } => {
                assert!(reason.contains("expected 128 hex chars"), "msg: {}", reason);
            }
            other => panic!("expected Transport, got {:?}", other),
        }
    }

    #[test]
    fn verify_detached_rejects_non_hex_chars() {
        let (_sk, pk_hex) = fixture_keypair();
        // 128 chars but with one non-hex character.
        let bad = "z".repeat(128);
        let err = verify_detached(b"x", &bad, &pk_hex).expect_err("non-hex must fail");
        match err {
            ClusterError::Transport { reason, .. } => {
                assert!(reason.contains("non-hex char"), "msg: {}", reason);
            }
            other => panic!("expected Transport, got {:?}", other),
        }
    }
}
