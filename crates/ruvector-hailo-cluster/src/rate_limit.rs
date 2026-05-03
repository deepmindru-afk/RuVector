//! Per-peer rate limiting for the worker (ADR-172 §3b MEDIUM mitigation,
//! iter 104).
//!
//! Prevents a single misbehaving client from saturating `/dev/hailo0` or
//! thrashing the LRU cache. Identity precedence: mTLS client-cert
//! subject (if present) > peer IP. Rotating IPs alone can't bypass a
//! per-cert limit; rotating certs requires re-issuance from the
//! authorized CA, which is the §1b mitigation's whole point.
//!
//! # Wire-up
//!
//! The worker constructs a single [`RateLimiter`] at startup and clones
//! it into a tonic `Interceptor` closure. The interceptor reads the peer
//! identity off `Request::extensions()` (`TlsConnectInfo<TcpConnectInfo>`
//! when TLS is on, plain `TcpConnectInfo` otherwise) and consults the
//! limiter. Returns `Status::resource_exhausted` on quota breach.
//!
//! Opt-in via `RUVECTOR_RATE_LIMIT_RPS` env var on the worker. Default
//! `0 = disabled` for back-compat.

use dashmap::DashMap;
use governor::{
    clock::DefaultClock,
    state::{InMemoryState, NotKeyed},
    Quota, RateLimiter as GovRateLimiter,
};
use std::num::NonZeroU32;
use std::sync::Arc;

/// Per-peer governor bucket. Aliased so the `RateLimiter` struct field
/// type stays readable (clippy `type_complexity`).
type Bucket = Arc<GovRateLimiter<NotKeyed, InMemoryState, DefaultClock>>;

/// Result type for [`RateLimiter::check`]. Carries a marker so clippy
/// doesn't flag `Result<(), ()>`; the marker doubles as a tracing-
/// friendly cause string for future debugging.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RateLimitDenied;
use tonic::transport::server::TcpConnectInfo;
#[cfg(feature = "tls")]
use tonic::transport::server::TlsConnectInfo;
use tonic::Request;

/// Extract a stable per-peer identity from a `tonic::Request` for
/// rate-limit bucketing. Precedence:
///
///   1. mTLS leaf cert sha256 (first 8 bytes hex) — `cert:<16hex>`
///   2. Peer IP                                   — `ip:<addr>`
///   3. Fallback                                  — `"anonymous"`
///
/// Cert hash is preferred because cert rotation requires CA re-issuance
/// (ADR-172 §1b mTLS); rotating IPs alone can't bypass a per-cert limit.
/// 8 bytes is enough collision-resistance for rate-limiter bucketing
/// (2^64 cert subjects before any collision even matters).
pub fn peer_identity<T>(req: &Request<T>) -> String {
    let ext = req.extensions();
    #[cfg(feature = "tls")]
    if let Some(tls) = ext.get::<TlsConnectInfo<TcpConnectInfo>>() {
        if let Some(certs) = tls.peer_certs() {
            if let Some(leaf) = certs.first() {
                use sha2::{Digest, Sha256};
                let mut h = Sha256::new();
                h.update(leaf.as_ref());
                let d = h.finalize();
                let mut hex = String::with_capacity(21);
                hex.push_str("cert:");
                for b in &d.as_slice()[..8] {
                    use std::fmt::Write as _;
                    write!(&mut hex, "{:02x}", b).unwrap();
                }
                return hex;
            }
        }
        if let Some(addr) = tls.get_ref().remote_addr() {
            return format!("ip:{}", addr.ip());
        }
    }
    if let Some(tcp) = ext.get::<TcpConnectInfo>() {
        if let Some(addr) = tcp.remote_addr() {
            return format!("ip:{}", addr.ip());
        }
    }
    "anonymous".into()
}

/// Per-peer leaky-bucket rate limiter.
///
/// Keyed by an opaque identity string (mTLS subject when known, peer IP
/// otherwise — the worker's interceptor decides). Backed by `governor`'s
/// in-memory state machine, no clock dependency, no allocation in the
/// fast path once the per-peer entry exists.
#[derive(Clone)]
pub struct RateLimiter {
    /// `(rps, burst)` — both `>0` when the limiter is active.
    quota: Quota,
    /// Sharded concurrent map (`dashmap`) so the hot path doesn't take
    /// a single global lock under load.
    buckets: Arc<DashMap<String, Bucket>>,
}

impl RateLimiter {
    /// Build a limiter that lets each peer issue `rps` requests per
    /// second with up to `burst` extra credit. Returns `None` when
    /// `rps == 0` so the caller can short-circuit the interceptor at
    /// build time.
    pub fn new(rps: u32, burst: u32) -> Option<Self> {
        let rps = NonZeroU32::new(rps)?;
        let burst = NonZeroU32::new(burst.max(1)).expect("burst clamped to >=1");
        Some(Self {
            quota: Quota::per_second(rps).allow_burst(burst),
            buckets: Arc::new(DashMap::new()),
        })
    }

    /// Build from `RUVECTOR_RATE_LIMIT_RPS` + `RUVECTOR_RATE_LIMIT_BURST`
    /// env vars. Returns `None` if either is missing or zero.
    pub fn from_env() -> Option<Self> {
        let rps = std::env::var("RUVECTOR_RATE_LIMIT_RPS")
            .ok()
            .and_then(|s| s.parse::<u32>().ok())
            .unwrap_or(0);
        if rps == 0 {
            return None;
        }
        let burst = std::env::var("RUVECTOR_RATE_LIMIT_BURST")
            .ok()
            .and_then(|s| s.parse::<u32>().ok())
            .unwrap_or(rps);
        Self::new(rps, burst)
    }

    /// Try to consume one request slot for `peer`. Returns `Ok(())` if
    /// allowed, `Err(RateLimitDenied)` if rate-limited. The interceptor
    /// only needs allow/deny; we deliberately don't compute a precise
    /// retry-after hint (governor's quanta-clock variant requires a
    /// separate clock dep that's not worth the build cost for one
    /// optional metadata field clients can compute themselves).
    pub fn check(&self, peer: &str) -> Result<(), RateLimitDenied> {
        let bucket = if let Some(b) = self.buckets.get(peer) {
            Arc::clone(b.value())
        } else {
            let lim = Arc::new(GovRateLimiter::direct(self.quota));
            // `entry().or_insert_with` would also work; explicit insert
            // keeps the read path lock-scope-narrow when the entry exists.
            self.buckets
                .entry(peer.to_string())
                .or_insert_with(|| Arc::clone(&lim));
            lim
        };
        bucket.check().map_err(|_| RateLimitDenied)
    }

    /// Total number of unique peers tracked. Useful for stats / metrics.
    pub fn tracked_peers(&self) -> usize {
        self.buckets.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn first_request_allowed_then_burst_exhausts() {
        // 1 rps, burst 3 — 3 requests through, 4th rate-limited.
        let r = RateLimiter::new(1, 3).expect("non-zero quota");
        assert!(r.check("peer-a").is_ok());
        assert!(r.check("peer-a").is_ok());
        assert!(r.check("peer-a").is_ok());
        assert!(r.check("peer-a").is_err(), "4th in burst should be rate-limited");
    }

    #[test]
    fn separate_peers_have_independent_buckets() {
        // Each peer gets its own quota; one bad client shouldn't impact others.
        let r = RateLimiter::new(1, 2).expect("non-zero quota");
        assert!(r.check("peer-a").is_ok());
        assert!(r.check("peer-a").is_ok());
        assert!(r.check("peer-a").is_err());
        // peer-b is untouched, still within burst.
        assert!(r.check("peer-b").is_ok());
        assert!(r.check("peer-b").is_ok());
        assert!(r.check("peer-b").is_err());
        // Both peers tracked.
        assert_eq!(r.tracked_peers(), 2);
    }

    #[test]
    fn zero_rps_returns_none_so_caller_skips_interceptor() {
        assert!(RateLimiter::new(0, 0).is_none());
    }

    // Iter 197 — both tests below mutate the same process-global env
    // vars (`RUVECTOR_RATE_LIMIT_RPS` / `_BURST`). Cargo runs tests in
    // parallel by default, so without serialization the wipe in
    // `from_env_disabled_when_unset` could race the set in
    // `from_env_picks_up_rps_with_default_burst` and either test
    // could see the other's mutation mid-flight. iter-190's session
    // sweep caught this as an intermittent failure (1 in N runs).
    // Process-local Mutex acquired for the duration of each env-
    // touching test serializes access without pulling a heavyweight
    // crate like `serial_test`.
    fn env_lock() -> std::sync::MutexGuard<'static, ()> {
        use std::sync::{Mutex, OnceLock};
        static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
        LOCK.get_or_init(|| Mutex::new(())).lock().unwrap_or_else(|p| p.into_inner())
    }

    #[test]
    fn from_env_disabled_when_unset() {
        let _guard = env_lock();
        unsafe {
            std::env::remove_var("RUVECTOR_RATE_LIMIT_RPS");
            std::env::remove_var("RUVECTOR_RATE_LIMIT_BURST");
        }
        assert!(RateLimiter::from_env().is_none());
    }

    #[test]
    fn from_env_picks_up_rps_with_default_burst() {
        let _guard = env_lock();
        // Set both -> Some(_); rps non-zero is the only requirement.
        unsafe {
            std::env::set_var("RUVECTOR_RATE_LIMIT_RPS", "5");
        }
        let r = RateLimiter::from_env().expect("rps=5 means active");
        // Burst defaults to rps when unset; 5 burst -> first 5 allowed.
        for _ in 0..5 {
            assert!(r.check("peer-x").is_ok());
        }
        assert!(r.check("peer-x").is_err());
        unsafe {
            std::env::remove_var("RUVECTOR_RATE_LIMIT_RPS");
        }
    }

    // ---- peer_identity tests ----

    #[test]
    fn peer_identity_no_extensions_returns_anonymous() {
        let req: Request<()> = Request::new(());
        assert_eq!(peer_identity(&req), "anonymous");
    }

    #[test]
    fn peer_identity_falls_back_to_peer_ip_when_no_cert() {
        let mut req: Request<()> = Request::new(());
        let addr: std::net::SocketAddr = "10.0.0.7:50051".parse().unwrap();
        req.extensions_mut().insert(TcpConnectInfo {
            local_addr: None,
            remote_addr: Some(addr),
        });
        assert_eq!(peer_identity(&req), "ip:10.0.0.7");
    }

    // The cert-hash path is not unit-testable because tonic's
    // `TlsConnectInfo` has no public constructor — it's produced by
    // the server's TLS handshake. End-to-end verification lives in
    // `tests/rate_limit_interceptor.rs` (always-on burst+deny flow)
    // and `tests/mtls_roundtrip.rs` (cert-issued client identities).
}
