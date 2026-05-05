#!/usr/bin/env bash
# cluster-smoke-test.sh — ADR-183 Tier 2 iter 12
#
# Integration smoke test for the full ruview vitals + brain stack.
# Checks each cluster node (workers + v0 master) for service health,
# gRPC liveness, SONA adaptation progress, and brain reachability.
#
# Exits 0 only when all assertions pass. Non-zero exit on any failure.
#
# Usage:
#   bash cluster-smoke-test.sh [--quiet]
#
#   --quiet  suppress pass lines; show only failures + final verdict

set -euo pipefail

QUIET=0
[[ "${1:-}" == "--quiet" ]] && QUIET=1

# Tailscale IPs / hostnames per ADR-183
WORKERS=(
  "root@100.80.54.16:cognitum-cluster-1:50055"
  "root@100.77.220.24:cognitum-cluster-2:50055"
  "root@100.73.75.53:cognitum-cluster-3:50055"
)
V0_HOST="genesis@100.77.59.83"
V0_BRAIN_PORT=9876
V0_GRPC_PORT=50054
V0_SERVICES=(
  "ruview-vitals-worker"
  "ruview-mcp-brain-mini"
  "ruview-pointcloud"
  "ruview-csi-sink"
)
WORKER_SERVICES=(
  "ruview-vitals-worker"
)

PASS=0
FAIL=0

pass() { PASS=$((PASS + 1)); [[ $QUIET -eq 0 ]] && echo "  [PASS] $*" || true; }
fail() { FAIL=$((FAIL + 1)); echo "  [FAIL] $*"; }

check_service() {
  local host="$1" name="$2"
  local status
  status=$(ssh -o ConnectTimeout=8 -o BatchMode=yes "$host" "systemctl is-active $name 2>&1" 2>&1 || true)
  if [[ "$status" == "active" ]]; then
    pass "$name active on $host"
  else
    fail "$name not active on $host (status=$status)"
  fi
}

check_grpc() {
  local host_ssh="$1" label="$2" port="$3"
  # Use netcat to verify the port is open — full gRPC health RPC would need grpcurl.
  local open
  open=$(ssh -o ConnectTimeout=8 -o BatchMode=yes "$host_ssh" \
    "timeout 3 bash -c 'echo > /dev/tcp/127.0.0.1/$port' 2>&1 && echo open || echo closed" 2>&1 || echo closed)
  if [[ "$open" == "open" ]]; then
    pass "gRPC :$port open on $label"
  else
    fail "gRPC :$port not open on $label"
  fi
}

check_sona_steps() {
  local host="$1" label="$2" min_steps="$3"
  local steps
  steps=$(ssh -o ConnectTimeout=8 -o BatchMode=yes "$host" \
    "journalctl -u ruview-vitals-worker --no-pager -n 500 -o cat 2>&1 | grep 'sona: gradient step' | tail -1 | grep -oP 'steps=\K[0-9]+'" 2>&1 || echo 0)
  steps="${steps//[^0-9]/}"
  steps="${steps:-0}"
  if [[ "$steps" -ge "$min_steps" ]]; then
    pass "SONA steps=$steps (≥ $min_steps) on $label"
  else
    fail "SONA steps=$steps (< $min_steps) on $label — adapter not converging"
  fi
}

check_relay() {
  local host="$1" label="$2"
  local has_relay
  # Check env file for RELAY_TARGETS, then check startup journal (may be old),
  # then check runtime log with a wider window.
  has_relay=$(ssh -o ConnectTimeout=8 -o BatchMode=yes "$host" \
    "grep -cE 'RUVIEW_VITALS_RELAY_TARGETS=.+' /etc/ruview-vitals-worker.env 2>/dev/null || \
     journalctl -u ruview-vitals-worker --no-pager -n 500 -o cat 2>&1 | grep -c 'UDP relay fan-out up' || echo 0" 2>&1 || echo 0)
  has_relay="${has_relay//[^0-9]/}"
  if [[ "${has_relay:-0}" -gt 0 ]]; then
    pass "relay fan-out active on $label"
  else
    fail "relay fan-out not detected on $label"
  fi
}

check_brain_http() {
  local status
  status=$(ssh -o ConnectTimeout=8 -o BatchMode=yes "$V0_HOST" \
    "curl -sf -o /dev/null -w '%{http_code}' http://127.0.0.1:$V0_BRAIN_PORT/health 2>&1 || \
     curl -sf -o /dev/null -w '%{http_code}' http://127.0.0.1:$V0_BRAIN_PORT/ 2>&1 || echo 000" 2>&1 || echo 000)
  status="${status//[^0-9]/}"
  if [[ "${status:-000}" =~ ^(200|204|404|405)$ ]]; then
    pass "brain HTTP /$V0_BRAIN_PORT reachable on v0 (HTTP $status)"
  else
    fail "brain HTTP /$V0_BRAIN_PORT not reachable on v0 (got $status)"
  fi
}

echo "=== ADR-183 cluster smoke test — $(date -u '+%Y-%m-%dT%H:%M:%SZ') ==="
echo ""

echo "-- cognitum-v0 services --"
for svc in "${V0_SERVICES[@]}"; do
  check_service "$V0_HOST" "$svc"
done
check_grpc "$V0_HOST" "cognitum-v0" "$V0_GRPC_PORT"
check_sona_steps "$V0_HOST" "cognitum-v0" 100
check_brain_http

echo ""
echo "-- worker nodes --"
for entry in "${WORKERS[@]}"; do
  host="${entry%%:*}"
  rest="${entry#*:}"
  label="${rest%%:*}"
  port="${rest##*:}"

  for svc in "${WORKER_SERVICES[@]}"; do
    check_service "$host" "$svc"
  done
  check_grpc "$host" "$label" "$port"
  check_sona_steps "$host" "$label" 100
  check_relay "$host" "$label"
done

echo ""
echo "=== Result: $PASS passed, $FAIL failed ==="

if [[ $FAIL -gt 0 ]]; then
  exit 1
fi
exit 0
