#!/usr/bin/env bash
# push-to-cluster.sh — copy ruview-vitals-worker + deploy bundle to a
# cognitum cluster Pi via Tailscale SSH and run the install script.
#
# Iter-7 helper used during ADR-183 Tier 1 cluster bring-up. Same
# spirit as ADR-179's `cross-build-bridges.sh` + push pattern, scoped
# to one node per invocation so failures are obvious.
#
# Usage:
#   bash push-to-cluster.sh <hostname> [<node_name>]
#
#   hostname   Tailscale hostname (e.g. cognitum-cluster-2). MUST be
#              reachable as root@<hostname> via Tailscale SSH.
#   node_name  Override RUVIEW_VITALS_NODE_NAME on the target. Defaults
#              to <hostname>.
#
# Env overrides:
#   BRAIN_URL   default http://192.168.1.123:9876 (ruvultra LAN brain).
#               Switch to http://cognitum-v0:9876 once Tier 2 stands
#               up the brain there.
#   BIN_PATH    default <repo>/target/aarch64-unknown-linux-gnu/release/
#               ruview-vitals-worker. The cross-build runs with
#               `RUSTFLAGS= cargo build -p ruview-vitals-worker
#               --release --target aarch64-unknown-linux-gnu` (the
#               empty RUSTFLAGS is required because the workspace
#               default forces `-fuse-ld=mold`, which has no aarch64
#               linker on this host).

set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "usage: $0 <hostname> [<node_name>]" >&2
  exit 1
fi

HOST="$1"
NODE_NAME="${2:-$HOST}"
BRAIN_URL="${BRAIN_URL:-http://192.168.1.123:9876}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
BIN_PATH="${BIN_PATH:-$REPO_ROOT/target/aarch64-unknown-linux-gnu/release/ruview-vitals-worker}"

if [[ ! -x "$BIN_PATH" ]]; then
  echo "binary not found: $BIN_PATH" >&2
  echo "build with: RUSTFLAGS= cargo build -p ruview-vitals-worker --release --target aarch64-unknown-linux-gnu --no-default-features" >&2
  exit 1
fi

REMOTE_DIR=/root/adr-183-deploy
echo "==> [$HOST] mkdir $REMOTE_DIR"
ssh "root@$HOST" "mkdir -p $REMOTE_DIR"

echo "==> [$HOST] scp binary + bundle"
scp -q "$BIN_PATH" \
       "$SCRIPT_DIR/ruview-vitals-worker.service" \
       "$SCRIPT_DIR/ruview-vitals-worker.env.example" \
       "$SCRIPT_DIR/install-ruview-vitals-worker.sh" \
       "root@$HOST:$REMOTE_DIR/"

echo "==> [$HOST] install + systemd"
ssh "root@$HOST" "
  set -e
  cd $REMOTE_DIR
  chmod +x ruview-vitals-worker install-ruview-vitals-worker.sh
  bash install-ruview-vitals-worker.sh $REMOTE_DIR/ruview-vitals-worker
  cat > /etc/ruview-vitals-worker.env <<EOF
RUVIEW_VITALS_UDP_LISTEN=0.0.0.0:5005
RUVIEW_VITALS_GRPC_LISTEN=0.0.0.0:50054
RUVIEW_VITALS_BRAIN_URL=$BRAIN_URL
RUVIEW_VITALS_BRAIN_INTERVAL_SECS=30
RUVIEW_VITALS_NODE_NAME=$NODE_NAME
RUVIEW_VITALS_WINDOW_FRAMES=50
RUVIEW_VITALS_VERBOSE=0
RUVIEW_VITALS_LOG=info,ruview_vitals_worker::brain=info
EOF
  systemctl restart ruview-vitals-worker.service
  sleep 2
  systemctl is-active ruview-vitals-worker.service
"

echo "==> [$HOST] post-deploy journal tail:"
ssh "root@$HOST" 'journalctl -u ruview-vitals-worker --no-pager -n 5'

echo "==> [$HOST] done."
