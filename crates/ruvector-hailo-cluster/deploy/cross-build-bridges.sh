#!/usr/bin/env bash
# Cross-compile every sensor bridge from x86_64 → aarch64, ready for
# deploy on a Pi 5 (cognitum-v0 or any aarch64 Linux box).
#
# Bridges cross-compiled (none link libhailort, so all three work
# cleanly without a Hailo aarch64 sysroot):
#
#   ruvector-mmwave-bridge   60 GHz mmWave radar UART/UDP
#   ruview-csi-bridge        RuView ADR-018 CSI UDP
#   ruvllm-bridge            ruvllm JSONL stdin/stdout adapter
#
# Companion to deploy/cross-build.sh (which handles the worker-side
# CLIs: embed, stats, fakeworker, cluster-bench).
#
# Usage:
#   bash cross-build-bridges.sh [--deploy <pi-tailnet-or-local-name>]
#
#   --deploy NAME   rsync the three bridges to NAME:/usr/local/bin/
#                   (uses tailscale ssh if NAME is on the tailnet,
#                    plain ssh otherwise; expects passwordless ssh).
#
# Re-run idempotently. cargo's incremental cache makes re-runs fast.

set -euo pipefail

CRATE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. && pwd)"
TARGET="aarch64-unknown-linux-gnu"
BINS=(ruvector-mmwave-bridge ruview-csi-bridge ruvllm-bridge)

DEPLOY_HOST=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --deploy)
      DEPLOY_HOST="${2:-}"
      [[ -z "$DEPLOY_HOST" ]] && { echo "--deploy needs a host" >&2; exit 1; }
      shift 2
      ;;
    -h|--help)
      sed -n '2,30p' "$0" | sed 's/^# \?//'
      exit 0
      ;;
    *)
      echo "unknown arg: $1" >&2; exit 1
      ;;
  esac
done

echo "==> [1/5] verify rustup target $TARGET"
if ! rustup target list --installed | grep -q "^$TARGET\$"; then
  echo "    installing"
  rustup target add "$TARGET"
else
  echo "    already installed"
fi

echo "==> [2/5] verify aarch64 C linker"
if ! command -v aarch64-linux-gnu-gcc >/dev/null 2>&1; then
  echo "    aarch64-linux-gnu-gcc not found." >&2
  echo "    Install with:  sudo apt-get install -y gcc-aarch64-linux-gnu" >&2
  exit 1
fi
echo "    $(which aarch64-linux-gnu-gcc)"

echo "==> [3/5] cross-compile all three bridges"
# Iter-122 reminder: the ruvector workspace ships a RUSTFLAGS=-C
# link-arg=-fuse-ld=mold default that breaks the xtensa/aarch64 cross
# link. `env -u RUSTFLAGS` strips it for this build only without
# touching the operator's shell env.
cd "$CRATE_DIR"
for bin in "${BINS[@]}"; do
  echo "    [+] $bin"
  env -u RUSTFLAGS \
      CARGO_TARGET_AARCH64_UNKNOWN_LINUX_GNU_LINKER=aarch64-linux-gnu-gcc \
      cargo build --release --target "$TARGET" --bin "$bin"
done

echo "==> [4/5] verify each artifact is aarch64 ELF"
ALL_OK=1
for bin in "${BINS[@]}"; do
  elf="target/$TARGET/release/$bin"
  if file "$elf" | grep -q 'ARM aarch64'; then
    sz="$(stat --format='%s' "$elf")"
    echo "    ✓ $bin  ($((sz / 1024)) KB)"
  else
    echo "    ✗ $bin  NOT aarch64" >&2
    ALL_OK=0
  fi
done
[[ $ALL_OK -eq 1 ]] || { echo "one or more bins failed verification" >&2; exit 2; }

echo "==> [5/5] deploy"
if [[ -z "$DEPLOY_HOST" ]]; then
  echo "    skipped (no --deploy <host>)"
  echo
  echo "Artifacts ready at:"
  for bin in "${BINS[@]}"; do
    echo "    $CRATE_DIR/target/$TARGET/release/$bin"
  done
  echo
  echo "To rsync to a Pi:"
  echo "    bash $0 --deploy cognitum-v0"
  exit 0
fi

echo "    deploying to $DEPLOY_HOST:/usr/local/bin/"
for bin in "${BINS[@]}"; do
  src="target/$TARGET/release/$bin"
  # Use scp for simplicity + universality; rsync would be slightly
  # faster on re-runs but isn't always installed on minimal Pis.
  scp -q "$src" "root@${DEPLOY_HOST}:/usr/local/bin/$bin"
  ssh "root@${DEPLOY_HOST}" "chmod +x /usr/local/bin/$bin"
  echo "    ✓ $bin"
done

echo
echo "Verify on the target:"
echo "    ssh root@${DEPLOY_HOST} 'for b in ${BINS[*]}; do /usr/local/bin/\$b --version; done'"
echo
echo "Then install the systemd service (if not already done):"
echo "    ssh root@${DEPLOY_HOST}"
echo "    cd /path/to/ruvector/crates/ruvector-hailo-cluster/deploy"
echo "    sudo bash install-bridge.sh /usr/local/bin/ruvector-mmwave-bridge   # mmwave"
echo "    sudo bash install-ruview-csi-bridge.sh /usr/local/bin/ruview-csi-bridge"
