#!/usr/bin/env bash
# Install ruvector-hailo-worker on a Pi 5 + AI HAT+.
#
# Run on the Pi (not on a dev host) after building the binary with:
#   cargo build --release --features hailo --bin ruvector-hailo-worker
#
# Idempotent — re-run after upgrading the binary.
#
# What this drops on the Pi (ADR-172 §3a iter-106 drop-root):
#   /usr/local/bin/ruvector-hailo-worker         (binary)
#   /var/lib/ruvector-hailo/                     (state dir, owned by
#                                                 ruvector-worker:ruvector-worker)
#   /etc/ruvector-hailo.env                      (config; preserved if
#                                                 it already exists)
#   /etc/systemd/system/ruvector-hailo-worker.service
#   /etc/udev/rules.d/99-hailo-ruvector.rules    (gives the
#                                                 ruvector-worker
#                                                 group rw on /dev/hailo*)
#   system user: ruvector-worker (no home, no shell)
#
# Usage:
#   sudo bash install.sh /path/to/ruvector-hailo-worker /path/to/models-dir

set -euo pipefail

if [[ $EUID -ne 0 ]]; then
  echo "must run as root (use sudo)" >&2; exit 1
fi
if [[ $# -lt 2 ]]; then
  echo "usage: $0 <path/to/ruvector-hailo-worker> <path/to/models-dir>" >&2
  echo "  models-dir must contain model.hef, vocab.txt, special_tokens.json" >&2
  exit 1
fi

WORKER_BIN="$1"
MODELS_SRC="$2"

if [[ ! -x "$WORKER_BIN" ]]; then
  echo "binary not executable: $WORKER_BIN" >&2; exit 1
fi
if [[ ! -d "$MODELS_SRC" ]]; then
  echo "models dir not found: $MODELS_SRC" >&2; exit 1
fi
if [[ ! -f "$MODELS_SRC/model.hef" ]]; then
  echo "warning: $MODELS_SRC/model.hef missing — worker will fail to start" >&2
fi

DEPLOY_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUVECTOR_USER="ruvector-worker"
RUVECTOR_GROUP="ruvector-worker"

echo "==> ensure system user $RUVECTOR_USER exists"
# `useradd --system` returns 9 if the user already exists; treat as ok.
# Idempotent re-runs are the common case (binary upgrades).
if ! getent passwd "$RUVECTOR_USER" >/dev/null; then
  useradd \
    --system \
    --no-create-home \
    --home-dir /var/lib/ruvector-hailo \
    --shell /usr/sbin/nologin \
    --comment "ruvector Hailo worker (ADR-172 §3a)" \
    "$RUVECTOR_USER"
  echo "    -> created"
else
  echo "    -> already exists"
fi

echo "==> install binary"
install -o root -g root -m 0755 "$WORKER_BIN" /usr/local/bin/ruvector-hailo-worker

echo "==> install models -> /var/lib/ruvector-hailo/models/all-minilm-l6-v2"
install -d -o "$RUVECTOR_USER" -g "$RUVECTOR_GROUP" -m 0750 \
  /var/lib/ruvector-hailo \
  /var/lib/ruvector-hailo/models \
  /var/lib/ruvector-hailo/models/all-minilm-l6-v2
cp -a "$MODELS_SRC/." /var/lib/ruvector-hailo/models/all-minilm-l6-v2/
chown -R "$RUVECTOR_USER":"$RUVECTOR_GROUP" /var/lib/ruvector-hailo

echo "==> install /etc/ruvector-hailo.env (skipped if exists)"
if [[ ! -f /etc/ruvector-hailo.env ]]; then
  install -o root -g root -m 0644 "$DEPLOY_DIR/ruvector-hailo.env.example" /etc/ruvector-hailo.env
  echo "    -> wrote default; edit if non-default bind/model dir wanted"
else
  echo "    -> existing /etc/ruvector-hailo.env preserved"
fi

echo "==> install udev rule (gives $RUVECTOR_GROUP group rw on /dev/hailo*)"
install -o root -g root -m 0644 \
  "$DEPLOY_DIR/99-hailo-ruvector.rules" \
  /etc/udev/rules.d/99-hailo-ruvector.rules
udevadm control --reload-rules
# Trigger every hailo device the kernel currently sees so existing
# nodes pick up the new ownership without a reboot.
for dev in /dev/hailo*; do
  if [[ -e "$dev" ]]; then
    udevadm trigger "$dev" || true
  fi
done

echo "==> install systemd unit"
install -o root -g root -m 0644 \
  "$DEPLOY_DIR/ruvector-hailo-worker.service" \
  /etc/systemd/system/ruvector-hailo-worker.service

echo "==> daemon-reload + enable"
systemctl daemon-reload
systemctl enable ruvector-hailo-worker.service

echo
echo "Installed (running as $RUVECTOR_USER, no root)."
echo "To start now:"
echo "    sudo systemctl start ruvector-hailo-worker"
echo "Tail logs:"
echo "    journalctl -u ruvector-hailo-worker -f"
echo "Verify drop-root:"
echo "    ps -o user,pid,cmd -C ruvector-hailo-worker"
echo "    ls -l /dev/hailo0   # expect group ${RUVECTOR_GROUP}"
