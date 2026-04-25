#!/usr/bin/env bash
# serve_public.sh — start the qusim DSE GUI behind a Cloudflare named tunnel.
#
# Defaults are tuned for ~3 concurrent users (e.g. teachers trying the tool).
# The Dash app binds to 127.0.0.1 so only cloudflared (or the local browser)
# can reach it. Both processes are killed on Ctrl+C.
#
# Prerequisites on the target machine:
#   1. Python venv at ./.venv with the gui requirements installed:
#        python3 -m venv .venv
#        .venv/bin/pip install -r gui/requirements.txt
#   2. cloudflared installed (https://pkg.cloudflare.com or the .deb release).
#   3. Cloudflare named tunnel set up. First time only:
#        cloudflared tunnel login                       # browser auth
#        cloudflared tunnel create qusim-dse            # writes ~/.cloudflared/<UUID>.json
#        cloudflared tunnel route dns qusim-dse qusim.gonzalvo.dev
#      Or copy ~/.cloudflared/ from a machine that already has it.
#
# Environment overrides:
#   QUSIM_HOST       (default: 127.0.0.1) bind address for the Dash app
#   QUSIM_PORT       (default: 8050)      bind port for the Dash app
#   TUNNEL_NAME      (default: qusim-dse) name passed to `cloudflared tunnel run`
#   PUBLIC_URL       (default: derived from config.yml) URL printed for sharing
#   SKIP_TUNNEL=1    start the app without the tunnel
#   SKIP_APP=1       start only the tunnel (assume app is already running)

set -euo pipefail

# Resolve the project root from this script's location so it works from
# anywhere (e.g. cron, systemd, another working directory).
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

QUSIM_HOST="${QUSIM_HOST:-127.0.0.1}"
QUSIM_PORT="${QUSIM_PORT:-8050}"
TUNNEL_NAME="${TUNNEL_NAME:-qusim-dse}"
LOG_DIR="${LOG_DIR:-/tmp/qusim-serve}"
mkdir -p "$LOG_DIR"
APP_LOG="$LOG_DIR/app.log"
TUNNEL_LOG="$LOG_DIR/cloudflared.log"

APP_PID=""
TUNNEL_PID=""

cleanup() {
  echo
  echo "[serve] shutting down..."
  if [[ -n "$TUNNEL_PID" ]] && kill -0 "$TUNNEL_PID" 2>/dev/null; then
    kill "$TUNNEL_PID" 2>/dev/null || true
  fi
  if [[ -n "$APP_PID" ]] && kill -0 "$APP_PID" 2>/dev/null; then
    kill "$APP_PID" 2>/dev/null || true
  fi
  wait 2>/dev/null || true
  echo "[serve] stopped."
}
trap cleanup EXIT INT TERM

# --- Preflight ----------------------------------------------------------------
if [[ "${SKIP_APP:-0}" != "1" ]]; then
  if [[ ! -x "$PROJECT_ROOT/.venv/bin/python" ]]; then
    echo "[serve] ERROR: $PROJECT_ROOT/.venv/bin/python not found." >&2
    echo "        Run: python3 -m venv .venv && .venv/bin/pip install -r gui/requirements.txt" >&2
    exit 1
  fi
fi

if [[ "${SKIP_TUNNEL:-0}" != "1" ]]; then
  if ! command -v cloudflared >/dev/null 2>&1; then
    echo "[serve] ERROR: cloudflared not on PATH." >&2
    echo "        Install: https://pkg.cloudflare.com (or download the .deb release)" >&2
    exit 1
  fi
  if [[ ! -f "$HOME/.cloudflared/cert.pem" ]]; then
    echo "[serve] ERROR: ~/.cloudflared/cert.pem missing." >&2
    echo "        Run: cloudflared tunnel login" >&2
    exit 1
  fi
  if ! cloudflared tunnel list 2>/dev/null | grep -q "[[:space:]]$TUNNEL_NAME[[:space:]]"; then
    echo "[serve] ERROR: tunnel '$TUNNEL_NAME' not found in your account." >&2
    echo "        Run: cloudflared tunnel create $TUNNEL_NAME" >&2
    echo "        Then: cloudflared tunnel route dns $TUNNEL_NAME <your.hostname>" >&2
    exit 1
  fi
fi

# --- Start the Dash app -------------------------------------------------------
if [[ "${SKIP_APP:-0}" != "1" ]]; then
  echo "[serve] starting Dash app on http://$QUSIM_HOST:$QUSIM_PORT (log: $APP_LOG)"
  QUSIM_HOST="$QUSIM_HOST" QUSIM_PORT="$QUSIM_PORT" \
    "$PROJECT_ROOT/.venv/bin/python" "$PROJECT_ROOT/gui/app.py" \
    > "$APP_LOG" 2>&1 &
  APP_PID=$!

  # Wait for the port to start accepting connections (max ~15s).
  for _ in $(seq 1 30); do
    if (echo > "/dev/tcp/$QUSIM_HOST/$QUSIM_PORT") 2>/dev/null; then
      break
    fi
    if ! kill -0 "$APP_PID" 2>/dev/null; then
      echo "[serve] ERROR: Dash app exited before opening the port. Tail of $APP_LOG:" >&2
      tail -30 "$APP_LOG" >&2 || true
      exit 1
    fi
    sleep 0.5
  done
fi

# --- Start the tunnel ---------------------------------------------------------
if [[ "${SKIP_TUNNEL:-0}" != "1" ]]; then
  # Try to derive a friendly URL from ~/.cloudflared/config.yml so we can print it.
  derived_url=""
  if [[ -f "$HOME/.cloudflared/config.yml" ]]; then
    host=$(grep -E '^[[:space:]]*-[[:space:]]*hostname:' "$HOME/.cloudflared/config.yml" \
             | head -1 | sed -E 's/.*hostname:[[:space:]]*//')
    if [[ -n "$host" ]]; then
      derived_url="https://$host"
    fi
  fi
  PUBLIC_URL="${PUBLIC_URL:-${derived_url:-(check your tunnel config for the hostname)}}"

  echo "[serve] starting cloudflared tunnel '$TUNNEL_NAME' (log: $TUNNEL_LOG)"
  cloudflared tunnel run "$TUNNEL_NAME" > "$TUNNEL_LOG" 2>&1 &
  TUNNEL_PID=$!

  # Wait until cloudflared logs at least one registered connection.
  for _ in $(seq 1 40); do
    if grep -q "Registered tunnel connection" "$TUNNEL_LOG" 2>/dev/null; then
      break
    fi
    if ! kill -0 "$TUNNEL_PID" 2>/dev/null; then
      echo "[serve] ERROR: cloudflared exited before connecting. Tail of $TUNNEL_LOG:" >&2
      tail -30 "$TUNNEL_LOG" >&2 || true
      exit 1
    fi
    sleep 0.5
  done

  echo
  echo "  Public URL:  $PUBLIC_URL"
  echo "  App log:     $APP_LOG"
  echo "  Tunnel log:  $TUNNEL_LOG"
  echo
fi

echo "[serve] running. Ctrl+C to stop."

# Wait on whichever child processes we started; exit if any of them dies.
while true; do
  if [[ -n "$APP_PID" ]] && ! kill -0 "$APP_PID" 2>/dev/null; then
    echo "[serve] Dash app exited. Stopping." >&2
    exit 1
  fi
  if [[ -n "$TUNNEL_PID" ]] && ! kill -0 "$TUNNEL_PID" 2>/dev/null; then
    echo "[serve] cloudflared exited. Stopping." >&2
    exit 1
  fi
  sleep 2
done
