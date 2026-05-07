#!/usr/bin/env bash
# health_check.sh — probe the public URL and relaunch serve_public.sh if down.
#
# Designed to be called from cron every few minutes. Silent on success;
# logs to $LOG_DIR/health.log on every failure / restart.
#
# Behaviour:
#   - HTTP 200 within timeout    → exit 0 silently.
#   - 3 consecutive failures      → kill any existing serve_public.sh
#                                   supervisor (and its children), then start
#                                   a fresh one in the background using the
#                                   same TUNNEL_* env this script was given.
#
# A short retry loop avoids triggering a relaunch during the brief 502
# window when the supervisor is respawning the Dash app (which the
# pull_and_restart.sh cron does on every new commit).
#
# Environment (all optional — defaults match serve_public.sh):
#   QUADRIS_URL         URL to probe (default: https://$TUNNEL_HOSTNAME)
#   TUNNEL_NAME       passed through to serve_public.sh
#   TUNNEL_UUID       passed through to serve_public.sh
#   TUNNEL_HOSTNAME   passed through to serve_public.sh (also used to derive
#                     QUADRIS_URL when not set)
#   QUADRIS_HOST        passed through to serve_public.sh
#   QUADRIS_PORT        passed through to serve_public.sh
#   LOG_DIR           default /tmp/quadris-serve
#   PROBE_TIMEOUT     curl --max-time, default 10
#   PROBE_RETRIES     attempts before declaring down, default 3
#   PROBE_RETRY_GAP   seconds between retries, default 5
#
# Cron suggestion (this laptop, quadris.gonzalvo.dev):
#   */5 * * * * QUADRIS_URL=https://quadris.gonzalvo.dev TUNNEL_NAME=quadris-dse \
#       TUNNEL_UUID=<uuid> TUNNEL_HOSTNAME=quadris.gonzalvo.dev \
#       /home/alejandro/dev/quadris/scripts/health_check.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

TUNNEL_HOSTNAME="${TUNNEL_HOSTNAME:-upv-dse.gonzalvo.dev}"
QUADRIS_URL="${QUADRIS_URL:-https://$TUNNEL_HOSTNAME}"

LOG_DIR="${LOG_DIR:-/tmp/quadris-serve}"
mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/health.log"
SUPERVISOR_OUT="$LOG_DIR/supervisor.out"

PROBE_TIMEOUT="${PROBE_TIMEOUT:-10}"
PROBE_RETRIES="${PROBE_RETRIES:-3}"
PROBE_RETRY_GAP="${PROBE_RETRY_GAP:-5}"

log() {
  echo "[$(date -Iseconds)] $*" >> "$LOG"
}

probe_once() {
  local code
  code=$(curl -s -o /dev/null --max-time "$PROBE_TIMEOUT" -w "%{http_code}" "$QUADRIS_URL" || echo "000")
  # Treat any 2xx as healthy. Cloudflare returns 502/520-525 when the origin
  # is unreachable, 530 when the tunnel is gone — all flagged as down.
  [[ "$code" =~ ^2[0-9][0-9]$ ]]
}

# Match the supervisor by absolute script path so we don't kill someone
# else's bash sitting on a coincidentally-named script.
supervisor_pid() {
  pgrep -f "bash.*$PROJECT_ROOT/scripts/serve_public.sh" | head -1
}

# --- Probe --------------------------------------------------------------------
attempt=1
while (( attempt <= PROBE_RETRIES )); do
  if probe_once; then
    # Healthy — silent exit.
    exit 0
  fi
  if (( attempt < PROBE_RETRIES )); then
    sleep "$PROBE_RETRY_GAP"
  fi
  attempt=$(( attempt + 1 ))
done

log "DOWN: $QUADRIS_URL failed $PROBE_RETRIES probes (timeout=${PROBE_TIMEOUT}s)"

# --- Stop any existing supervisor --------------------------------------------
existing=$(supervisor_pid || true)
if [[ -n "${existing:-}" ]]; then
  log "killing stale supervisor pid=$existing"
  kill -TERM "$existing" 2>/dev/null || true
  # Give the trap a few seconds to clean up child processes.
  for _ in $(seq 1 10); do
    if ! kill -0 "$existing" 2>/dev/null; then break; fi
    sleep 1
  done
  if kill -0 "$existing" 2>/dev/null; then
    log "supervisor pid=$existing did not exit on TERM, sending KILL"
    kill -KILL "$existing" 2>/dev/null || true
    sleep 1
  fi
fi

# Belt and braces: kill any stragglers (orphaned tunnels or app from a
# crashed supervisor that didn't run its trap).
pkill -f "$PROJECT_ROOT/.venv/bin/python.*$PROJECT_ROOT/gui/app.py" 2>/dev/null || true
pkill -f "cloudflared.*tunnel run.*${TUNNEL_NAME:-}" 2>/dev/null || true

# --- Start a fresh supervisor -------------------------------------------------
# Pass through the same env so the new supervisor binds to the right tunnel.
log "starting fresh supervisor"
nohup env \
  ${TUNNEL_NAME:+TUNNEL_NAME="$TUNNEL_NAME"} \
  ${TUNNEL_UUID:+TUNNEL_UUID="$TUNNEL_UUID"} \
  ${TUNNEL_HOSTNAME:+TUNNEL_HOSTNAME="$TUNNEL_HOSTNAME"} \
  ${QUADRIS_HOST:+QUADRIS_HOST="$QUADRIS_HOST"} \
  ${QUADRIS_PORT:+QUADRIS_PORT="$QUADRIS_PORT"} \
  "$PROJECT_ROOT/scripts/serve_public.sh" >> "$SUPERVISOR_OUT" 2>&1 &
disown

# Wait briefly so we can log whether it managed to bring the URL back.
sleep 15
if probe_once; then
  log "RECOVERED: $QUADRIS_URL is back"
else
  log "still down after relaunch attempt — see $SUPERVISOR_OUT and $LOG_DIR/{app,cloudflared}.log"
fi
