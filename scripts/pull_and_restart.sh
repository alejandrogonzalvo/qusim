#!/usr/bin/env bash
# pull_and_restart.sh — fetch origin/main and restart the Dash app if there
# are new commits. Designed to be called from cron every few minutes.
#
# Behavior:
#   - No new commits  →  exit 0 silently (no log line, no kill).
#   - New commits     →  fast-forward pull, then SIGTERM the running gui/app.py
#                        process. The serve_public.sh supervisor respawns it
#                        with the freshly pulled code.
#   - Conflict / non-FF →  log + exit 1 (no destructive action).
#
# Cloudflared is left alone — the public URL stays up while the app
# restarts (the tunnel will return 502 for a few seconds during the gap).
#
# Cron suggestion:
#   */5 * * * * /home/alejandro/dev/quadris/scripts/pull_and_restart.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

LOG_DIR="${LOG_DIR:-/tmp/quadris-serve}"
mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/pull.log"

log() {
  echo "[$(date -Iseconds)] $*" >> "$LOG"
}

# Cron runs in a stripped environment with no ssh-agent. The repo uses an
# SSH remote, so we need a key path explicitly. ~/.ssh/id_ed25519 covers
# the common case; override with QUADRIS_SSH_KEY if you use a different key.
if [[ -z "${GIT_SSH_COMMAND:-}" ]]; then
  default_key="$HOME/.ssh/id_ed25519"
  key="${QUADRIS_SSH_KEY:-$default_key}"
  if [[ -r "$key" ]]; then
    export GIT_SSH_COMMAND="ssh -i $key -o BatchMode=yes -o StrictHostKeyChecking=accept-new"
  fi
fi

git fetch --quiet origin main 2>>"$LOG" || {
  log "git fetch failed — see above"
  exit 1
}

LOCAL=$(git rev-parse HEAD)
REMOTE=$(git rev-parse origin/main)

if [[ "$LOCAL" == "$REMOTE" ]]; then
  # Up to date — silent exit. Don't spam the log on every tick.
  exit 0
fi

# Only act when origin is strictly ahead. If we're ahead (unpushed local
# commits) or diverged, do nothing — pulling those cases would either be
# a no-op (and uselessly bounce the app) or destructive.
behind_count=$(git rev-list --count HEAD..origin/main)
if [[ "$behind_count" -eq 0 ]]; then
  # Local is ahead of, or has diverged from, origin. Silent no-op.
  exit 0
fi

log "origin/main has $behind_count new commit(s): $LOCAL → $REMOTE"

# Refuse to pull if the working tree has uncommitted changes that could
# block a fast-forward (the supervisor isn't running interactively, so
# we can't ask the user what to do).
if ! git diff --quiet || ! git diff --cached --quiet; then
  log "ABORT: working tree has uncommitted changes; refusing to pull"
  exit 1
fi

if ! git pull --ff-only --quiet origin main 2>>"$LOG"; then
  log "ABORT: fast-forward pull failed (diverged history?)"
  exit 1
fi

NEW_HEAD=$(git rev-parse HEAD)
log "pulled to $NEW_HEAD"

# Tell the supervisor (serve_public.sh) to respawn the app with new code.
# Match the venv'd python to avoid hitting unrelated Python processes.
target_pids=$(pgrep -f "$PROJECT_ROOT/.venv/bin/python.*$PROJECT_ROOT/gui/app.py" || true)

if [[ -z "$target_pids" ]]; then
  log "no running gui/app.py process to restart (pulled, but nothing to signal)"
  exit 0
fi

for pid in $target_pids; do
  if kill -TERM "$pid" 2>>"$LOG"; then
    log "sent SIGTERM to gui/app.py pid=$pid"
  else
    log "kill $pid failed"
  fi
done
