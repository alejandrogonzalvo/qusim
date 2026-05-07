# qusim deployment

How to run the DSE GUI for personal use, and how to put it behind a public
URL with auto-recovery and continuous deployment from `origin/main`.

The repo ships three deployment helpers under [`scripts/`](../scripts):

| Script | Role |
|---|---|
| [`serve_public.sh`](../scripts/serve_public.sh) | Foreground supervisor. Starts the Dash app and a Cloudflare named tunnel; respawns the app if it dies. |
| [`health_check.sh`](../scripts/health_check.sh) | Cron-driven watchdog. Probes the public URL and relaunches `serve_public.sh` after 3 consecutive failures. |
| [`pull_and_restart.sh`](../scripts/pull_and_restart.sh) | Cron-driven CD. Fast-forwards `origin/main` and SIGTERMs the running app; the supervisor respawns it with the new code. |

All three resolve the project root from their own location, so they work
from any working directory (cron, systemd, another shell).

## Mode A: local dev (single user)

For development on your laptop. No tunnel, no watchdog, no cron.

```bash
git clone <repo> qusim && cd qusim
python3.12 -m venv .venv && source .venv/bin/activate
pip install maturin
maturin develop --release          # builds the Rust extension into the venv
pip install -e ".[gui]"            # Dash + Cytoscape
qusim-dse                          # → http://127.0.0.1:8050
```

Override the bind address with `QUSIM_HOST=0.0.0.0 QUSIM_PORT=8080 qusim-dse`
if you want LAN access. See [`README.md`](../README.md) for the full
install matrix (`[dev]` extras, `cargo test`, etc.).

## Mode B: public deployment (small group, auto-recovery)

This is the supported production path: a Dash process bound to localhost,
fronted by a Cloudflare named tunnel, with a cron-driven watchdog and
auto-pull. Capacity target is ~3 concurrent users (the Flask dev server
runs `threaded=True`; gate-level sweeps are CPU-bound and serialize on
the Rust core).

### Architecture

```
┌─────────────────────────────────────────────────────┐
│ cron (every 5 min)                                  │
│   pull_and_restart.sh  ─── git fetch + SIGTERM app  │
│   health_check.sh      ─── probe URL, relaunch on   │
│                            3 consecutive failures   │
└─────────────────────────────────────────────────────┘
            │ signals
            ▼
┌──── serve_public.sh (foreground supervisor) ────────┐
│   ├─ qusim-dse (gui/app.py)  bound to 127.0.0.1     │
│   └─ cloudflared tunnel run <name>                  │
└─────────────────────────────────────────────────────┘
            │
            ▼
   https://<your-hostname>          (Cloudflare edge)
```

### One-time bootstrap on a fresh VM

The instructions below assume Ubuntu/Debian. Adjust package names for
other distros.

#### 1. System dependencies

```bash
sudo apt update
sudo apt install -y python3.12 python3.12-venv python3.12-dev \
                    build-essential pkg-config curl git
```

The Rust toolchain is fetched on demand by `maturin`; if `maturin
develop --release` fails with "rustc not found", install Rust:

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
. "$HOME/.cargo/env"
```

#### 2. Install cloudflared

Cloudflare publishes a Debian repo:

```bash
sudo mkdir -p --mode=0755 /usr/share/keyrings
curl -fsSL https://pkg.cloudflare.com/cloudflare-main.gpg \
  | sudo tee /usr/share/keyrings/cloudflare-main.gpg >/dev/null
echo 'deb [signed-by=/usr/share/keyrings/cloudflare-main.gpg] \
  https://pkg.cloudflare.com/cloudflared any main' \
  | sudo tee /etc/apt/sources.list.d/cloudflared.list
sudo apt update && sudo apt install -y cloudflared
```

If the repo is not available on your distro, grab the static `.deb` /
`.rpm` from <https://github.com/cloudflare/cloudflared/releases>.

#### 3. Clone and build

```bash
git clone <repo> ~/qusim && cd ~/qusim
python3.12 -m venv .venv
.venv/bin/pip install maturin
.venv/bin/maturin develop --release
.venv/bin/pip install -e ".[gui]"
```

The `serve_public.sh` script expects `./.venv/bin/python` to exist at
the project root. If you build elsewhere, symlink the venv in.

#### 4. Create a Cloudflare named tunnel

```bash
cloudflared tunnel login                          # opens a browser; pick a zone
cloudflared tunnel create <tunnel-name>           # writes ~/.cloudflared/<UUID>.json
cloudflared tunnel route dns <tunnel-name> <hostname>
```

`<hostname>` is whatever DNS record you want (e.g. `qusim.example.org`).
Note the UUID; you will pass it to `serve_public.sh` as `TUNNEL_UUID`.

The script does **not** depend on `~/.cloudflared/config.yml`. It
generates a self-contained config in `$LOG_DIR/cloudflared-config.yml`
on each run, using the env vars below.

#### 5. Smoke test in the foreground

```bash
TUNNEL_NAME=<tunnel-name> \
TUNNEL_UUID=<uuid> \
TUNNEL_HOSTNAME=<hostname> \
QUSIM_PORT=8060 \
./scripts/serve_public.sh
```

The script waits for the app port and a `Registered tunnel connection`
log line before printing the public URL. Hit it from a browser to
confirm. Logs land in `/tmp/qusim-serve/{app,cloudflared}.log` (override
with `LOG_DIR=...`).

`Ctrl+C` shuts both children down via the trap.

#### 6. Wire up cron

Once the foreground run is healthy, install the cron entries and let
the system babysit it. `crontab -e`:

```cron
# qusim deployment
*/5 * * * * QUSIM_URL=https://<hostname> QUSIM_PORT=8060 \
    TUNNEL_NAME=<tunnel-name> TUNNEL_UUID=<uuid> TUNNEL_HOSTNAME=<hostname> \
    /home/<user>/qusim/scripts/health_check.sh
*/5 * * * * /home/<user>/qusim/scripts/pull_and_restart.sh
@reboot     QUSIM_URL=https://<hostname> QUSIM_PORT=8060 \
    TUNNEL_NAME=<tunnel-name> TUNNEL_UUID=<uuid> TUNNEL_HOSTNAME=<hostname> \
    /home/<user>/qusim/scripts/health_check.sh
```

What each line does:

- `health_check.sh` probes `QUSIM_URL` every 5 min. After 3 failures it
  kills any stale supervisor and starts a fresh one. The `@reboot`
  entry boots the supervisor on startup; no separate systemd unit is
  required.
- `pull_and_restart.sh` checks for new commits on `origin/main`. If the
  remote is strictly ahead, it fast-forwards and SIGTERMs `gui/app.py`.
  The supervisor's restart loop respawns the app with the new code.
  Cloudflared keeps running, so the public URL stays up; the tunnel
  returns 502 for the few seconds the app is down.

The pull script needs an SSH key reachable from cron (the repo uses an
SSH remote and cron has no ssh-agent). Default is `~/.ssh/id_ed25519`;
override with `QUSIM_SSH_KEY=/path/to/key` in the crontab line if
needed. See [`pull_and_restart.sh`](../scripts/pull_and_restart.sh)
lines 35-41.

### Operations

#### Logs

All under `$LOG_DIR` (default `/tmp/qusim-serve/`):

| File | Contents |
|---|---|
| `app.log` | Dash stdout/stderr (every request, every callback exception) |
| `cloudflared.log` | Tunnel registration + per-request edge logs |
| `health.log` | Watchdog failures and recoveries (silent on success) |
| `pull.log` | New commits pulled, restart signals sent |
| `supervisor.out` | `nohup` output of supervisors started by the watchdog |

`/tmp` is volatile; if you want logs to survive reboots, set
`LOG_DIR=/var/log/qusim` (and create it with the right ownership) in
both the cron lines and `serve_public.sh` invocations.

#### Manual restart

```bash
pkill -f "serve_public.sh"     # supervisor + children exit via trap
# wait ~5 min for the @reboot/cron health_check to relaunch, OR:
TUNNEL_NAME=... TUNNEL_UUID=... TUNNEL_HOSTNAME=... \
  nohup ./scripts/serve_public.sh > /tmp/qusim-serve/supervisor.out 2>&1 &
disown
```

#### Manual deploy of a hotfix

The pull script only acts on commits already on `origin/main`. If you
need to ship something faster than the 5-minute cron tick:

```bash
cd ~/qusim
git fetch && git pull --ff-only
pkill -f ".venv/bin/python.*gui/app.py"   # supervisor respawns it
```

#### Changing port / hostname / tunnel

All three scripts read the same env vars (`TUNNEL_NAME`, `TUNNEL_UUID`,
`TUNNEL_HOSTNAME`, `QUSIM_HOST`, `QUSIM_PORT`, `QUSIM_URL` for
`health_check.sh`). Update both cron entries and any active supervisor
process. The defaults in `serve_public.sh` are the original author's
tunnel; rely on the env vars rather than editing the script.

### Migrating from one host to another

If you are moving the deployment off this laptop and onto a VM:

1. On the **VM**, follow the bootstrap above. Create a *new* Cloudflare
   tunnel. UUIDs are tied to the host that ran `cloudflared tunnel
   create`. Either reuse the same hostname (delete the old DNS route
   first) or pick a new one. The new operator can choose freely; nothing
   in the codebase hardcodes a public URL.
2. On the **old laptop**, stop and disable the deployment:
   ```bash
   crontab -e                      # remove the qusim lines
   pkill -f serve_public.sh
   pkill -f "cloudflared.*tunnel run"
   ```
3. (Optional) Delete the old tunnel from Cloudflare so it stops
   appearing in the dashboard:
   ```bash
   cloudflared tunnel delete <old-tunnel-name>
   ```

The repo itself is host-agnostic. No path, hostname, or UUID is baked
into the Python or Rust code; the deployment knobs all live in env vars
read by the three shell scripts.

## Capacity and known limits

- Single-process Flask dev server. `app.run(threaded=True)` gives
  per-request threads, but the Rust core holds the GIL during sweeps, so
  one heavy sweep blocks the others. Rule of thumb: 3 concurrent
  *active* sweepers; many more passive viewers are fine.
- No process manager beyond the supervisor's restart loop. If the host
  is rebooted, the `@reboot` cron line brings everything back; if cron
  itself is broken, nothing recovers automatically.
- No HTTPS termination on the origin. Cloudflare handles TLS; the app
  only listens on `127.0.0.1` so it is not directly reachable from the
  LAN. `QUSIM_HOST=0.0.0.0` opens that up if you need to bypass the
  tunnel for debugging.
- Sweep results live in process memory. A restart drops everything in
  the current GUI session; users can save/load via the `Sessions` panel
  ([`gui/session.py`](../gui/session.py)).
