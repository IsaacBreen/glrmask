#!/usr/bin/env bash
set -euo pipefail

SERVER_NAME="${GLRMASK_BENCH_SERVER:-glrmask-bench}"
REMOTE_ROOT="${GLRMASK_BENCH_REMOTE_ROOT:-/root/bench}"
REMOTE_REPO="${GLRMASK_BENCH_REMOTE_REPO:-${REMOTE_ROOT}/glrmask}"
REMOTE_BRANCH="${GLRMASK_BENCH_REMOTE_BRANCH:-worker2-remote}"
SESSION_NAME="${GLRMASK_BENCH_SESSION:-glrmask-bench}"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
LOCAL_REPO="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

if [[ -x /opt/homebrew/bin/hcloud ]]; then
    HCLOUD_BIN="${HCLOUD_BIN:-/opt/homebrew/bin/hcloud}"
else
    HCLOUD_BIN="${HCLOUD_BIN:-hcloud}"
fi

SSH_OPTS=(-o StrictHostKeyChecking=no)
if [[ -n "${GLRMASK_BENCH_KNOWN_HOSTS:-}" ]]; then
    SSH_OPTS+=(-o UserKnownHostsFile="${GLRMASK_BENCH_KNOWN_HOSTS}")
fi

die() {
    echo "error: $*" >&2
    exit 1
}

require_cmd() {
    command -v "$1" >/dev/null 2>&1 || die "missing required command: $1"
}

require_token() {
    [[ -n "${HETZNER_API_KEY:-}" ]] || die "HETZNER_API_KEY must be set"
    export HCLOUD_TOKEN="${HETZNER_API_KEY}"
}

server_ip() {
    require_cmd "${HCLOUD_BIN}"
    require_token
    "${HCLOUD_BIN}" server ip "${SERVER_NAME}"
}

remote_ssh() {
    local ip
    ip="$(server_ip)"
    ssh "${SSH_OPTS[@]}" root@"${ip}" "$@"
}

remote_bash() {
    local ip
    ip="$(server_ip)"
    ssh "${SSH_OPTS[@]}" root@"${ip}" 'bash -s' -- "$@"
}

sync_repo() {
    local ip
    ip="$(server_ip)"
    rsync -az --delete \
        --exclude 'target/' \
        --exclude '__pycache__/' \
        --exclude '.pytest_cache/' \
        --exclude '.mypy_cache/' \
        --exclude '.ruff_cache/' \
        --exclude '.venv/' \
        --exclude '.git/' \
        "${LOCAL_REPO}/" root@"${ip}":"${REMOTE_REPO}/"

    remote_bash "${REMOTE_REPO}" "${REMOTE_BRANCH}" <<'REMOTE'
set -euo pipefail
repo="$1"
branch="$2"
git config --global --add safe.directory "${repo}" || true
if git -C "${repo}" show-ref --verify --quiet "refs/heads/${branch}"; then
    git -C "${repo}" checkout "${branch}" >/dev/null
else
    git -C "${repo}" checkout -b "${branch}" >/dev/null
fi
REMOTE
}

build_remote() {
    remote_bash "${REMOTE_ROOT}" "${REMOTE_REPO}" <<'REMOTE'
set -euo pipefail
remote_root="$1"
repo="$2"
source /root/.cargo/env
source "${remote_root}/.venv/bin/activate"
cd "${repo}/python"
maturin develop --release
cd "${repo}"
cargo build --release -q
REMOTE
}

shell_remote() {
    remote_ssh -t bash -lc "source '${REMOTE_ROOT}/env.sh' && exec bash -i"
}

tmux_remote() {
    remote_bash "${SESSION_NAME}" "${REMOTE_ROOT}" <<'REMOTE'
set -euo pipefail
session="$1"
remote_root="$2"
if ! tmux has-session -t "${session}" 2>/dev/null; then
    tmux new-session -d -s "${session}" "bash -lc 'source ${remote_root}/env.sh; exec bash -i'"
fi
exec tmux attach -t "${session}"
REMOTE
}

exec_remote() {
    [[ "$#" -gt 0 ]] || die "exec requires a command"
    remote_bash "${REMOTE_ROOT}" "${REMOTE_REPO}" "$@" <<'REMOTE'
set -euo pipefail
remote_root="$1"
repo="$2"
shift 2
source /root/.cargo/env
source "${remote_root}/.venv/bin/activate"
cd "${repo}"
exec "$@"
REMOTE
}

fetch_remote() {
    [[ "$#" -ge 1 ]] || die "fetch requires a remote path"
    local ip
    local remote_path="$1"
    local local_path="${2:-.}"
    ip="$(server_ip)"
    rsync -az root@"${ip}":"${remote_path}" "${local_path}"
}

status_remote() {
    require_cmd "${HCLOUD_BIN}"
    require_token
    "${HCLOUD_BIN}" server list
    echo
    remote_bash "${REMOTE_ROOT}" "${REMOTE_REPO}" <<'REMOTE'
set -euo pipefail
remote_root="$1"
repo="$2"
git config --global --add safe.directory "${repo}" || true
printf 'remote_branch='
git -C "${repo}" branch --show-current
printf '\nremote_head='
git -C "${repo}" rev-parse --short HEAD
printf '\npython='
source "${remote_root}/.venv/bin/activate"
python --version
REMOTE
}

destroy_remote() {
    require_cmd "${HCLOUD_BIN}"
    require_token
    "${HCLOUD_BIN}" server delete "${SERVER_NAME}"
}

usage() {
    cat <<'USAGE'
Usage: scripts/hetzner_bench.sh <command> [args...]

Commands:
  ip                 Print the current server IP
  status             Show Hetzner server state and remote branch info
  sync               Rsync local glrmask/ to /root/bench/glrmask on the server
  build              Build the remote Python extension and Rust release binary
  shell              Open an interactive remote shell with cargo + venv loaded
  tmux               Attach/create the remote tmux session
  exec <cmd...>      Run a one-off command inside /root/bench/glrmask
  fetch <src> [dst]  Copy a remote file or directory back locally
  destroy            Destroy the Hetzner server
USAGE
}

main() {
    local command="${1:-}"
    case "${command}" in
        ip)
            server_ip
            ;;
        status)
            status_remote
            ;;
        sync)
            sync_repo
            ;;
        build)
            build_remote
            ;;
        shell)
            shell_remote
            ;;
        tmux)
            tmux_remote
            ;;
        exec)
            shift || true
            exec_remote "$@"
            ;;
        fetch)
            shift || true
            fetch_remote "$@"
            ;;
        destroy)
            destroy_remote
            ;;
        ""|-h|--help|help)
            usage
            ;;
        *)
            die "unknown command: ${command}"
            ;;
    esac
}

main "$@"