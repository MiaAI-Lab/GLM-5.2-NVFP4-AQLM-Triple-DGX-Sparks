#!/usr/bin/env bash
# stop.sh — tear down GLM-5.2 NVFP4+AQLM serve + Ray on all 3 Sparks
#
# Stops everything this stack started:
#   - vLLM API / EngineCore inside the Ray head container
#   - Ray head + worker docker containers (glm52-aqlm-*)
#   - local log-tail / pid files
#   - leftover probe helpers tied to this stack
#
# Does NOT delete images (GHCR/local), weights, or JIT caches.
# Optional: UNMOUNT=1 also unmounts SSHFS weight mounts on workers
# (only needed if you used ./start.sh mount instead of sync).
#
# Usage:
#   ./stop.sh
#   UNMOUNT=1 ./stop.sh
#   ./start.sh stop          # same (delegates here)
#
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

if [[ -f "$ROOT/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "$ROOT/.env"
  set +a
fi

_abs() { readlink -f "$1" 2>/dev/null || echo "$1"; }

HEAD_IP="${HEAD_IP:-10.0.0.1}"
WORKER1_IP="${WORKER1_IP:-${WORKER_IP:-10.0.0.2}}"
WORKER2_IP="${WORKER2_IP:-10.0.0.3}"
WORKER_HOSTS=("${WORKER1_HOST:-$WORKER1_IP}" "${WORKER2_HOST:-$WORKER2_IP}")
WORKER_USER="${WORKER_USER:-$USER}"
WEIGHT_LINK_ROOT="${WEIGHT_LINK_ROOT:-}"
SSH_IDENTITY="${SSH_IDENTITY:-$HOME/.ssh/id_ed25519_shared}"
SSH_IDENTITY="$(_abs "$SSH_IDENTITY")"
REMOTE_PY="${REMOTE_PY:-$ROOT/scripts/remote.py}"
WEIGHT_NAME="${WEIGHT_NAME:-GLM-5.2-NVFP4-AQLM-hybrid}"

HEAD_CTN="${HEAD_CTN:-glm52-aqlm-head}"
WORKER_CTN="${WORKER_CTN:-glm52-aqlm-worker}"
SERVE_CTN="${SERVE_CTN:-glm52-aqlm-serve}"
PORT="${PORT:-8888}"

GREEN=$'\033[0;32m'; YELLOW=$'\033[1;33m'; RED=$'\033[0;31m'; NC=$'\033[0m'
info() { echo "${GREEN}[+]${NC} $*"; }
warn() { echo "${YELLOW}[!]${NC} $*"; }
err()  { echo "${RED}[x]${NC} $*" >&2; }

[[ -f "$REMOTE_PY" ]] || { err "missing $REMOTE_PY"; exit 1; }

remote_on() {
  local host="$1"; shift
  local to_args=()
  if [[ "${1:-}" == "--timeout" ]]; then
    to_args=(--timeout "$2")
    shift 2
  elif [[ -n "${REMOTE_TIMEOUT:-}" ]]; then
    to_args=(--timeout "$REMOTE_TIMEOUT")
  fi
  local env_args=()
  [[ -f "$ROOT/.env" ]] && env_args=(--env-file "$ROOT/.env")
  python3 "$REMOTE_PY" "${env_args[@]}" \
    --host "$host" --user "$WORKER_USER" \
    --identity "$SSH_IDENTITY" \
    "${to_args[@]}" \
    "bash -lc $(printf '%q' "$*")"
}

# Kill vLLM / Ray leftovers inside a running container (best-effort).
_stop_inside_ctn() {
  local ctn="$1"
  docker exec "$ctn" bash -lc '
    pkill -f "[v]llm serve" >/dev/null 2>&1 || true
    pkill -f "[E]ngineCore" >/dev/null 2>&1 || true
    pkill -f "nccl_probe_worker" >/dev/null 2>&1 || true
    if command -v ray >/dev/null 2>&1; then
      ray stop --force >/dev/null 2>&1 || true
    fi
  ' 2>/dev/null || true
}

_stop_inside_ctn_remote() {
  local host="$1" ctn="$2"
  remote_on "$host" "
    if docker ps --format '{{.Names}}' | grep -qx $(printf '%q' "$ctn"); then
      docker exec $(printf '%q' "$ctn") bash -lc '
        pkill -f \"[v]llm serve\" >/dev/null 2>&1 || true
        pkill -f \"[E]ngineCore\" >/dev/null 2>&1 || true
        pkill -f nccl_probe_worker >/dev/null 2>&1 || true
        command -v ray >/dev/null 2>&1 && ray stop --force >/dev/null 2>&1 || true
      ' 2>/dev/null || true
    fi
  " || true
}

# Remove every container whose name matches glm52-aqlm*
_rm_glm52_ctns_local() {
  local ids
  ids=$(docker ps -aq --filter "name=glm52-aqlm" 2>/dev/null || true)
  if [[ -n "${ids}" ]]; then
    # shellcheck disable=SC2086
    docker rm -f $ids >/dev/null 2>&1 || true
  fi
  docker rm -f "$HEAD_CTN" "$WORKER_CTN" "$SERVE_CTN" >/dev/null 2>&1 || true
}

_rm_glm52_ctns_remote() {
  local host="$1"
  remote_on "$host" "
    ids=\$(docker ps -aq --filter name=glm52-aqlm 2>/dev/null || true)
    if [[ -n \"\$ids\" ]]; then
      docker rm -f \$ids >/dev/null 2>&1 || true
    fi
    docker rm -f ${WORKER_CTN} ${SERVE_CTN} ${HEAD_CTN} >/dev/null 2>&1 || true
    pkill -f nccl_probe_worker >/dev/null 2>&1 || true
  " || warn "SSH cleanup failed on $host (node unreachable?)"
}

info "=== stop GLM-5.2 NVFP4+AQLM stack on head + workers ==="

# 1) Local log-tail / pid helpers
if [[ -f "$ROOT/logs/glm52-aqlm-logtail.pid" ]]; then
  kill "$(cat "$ROOT/logs/glm52-aqlm-logtail.pid")" 2>/dev/null || true
  rm -f "$ROOT/logs/glm52-aqlm-logtail.pid"
fi
if [[ -f "$ROOT/logs/glm52-aqlm.pid" ]]; then
  kill "$(cat "$ROOT/logs/glm52-aqlm.pid")" 2>/dev/null || true
  rm -f "$ROOT/logs/glm52-aqlm.pid"
fi
pkill -f "tail -n \+1 -F .*/logs/glm52-aqlm.log" >/dev/null 2>&1 || true
pkill -f "tail -n \+1 -F /tmp/vllm-serve.log" >/dev/null 2>&1 || true

# 2) Soft-stop processes inside containers, then remove containers
info "head ($HEAD_IP): stop vLLM/Ray inside $HEAD_CTN"
if docker ps --format '{{.Names}}' 2>/dev/null | grep -qx "$HEAD_CTN"; then
  _stop_inside_ctn "$HEAD_CTN"
fi
if docker ps --format '{{.Names}}' 2>/dev/null | grep -qx "$SERVE_CTN"; then
  _stop_inside_ctn "$SERVE_CTN"
fi
info "head: remove glm52-aqlm* containers"
_rm_glm52_ctns_local

# 3) Workers
for h in "${WORKER_HOSTS[@]}"; do
  info "worker $WORKER_USER@$h: stop + remove glm52-aqlm*"
  _stop_inside_ctn_remote "$h" "$WORKER_CTN"
  _rm_glm52_ctns_remote "$h"
done

# 4) Optional SSHFS unmount (./start.sh mount only)
if [[ "${UNMOUNT:-0}" == "1" ]]; then
  for h in "${WORKER_HOSTS[@]}"; do
    info "worker $h: unmount weights"
    if [[ -n "$WEIGHT_LINK_ROOT" ]]; then
      remote_on "$h" "fusermount -u $(printf '%q' "$WEIGHT_LINK_ROOT")/${WEIGHT_NAME} 2>/dev/null || umount $(printf '%q' "$WEIGHT_LINK_ROOT")/${WEIGHT_NAME} 2>/dev/null || true" \
        || true
    else
      remote_on "$h" "fusermount -u \"\$HOME/models/hf/${WEIGHT_NAME}\" 2>/dev/null || umount \"\$HOME/models/hf/${WEIGHT_NAME}\" 2>/dev/null || true" \
        || true
    fi
  done
fi

# 5) Verify
echo
info "verify"
if curl -sf --max-time 2 "http://127.0.0.1:${PORT}/v1/models" >/dev/null 2>&1; then
  warn "API still responding on :${PORT}"
else
  info "API down on :${PORT}"
fi

local_left=$(docker ps --format '{{.Names}}' 2>/dev/null | grep glm52-aqlm || true)
if [[ -n "$local_left" ]]; then
  warn "head still has: $local_left"
else
  info "head: no glm52-aqlm containers"
fi

for h in "${WORKER_HOSTS[@]}"; do
  left=$(remote_on "$h" "docker ps --format '{{.Names}}' | grep glm52-aqlm || true" 2>/dev/null || echo "SSH_FAIL")
  if [[ -z "$left" ]]; then
    info "$h: no glm52-aqlm containers"
  elif [[ "$left" == "SSH_FAIL" ]]; then
    warn "$h: SSH verify failed"
  else
    warn "$h: still has: $left"
  fi
done

info "stopped (images + weights kept)"
