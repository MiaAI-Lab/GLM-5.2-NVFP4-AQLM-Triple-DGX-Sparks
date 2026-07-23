#!/usr/bin/env bash
# start.sh — GLM-5.2 NVFP4+AQLM hybrid on 3× DGX Spark
#
# Checkpoint: jarrelscy/GLM-5.2-NVFP4-AQLM-hybrid (~292 GB)
#   https://huggingface.co/jarrelscy/GLM-5.2-NVFP4-AQLM-hybrid
# Serving stack (NOT stock vLLM — hybrid NVFP4+AQLM MoE loader/kernels):
#   git clone -b glm52-sm120 https://github.com/jarrelscy/vllm-glm52-sm120
#   docker build -f Dockerfile.glm52-sm120 -t glm52-sm120 .
#
# Model-card recipe is single-node TP4+DCP4+MTP on 4× RTX PRO 6000 (sm120 amd64).
# This script remaps to 3 Sparks (TP3+DCP3, RoCE NCCL) using the bare-launch
# flags from the card — bypassing the image PARALLEL= entrypoint (Ray multi-node).
#
# HARD STOPS (see ./start.sh doctor):
#   - Official Dockerfile is CUDA x86 / sm120. Sparks are aarch64 / sm121.
#   - No published arm64 image. ./start.sh build on Spark will not yield a
#     drop-in serving image without a port (TORCH_CUDA_ARCH_LIST + kernels).
#   - TP3: num_attention_heads=64, n_routed_experts=256, index_n_heads=32
#     are not divisible by 3.
#
# Cluster (typical): same OS user + SSH key on all 3 Sparks.
#   head    10.0.0.1  (run start.sh here)
#   worker1 10.0.0.2
#   worker2 10.0.0.3
# Asymmetric users: set WORKER_USER / optional WEIGHT_LINK_ROOT in .env.
#
# Usage:
#   ./start.sh                 # doctor → build/pull → download → sync → ray → serve
#   ./start.sh serve           # same (default)
#   ./start.sh doctor          # connectivity + prereq report
#   ./start.sh build           # clone fork + docker build -f Dockerfile.glm52-sm120
#   ./start.sh pull            # distribute image head → workers (docker save/rsync/load)
#   ./start.sh download        # hf download checkpoint onto head
#   ./start.sh sync            # rsync weights head → both workers (local NVMe)
#   ./start.sh sync_dspark     # rsync DSpark draft → workers
#   ./start.sh mount           # SSHFS fallback if sync skipped
#   ./start.sh ray             # Ray head + 2 workers (docker; mounts draft if ENABLE_DSPARK)
#   ./start.sh stop            # tear down serve + ray containers
#   ./start.sh status          # what's up
#   ./start.sh smoke           # coherence probe against :8000
#
# Config via .env (same directory) or environment.
#
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

# ─── load .env ────────────────────────────────────────────────────────────────
if [[ -f "$ROOT/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "$ROOT/.env"
  set +a
fi

# ─── cluster / paths ──────────────────────────────────────────────────────────
HEAD_IP="${HEAD_IP:-10.0.0.1}"
WORKER1_IP="${WORKER1_IP:-${WORKER_IP:-10.0.0.2}}"
WORKER2_IP="${WORKER2_IP:-10.0.0.3}"
WORKER_HOSTS=("${WORKER1_HOST:-$WORKER1_IP}" "${WORKER2_HOST:-$WORKER2_IP}")
WORKER_IPS=("$WORKER1_IP" "$WORKER2_IP")
# Same username on every Spark is the common case; override for mixed accounts.
WORKER_USER="${WORKER_USER:-$USER}"
# Optional absolute weight-link root on workers (default: $HOME/models/hf on the remote).
# Example asymmetric: WEIGHT_LINK_ROOT=/home/mia/models/hf
WEIGHT_LINK_ROOT="${WEIGHT_LINK_ROOT:-}"
FABRIC_IFACE="${FABRIC_IFACE:-enp1s0f1np1}"
# Gloo needs a full L2/L3 mesh. CX7 enp1s0f* is pairwise-only on this cluster;
# lo advertises 127.0.0.1. Use the LAN NIC (enP7s7 / 192.168.1.0/24) for Gloo/NCCL
# TCP bootstrap. Prefer NCCL_NET=IB / RoCE when healthy; Socket over LAN is the
# bring-up fallback (set NCCL_IB_DISABLE=1 NCCL_NET=Socket in .env).
GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-enP7s7}"
NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-enP7s7}"
IB_HCA="${IB_HCA:-rocep1s0f0,rocep1s0f1}"
NCCL_NET="${NCCL_NET:-IB}"
NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-0}"
# Pairwise CX7 triangle: keep ports unmerged and match QPs by RoCE subnet GID.
NCCL_IB_MERGE_NICS="${NCCL_IB_MERGE_NICS:-0}"
NCCL_IB_SUBNET_AWARE_ROUTING="${NCCL_IB_SUBNET_AWARE_ROUTING:-1}"
NCCL_CROSS_NIC="${NCCL_CROSS_NIC:-1}"
NCCL_GIN_ENABLE="${NCCL_GIN_ENABLE:-0}"
NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-0}"
NCCL_SHM_DISABLE="${NCCL_SHM_DISABLE:-0}"
NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
# Sapid RUNBOOK: Torch bundles NCCL 2.28.9; pin ≥2.30.7 for FULL/FAP cudagraph + TP on GB10+CX7.
NCCL_HOST_DIR="${NCCL_HOST_DIR:-$HOME/nccl-2.30.7}"
NCCL_CONTAINER_DIR="${NCCL_CONTAINER_DIR:-/nccl}"
# Prefer the versioned .so (host libnccl.so.2 often absolute-symlinks outside the mount).
VLLM_NCCL_SO_PATH="${VLLM_NCCL_SO_PATH:-$NCCL_CONTAINER_DIR/libnccl.so.2.30.7}"
NCCL_GRAPH_MIXING_SUPPORT="${NCCL_GRAPH_MIXING_SUPPORT:-1}"
SSH_IDENTITY="${SSH_IDENTITY:-$HOME/.ssh/id_ed25519_shared}"

HF_REPO="${HF_REPO:-jarrelscy/GLM-5.2-NVFP4-AQLM-hybrid}"
MODEL_DIR="${MODEL_DIR:-$HOME/models/hf/GLM-5.2-NVFP4-AQLM-hybrid}"
COMMON_MODEL="${COMMON_MODEL:-/var/tmp/glm52-aqlm}"
WORKER_MODEL_LOCAL="${WORKER_MODEL_LOCAL:-models/hf/GLM-5.2-NVFP4-AQLM-hybrid}"
WEIGHT_NAME="GLM-5.2-NVFP4-AQLM-hybrid"
EXPECTED_SHARDS="${EXPECTED_SHARDS:-77}"

# DSpark draft (RedHatAI/GLM-5.2-speculator.dspark)
DSPARK_MODEL_DIR="${DSPARK_MODEL_DIR:-$HOME/models/hf/GLM-5.2-speculator.dspark}"
COMMON_DSPARK="${COMMON_DSPARK:-/var/tmp/glm52-dspark}"
WORKER_DSPARK_LOCAL="${WORKER_DSPARK_LOCAL:-models/hf/GLM-5.2-speculator.dspark}"
DSPARK_CONTAINER_PATH="${DSPARK_CONTAINER_PATH:-/models/dspark}"
DSPARK_SPEC_MODEL="${DSPARK_SPEC_MODEL:-$DSPARK_CONTAINER_PATH}"
DSPARK_SPEC_TOKENS="${DSPARK_SPEC_TOKENS:-7}"

IMAGE="${IMAGE:-glm52-aqlm-sm121}"
DOCKERFILE="${DOCKERFILE:-Dockerfile.glm52-sm121}"
HEAD_CTN="${HEAD_CTN:-glm52-aqlm-head}"
WORKER_CTN="${WORKER_CTN:-glm52-aqlm-worker}"
SERVE_CTN="${SERVE_CTN:-glm52-aqlm-serve}"

TP_SIZE="${TP_SIZE:-3}"
DCP_SIZE="${DCP_SIZE:-1}"
PP_SIZE="${PP_SIZE:-1}"
DCP_COMM_BACKEND="${DCP_COMM_BACKEND:-ag_rs}"
ENABLE_MTP="${ENABLE_MTP:-0}"
ENABLE_DSPARK="${ENABLE_DSPARK:-0}"

PORT="${PORT:-8000}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.88}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-2}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-4096}"
MTP_SPEC_TOKENS="${MTP_SPEC_TOKENS:-3}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-glm-5.2}"
KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-fp8_ds_mla}"

RAY_PORT="${RAY_PORT:-6379}"
OBJECT_STORE="${OBJECT_STORE:-134217728}"
OBJECT_SPILLING_DIR="${OBJECT_SPILLING_DIR:-/var/tmp/ray-spill}"
NCCL_MAX_NCHANNELS="${NCCL_MAX_NCHANNELS:-4}"
NCCL_MIN_NCHANNELS="${NCCL_MIN_NCHANNELS:-4}"
NCCL_BUFFSIZE="${NCCL_BUFFSIZE:-1048576}"

VLLM_FORK_URL="${VLLM_FORK_URL:-https://github.com/jarrelscy/vllm-glm52-sm120}"
VLLM_FORK_BRANCH="${VLLM_FORK_BRANCH:-glm52-sm120}"
VLLM_FORK_DIR="${VLLM_FORK_DIR:-$ROOT/dev/vendor/vllm-glm52-sm120}"

SERVE_LOG="${SERVE_LOG:-$ROOT/logs/glm52-aqlm.log}"
PID_FILE="${PID_FILE:-$ROOT/logs/glm52-aqlm.pid}"
REMOTE_PY="$ROOT/scripts/remote.py"

_abs() { readlink -f "$1" 2>/dev/null || echo "$1"; }
MODEL_DIR="$(_abs "$MODEL_DIR")"
DSPARK_MODEL_DIR="$(_abs "$DSPARK_MODEL_DIR")"
SSH_IDENTITY="$(_abs "$SSH_IDENTITY")"

# DSpark replaces MTP when enabled (do not stack).
if [[ "${ENABLE_DSPARK:-0}" == "1" ]]; then
  ENABLE_MTP=0
fi

RED=$'\033[0;31m'; GREEN=$'\033[0;32m'; YELLOW=$'\033[1;33m'; NC=$'\033[0m'
info() { echo "${GREEN}[+]${NC} $*"; }
warn() { echo "${YELLOW}[!]${NC} $*"; }
err()  { echo "${RED}[x]${NC} $*" >&2; }
die()  { err "$@"; exit 1; }

mkdir -p "$ROOT/logs"

# ─── SSH helpers (multi-worker) ───────────────────────────────────────────────
remote_on() {
  local host="$1"; shift
  local to_args=()
  if [[ "${1:-}" == "--timeout" ]]; then
    to_args=(--timeout "$2")
    shift 2
  elif [[ -n "${REMOTE_TIMEOUT:-}" ]]; then
    to_args=(--timeout "$REMOTE_TIMEOUT")
  fi
  python3 "$REMOTE_PY" --env-file "$ROOT/.env" \
    --host "$host" --user "$WORKER_USER" \
    --identity "$SSH_IDENTITY" \
    "${to_args[@]}" \
    "bash -lc $(printf '%q' "$*")"
}

remote_ok_on() { remote_on "$@" >/dev/null 2>&1; }

remote_all() {
  local host
  for host in "${WORKER_HOSTS[@]}"; do
    info "→ $WORKER_USER@$host: $*"
    remote_on "$host" "$@"
  done
}

ssh_rsync_e() {
  printf 'ssh -i %q -o IdentitiesOnly=yes -o StrictHostKeyChecking=accept-new' \
    "$SSH_IDENTITY"
}

ensure_ssh_key_on() {
  local host="$1"
  local pub
  pub=$(cat "${SSH_IDENTITY}.pub" 2>/dev/null || true)
  [[ -n "$pub" ]] || die "Missing ${SSH_IDENTITY}.pub"
  remote_on "$host" "mkdir -p ~/.ssh && chmod 700 ~/.ssh
    touch ~/.ssh/authorized_keys && chmod 600 ~/.ssh/authorized_keys
    grep -qxF '$pub' ~/.ssh/authorized_keys || echo '$pub' >> ~/.ssh/authorized_keys"
  if ! remote_ok_on "$host" "test -f ~/.ssh/id_ed25519_shared"; then
    info "Copying shared private key → $host (for SSHFS back to head)..."
    scp -o BatchMode=yes -o IdentitiesOnly=yes -i "$SSH_IDENTITY" \
      "$SSH_IDENTITY" \
      "${WORKER_USER}@${host}:.ssh/id_ed25519_shared" \
      || die "scp key to $host failed"
    remote_on "$host" "chmod 600 ~/.ssh/id_ed25519_shared"
  fi
}

ensure_ssh_keys() {
  local h
  for h in "${WORKER_HOSTS[@]}"; do
    ensure_ssh_key_on "$h"
  done
}

gid_index_local() {
  local ip="$1"
  local hex hca g i
  hex=$(printf '%02x%02x:%02x%02x' $(echo "$ip" | tr . ' '))
  local IFS=','
  for hca in $IB_HCA; do
    hca="${hca// /}"
    [[ -d /sys/class/infiniband/"$hca" ]] || continue
    for g in /sys/class/infiniband/"$hca"/ports/1/gids/*; do
      i=${g##*/}
      [[ "$(cat /sys/class/infiniband/"$hca"/ports/1/gid_attrs/types/"$i" 2>/dev/null)" == "RoCE v2" ]] || continue
      case "$(cat "$g")" in *ffff:"$hex") echo "$i"; return 0;; esac
    done
  done
  return 1
}

gid_index_remote() {
  local host="$1" ip="$2"
  local hex
  hex=$(printf '%02x%02x:%02x%02x' $(echo "$ip" | tr . ' '))
  remote_on "$host" "for hca in \$(echo $IB_HCA | tr ',' ' '); do
    [ -d /sys/class/infiniband/\$hca ] || continue;
    for g in /sys/class/infiniband/\$hca/ports/1/gids/*; do
      i=\${g##*/};
      [ \"\$(cat /sys/class/infiniband/\$hca/ports/1/gid_attrs/types/\$i 2>/dev/null)\" = 'RoCE v2' ] || continue;
      case \$(cat \$g) in *ffff:$hex) echo \$i; exit 0;; esac;
    done;
  done"
}

# Resolve to a real directory for docker -v (symlink sources make dockerd
# "mkdir ... file exists" on this Ubuntu/containerd).
model_mount_src() {
  local p="${1:-$COMMON_MODEL}"
  if [[ -L "$p" ]]; then
    readlink -f "$p"
  elif [[ -d "$p" ]]; then
    echo "$p"
  else
    echo "$MODEL_DIR"
  fi
}

dspark_mount_src() {
  local p="${1:-$COMMON_DSPARK}"
  if [[ -L "$p" ]]; then
    readlink -f "$p"
  elif [[ -d "$p" ]]; then
    echo "$p"
  else
    echo "$DSPARK_MODEL_DIR"
  fi
}

spec_enabled() {
  [[ "${ENABLE_DSPARK:-0}" == "1" || "${ENABLE_MTP:-0}" == "1" ]]
}

spec_label() {
  if [[ "${ENABLE_DSPARK:-0}" == "1" ]]; then
    echo "DSpark-$DSPARK_SPEC_TOKENS"
  elif [[ "${ENABLE_MTP:-0}" == "1" ]]; then
    echo "MTP-$MTP_SPEC_TOKENS"
  else
    echo "no-spec"
  fi
}

# Activate the fork venv then run a command (image ENTRYPOINT is PARALLEL= modes;
# Ray multi-node needs bare vllm serve).
vllm_bash() {
  printf 'cd /opt/vllm && source .venv/bin/activate && %s' "$1"
}

# Append docker -e flags for a Ray/serve container (multi-node RoCE).
# AQLM stack knobs from model-card tp4-1m-mtp; NCCL IB for Sparks (not NCCL_P2P_LEVEL=SYS —
# that is for single-box PCIe 4-GPU; multi-node RoCE wants NCCL_NET=IB).
docker_env_args() {
  local -n _dea=$1
  local hip="$2"
  local gid="${3:-}"
  _dea+=(
    -e "VLLM_HOST_IP=$hip"
    -e "HOST_IP=$hip"
    -e "RAY_memory_usage_threshold=0.99"
    -e "RAY_memory_monitor_refresh_ms=0"
    -e "CUDA_DEVICE_ORDER=PCI_BUS_ID"
    -e "CUDA_DEVICE_MAX_CONNECTIONS=32"
    -e "CUTE_DSL_ARCH=sm_121a"
    -e "TORCH_CUDA_ARCH_LIST=12.1a"
    -e "OMP_NUM_THREADS=16"
    -e "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
    -e "SAFETENSORS_FAST_GPU=1"
    -e "NCCL_NET=$NCCL_NET"
    -e "NCCL_IB_DISABLE=$NCCL_IB_DISABLE"
    -e "NCCL_IB_HCA=$IB_HCA"
    -e "NCCL_SOCKET_IFNAME=$NCCL_SOCKET_IFNAME"
    -e "GLOO_SOCKET_IFNAME=$GLOO_SOCKET_IFNAME"
    -e "NCCL_P2P_DISABLE=$NCCL_P2P_DISABLE"
    -e "NCCL_SHM_DISABLE=$NCCL_SHM_DISABLE"
    -e "NCCL_MAX_NCHANNELS=$NCCL_MAX_NCHANNELS"
    -e "NCCL_MIN_NCHANNELS=$NCCL_MIN_NCHANNELS"
    -e "NCCL_BUFFSIZE=$NCCL_BUFFSIZE"
    -e "NCCL_PROTO=LL,LL128,Simple"
    -e "NCCL_CROSS_NIC=${NCCL_CROSS_NIC:-1}"
    -e "NCCL_IB_MERGE_NICS=${NCCL_IB_MERGE_NICS:-0}"
    -e "NCCL_IB_SUBNET_AWARE_ROUTING=${NCCL_IB_SUBNET_AWARE_ROUTING:-1}"
    -e "NCCL_GIN_ENABLE=${NCCL_GIN_ENABLE:-0}"
    -e "NCCL_CUMEM_ENABLE=0"
    -e "NCCL_IGNORE_CPU_AFFINITY=1"
    -e "NCCL_DEBUG=${NCCL_DEBUG:-WARN}"
    -e "NCCL_GRAPH_MIXING_SUPPORT=$NCCL_GRAPH_MIXING_SUPPORT"
    -e "VLLM_NCCL_SO_PATH=$VLLM_NCCL_SO_PATH"
    -e "FLASHINFER_DISABLE_VERSION_CHECK=1"
    -e "VLLM_SPARSE_INDEXER_MAX_LOGITS_MB=${VLLM_SPARSE_INDEXER_MAX_LOGITS_MB:-192}"
    -e "VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=0"
    -e "VLLM_DISABLE_FP8_W8A16=${VLLM_DISABLE_FP8_W8A16:-1}"
    -e "VLLM_MTP_INDEX_SHARE=${VLLM_MTP_INDEX_SHARE:-1}"
    -e "GLM_MOE_LANE_ROWS=${GLM_MOE_LANE_ROWS:-1}"
    -e "GLM_NVFP4_LUT256=${GLM_NVFP4_LUT256:-1}"
    -e "GLM_SHARED_EXPERTS_DEBUG=${GLM_SHARED_EXPERTS_DEBUG:-1}"
    -e "VLLM_GLM_TP_PAD=${VLLM_GLM_TP_PAD:-1}"
    -e "VLLM_MARLIN_USE_ATOMIC_ADD=${VLLM_MARLIN_USE_ATOMIC_ADD:-0}"
    -e "VLLM_WORKER_MULTIPROC_METHOD=spawn"
    -e "VLLM_ALLOW_LONG_MAX_MODEL_LEN=1"
    -e "VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=${VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS:-1800}"
    -e "MODEL_DIR=/models/1m"
  )
  if [[ -n "$gid" ]]; then
    _dea+=(-e "NCCL_IB_GID_INDEX=$gid")
  elif [[ -n "${NCCL_IB_GID_INDEX:-}" ]]; then
    _dea+=(-e "NCCL_IB_GID_INDEX=$NCCL_IB_GID_INDEX")
  fi
}

# Common docker run flags. IB device + IPC_LOCK required on Spark for RoCE.
docker_common_args() {
  local -n _dca=$1
  local src
  src=$(model_mount_src)
  [[ -d "$src" ]] || die "model mount src missing: $src"
  _dca+=(
    --network host
    --ipc host
    --privileged
    --cap-add IPC_LOCK
    --gpus all
    --shm-size "${SHM_SIZE:-10gb}"
    --ulimit "memlock=-1:-1"
    --ulimit stack=67108864
    --device /dev/infiniband:/dev/infiniband
    -v "$src:/models/1m:ro"
  )
  if [[ "${ENABLE_DSPARK:-0}" == "1" ]]; then
    local dsrc
    dsrc=$(dspark_mount_src)
    [[ -f "$dsrc/model.safetensors" || -f "$dsrc/config.json" ]] \
      || die "DSpark mount src missing: $dsrc (set DSPARK_MODEL_DIR)"
    _dca+=(-v "$dsrc:$DSPARK_CONTAINER_PATH:ro")
  fi
  if [[ -f "$NCCL_HOST_DIR/libnccl.so.2.30.7" || -f "$NCCL_HOST_DIR/libnccl.so.2" ]]; then
    _dca+=(-v "$NCCL_HOST_DIR:$NCCL_CONTAINER_DIR:ro")
  else
    warn "NCCL host dir missing ($NCCL_HOST_DIR) — using Torch-bundled NCCL"
  fi
}

fabric_ip_local() {
  ip -4 -o addr show dev "$FABRIC_IFACE" 2>/dev/null \
    | awk '{print $4}' | head -1 | cut -d/ -f1
}

resolve_gid() {
  local want_ip="$1"
  local g
  g=$(gid_index_local "$want_ip" 2>/dev/null || true)
  if [[ -z "$g" ]]; then
    local fip
    fip=$(fabric_ip_local || true)
    [[ -n "$fip" && "$fip" != "$want_ip" ]] && g=$(gid_index_local "$fip" 2>/dev/null || true)
  fi
  echo "${g:-${NCCL_IB_GID_INDEX:-3}}"
}

# Worker-side docker env fragment (same AQLM + RoCE knobs as head).
worker_docker_env_lines() {
  local wip="$1" wgid="$2"
  cat <<EOF
        -e VLLM_HOST_IP=$wip -e HOST_IP=$wip \\
        -e RAY_memory_usage_threshold=0.99 -e RAY_memory_monitor_refresh_ms=0 \\
        -e CUDA_DEVICE_ORDER=PCI_BUS_ID -e CUDA_DEVICE_MAX_CONNECTIONS=32 \\
        -e CUTE_DSL_ARCH=sm_121a -e TORCH_CUDA_ARCH_LIST=12.1a \\
        -e OMP_NUM_THREADS=16 \\
        -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \\
        -e SAFETENSORS_FAST_GPU=1 \\
        -e NCCL_NET=$NCCL_NET -e NCCL_IB_DISABLE=$NCCL_IB_DISABLE -e NCCL_IB_HCA=$IB_HCA \\
        -e NCCL_SOCKET_IFNAME=$NCCL_SOCKET_IFNAME -e GLOO_SOCKET_IFNAME=$GLOO_SOCKET_IFNAME \\
        -e NCCL_P2P_DISABLE=$NCCL_P2P_DISABLE -e NCCL_SHM_DISABLE=$NCCL_SHM_DISABLE \\
        -e NCCL_MAX_NCHANNELS=$NCCL_MAX_NCHANNELS -e NCCL_MIN_NCHANNELS=$NCCL_MIN_NCHANNELS \\
        -e NCCL_BUFFSIZE=$NCCL_BUFFSIZE \\
        -e NCCL_PROTO=LL,LL128,Simple -e NCCL_CROSS_NIC=${NCCL_CROSS_NIC:-1} \\
        -e NCCL_IB_MERGE_NICS=${NCCL_IB_MERGE_NICS:-0} \\
        -e NCCL_IB_SUBNET_AWARE_ROUTING=${NCCL_IB_SUBNET_AWARE_ROUTING:-1} \\
        -e NCCL_GIN_ENABLE=${NCCL_GIN_ENABLE:-0} \\
        -e NCCL_CUMEM_ENABLE=0 -e NCCL_IGNORE_CPU_AFFINITY=1 -e NCCL_DEBUG=${NCCL_DEBUG:-WARN} \\
        -e NCCL_GRAPH_MIXING_SUPPORT=$NCCL_GRAPH_MIXING_SUPPORT \\
        -e VLLM_NCCL_SO_PATH=$VLLM_NCCL_SO_PATH \\
        -e NCCL_IB_GID_INDEX=$wgid \\
        -e FLASHINFER_DISABLE_VERSION_CHECK=1 \\
        -e VLLM_SPARSE_INDEXER_MAX_LOGITS_MB=${VLLM_SPARSE_INDEXER_MAX_LOGITS_MB:-192} \\
        -e VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=0 \\
        -e VLLM_DISABLE_FP8_W8A16=${VLLM_DISABLE_FP8_W8A16:-1} \\
        -e VLLM_MTP_INDEX_SHARE=${VLLM_MTP_INDEX_SHARE:-1} \\
        -e GLM_MOE_LANE_ROWS=${GLM_MOE_LANE_ROWS:-1} \\
        -e GLM_NVFP4_LUT256=${GLM_NVFP4_LUT256:-1} \\
        -e GLM_SHARED_EXPERTS_DEBUG=${GLM_SHARED_EXPERTS_DEBUG:-1} \\
        -e VLLM_GLM_TP_PAD=${VLLM_GLM_TP_PAD:-1} \\
        -e VLLM_MARLIN_USE_ATOMIC_ADD=${VLLM_MARLIN_USE_ATOMIC_ADD:-0} \\
        -e VLLM_WORKER_MULTIPROC_METHOD=spawn \\
        -e VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 \\
        -e VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=${VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS:-1800} \\
        -e MODEL_DIR=/models/1m \\
EOF
}

# ─── commands ─────────────────────────────────────────────────────────────────

cmd_doctor() {
  echo "=== doctor (GLM-5.2 NVFP4+AQLM · 3× Spark) ==="
  echo "head:     $(hostname) / $(whoami) @ $HEAD_IP"
  echo "workers:  ${WORKER_USER}@${WORKER_HOSTS[*]}  (fabric ${WORKER_IPS[*]})"
  echo "image:    $IMAGE  (fork: $VLLM_FORK_URL @$VLLM_FORK_BRANCH)"
  echo "model:    $MODEL_DIR  →  $COMMON_MODEL"
  if [[ "${ENABLE_DSPARK:-0}" == "1" ]]; then
    echo "dspark:   $DSPARK_MODEL_DIR  →  $COMMON_DSPARK  →  $DSPARK_CONTAINER_PATH"
  fi
  echo "parallel: TP=$TP_SIZE DCP=$DCP_SIZE PP=$PP_SIZE  len=$MAX_MODEL_LEN  util=$GPU_MEM_UTIL"
  echo "serve:    $(spec_label)  kv=$KV_CACHE_DTYPE  port=$PORT"
  echo

  local ok=0
  command -v docker >/dev/null || { warn "docker missing on head"; ok=1; }
  command -v nvidia-smi >/dev/null && info "GPU: $(nvidia-smi -L | head -1)" || { warn "nvidia-smi missing"; ok=1; }
  [[ -f "$SSH_IDENTITY" ]] && info "SSH key $SSH_IDENTITY" || { warn "SSH key missing"; ok=1; }

  local host_arch
  host_arch=$(uname -m)
  case "$host_arch" in
    aarch64|arm64) host_arch=arm64 ;;
    x86_64|amd64) host_arch=amd64 ;;
  esac
  info "host arch: $host_arch (GB10 expects arm64 + sm121)"

  if docker image inspect "$IMAGE" >/dev/null 2>&1; then
    local img_arch
    img_arch=$(docker image inspect -f '{{.Architecture}}' "$IMAGE" 2>/dev/null || echo '?')
    if [[ "$img_arch" != "$host_arch" ]]; then
      err "IMAGE ARCH MISMATCH: $IMAGE is '$img_arch' but this Spark is '$host_arch'"
      err "  Official Dockerfile.glm52-sm120 is linux/amd64 (RTX PRO 6000 / sm120)."
      err "  Need an arm64+sm121 build of jarrelscy/vllm-glm52-sm120 — not published."
      ok=1
    else
      info "image arch OK ($img_arch)"
    fi
  else
    warn "image $IMAGE not present — run: ./start.sh build  (then ./start.sh pull)"
    if [[ "$host_arch" == "arm64" ]]; then
      info "Use ./start.sh build with Dockerfile.glm52-sm121 for native arm64/sm121"
    fi
  fi

  if [[ -f "$MODEL_DIR/config.json" ]]; then
    info "checkpoint config.json OK"
    local n
    n=$(find "$MODEL_DIR" -maxdepth 1 -name 'model-*-of-*.safetensors' 2>/dev/null | wc -l | tr -d ' ')
    if [[ "${n:-0}" -ge "$EXPECTED_SHARDS" ]]; then
      info "checkpoint shards OK ($n / $EXPECTED_SHARDS)"
    else
      warn "partial checkpoint ($n / $EXPECTED_SHARDS shards) — run: ./start.sh download"
      ok=1
    fi
  else
    warn "checkpoint not downloaded yet — run: ./start.sh download"
    ok=1
  fi

  if (( TP_SIZE * PP_SIZE != 3 )); then
    warn "TP×PP=$((TP_SIZE * PP_SIZE)) but cluster has 3 GPUs — expect Ray world-size mismatch"
  fi
  info "VLLM_GLM_TP_PAD: heads 64→96, MoE I 2048→2112 (see TP3.md)"
  warn "AQLM ≈292 GB on 3×~121 GB → ~97 GB/rank; start maxlen=$MAX_MODEL_LEN DCP=$DCP_SIZE spec=$(spec_label)"
  warn "kv-cache-dtype MUST be fp8_ds_mla (not plain fp8)"

  local h
  for h in "${WORKER_HOSTS[@]}"; do
    if remote_on "$h" "hostname" >/tmp/glm52a-host-"$h".txt 2>/tmp/glm52a-ssh-"$h".err; then
      info "SSH $h OK → $(tr -d '\r' </tmp/glm52a-host-"$h".txt)"
      remote_on "$h" "command -v docker >/dev/null && nvidia-smi -L | head -1" \
        || { warn "docker/GPU check failed on $h"; ok=1; }
    else
      err "SSH to $h FAILED"
      cat /tmp/glm52a-ssh-"$h".err || true
      ok=1
    fi
  done

  if [[ "$ok" -ne 0 ]]; then
    if [[ "${DOCTOR_STRICT:-1}" == "1" ]]; then
      die "doctor found blocking issues (see above)"
    fi
    warn "doctor: issues found (continuing because DOCTOR_STRICT=0)"
    return 1
  fi
  info "doctor: ready"
}

cmd_build() {
  info "=== build $IMAGE from $VLLM_FORK_DIR ($DOCKERFILE) ==="
  local host_arch
  host_arch=$(uname -m)
  case "$host_arch" in
    aarch64|arm64)
      info "Building native arm64/sm121 image ($DOCKERFILE)"
      ;;
    *)
      warn "Host is $host_arch — Spark serve needs arm64. Building anyway."
      ;;
  esac

  mkdir -p "$(dirname "$VLLM_FORK_DIR")"
  if [[ ! -d "$VLLM_FORK_DIR/.git" && ! -f "$VLLM_FORK_DIR/$DOCKERFILE" ]]; then
    info "cloning → $VLLM_FORK_DIR"
    git clone --depth 1 -b "$VLLM_FORK_BRANCH" "$VLLM_FORK_URL" "$VLLM_FORK_DIR"
  fi

  [[ -f "$VLLM_FORK_DIR/$DOCKERFILE" ]] \
    || die "missing $DOCKERFILE in $VLLM_FORK_DIR"

  info "docker build -f $DOCKERFILE -t $IMAGE ."
  (cd "$VLLM_FORK_DIR" && docker build -f "$DOCKERFILE" -t "$IMAGE" .)
  docker image inspect "$IMAGE" >/dev/null
  local img_arch
  img_arch=$(docker image inspect -f '{{.Architecture}}' "$IMAGE")
  info "built $IMAGE arch=$img_arch"
  case "$(uname -m)" in
    aarch64|arm64)
      [[ "$img_arch" == "arm64" ]] || die "expected arm64 image, got $img_arch"
      ;;
  esac
}

cmd_pull() {
  # Default: ensure image on head (build if missing), then docker save → rsync → load.
  # Set PULL_FROM_REGISTRY=1 only if IMAGE is a registry ref that workers can pull.
  info "=== docker image → all nodes ($IMAGE) ==="
  ensure_ssh_keys

  if [[ "${PULL_FROM_REGISTRY:-0}" == "1" ]]; then
    info "PULL_FROM_REGISTRY=1 — pulling on every node"
    docker pull "$IMAGE"
    local h
    for h in "${WORKER_HOSTS[@]}"; do
      info "pulling on $h ..."
      remote_on "$h" --timeout "${PULL_TIMEOUT:-0}" "docker pull $(printf '%q' "$IMAGE")"
    done
    info "pull complete"
    return 0
  fi

  if ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
    info "image missing on head — running build"
    cmd_build
  else
    info "image already on head"
  fi

  local head_id
  head_id=$(docker image inspect -f '{{.Id}}' "$IMAGE")
  info "head image id: $head_id"

  local h need=()
  for h in "${WORKER_HOSTS[@]}"; do
    local wid
    wid=$(remote_on "$h" --timeout 60 \
      "docker image inspect -f '{{.Id}}' $(printf '%q' "$IMAGE") 2>/dev/null" | tr -d '\r' || true)
    if [[ "$wid" == "$head_id" ]]; then
      info "$h already has same image id — skip"
    else
      need+=("$h")
      info "$h needs image (have: ${wid:-none})"
    fi
  done

  if [[ "${#need[@]}" -eq 0 ]]; then
    info "all workers already in sync"
    return 0
  fi

  local tar="${IMAGE_TAR:-/var/tmp/glm52-aqlm-image.tar}"
  info "docker save → $tar (one-time; large)..."
  docker save -o "$tar" "$IMAGE"
  ls -lh "$tar"

  local ssh_e
  ssh_e=$(ssh_rsync_e)
  for h in "${need[@]}"; do
    info "rsync image tar → $h:/var/tmp/ ..."
    remote_on "$h" --timeout 60 "mkdir -p /var/tmp"
    rsync -aH --info=progress2 -e "$ssh_e" \
      "$tar" "${WORKER_USER}@${h}:/var/tmp/glm52-aqlm-image.tar"
    info "docker load on $h ..."
    remote_on "$h" --timeout "${PULL_TIMEOUT:-0}" \
      "docker load -i /var/tmp/glm52-aqlm-image.tar && rm -f /var/tmp/glm52-aqlm-image.tar"
  done

  if [[ "${KEEP_IMAGE_TAR:-0}" != "1" ]]; then
    rm -f "$tar"
  fi
  info "image sync complete"
}

cmd_download() {
  info "=== download $HF_REPO → $MODEL_DIR ==="
  mkdir -p "$MODEL_DIR"
  if [[ -f "$MODEL_DIR/config.json" && -f "$MODEL_DIR/model.safetensors.index.json" ]]; then
    local n
    n=$(find "$MODEL_DIR" -maxdepth 1 -name 'model-*-of-*.safetensors' 2>/dev/null | wc -l | tr -d ' ')
    if [[ "${n:-0}" -ge "$EXPECTED_SHARDS" ]]; then
      info "checkpoint already present ($n safetensors) — skip download"
    else
      warn "partial download ($n shards) — resuming"
      hf download "$HF_REPO" --local-dir "$MODEL_DIR"
    fi
  else
    command -v hf >/dev/null || die "hf CLI missing (pip install -U huggingface_hub)"
    hf download "$HF_REPO" --local-dir "$MODEL_DIR"
  fi
  [[ -f "$MODEL_DIR/config.json" ]] || die "download incomplete: no config.json"
  local n
  n=$(find "$MODEL_DIR" -maxdepth 1 -name 'model-*-of-*.safetensors' 2>/dev/null | wc -l | tr -d ' ')
  [[ "${n:-0}" -ge "$EXPECTED_SHARDS" ]] \
    || warn "expected $EXPECTED_SHARDS shards, have $n — download may still be running"

  mkdir -p "$(dirname "$COMMON_MODEL")"
  ln -sfn "$MODEL_DIR" "$COMMON_MODEL"
  info "head: $COMMON_MODEL → $MODEL_DIR"
}

cmd_sync() {
  info "=== rsync checkpoint → workers (local NVMe) ==="
  ensure_ssh_keys
  [[ -f "$MODEL_DIR/config.json" ]] || die "no checkpoint — run ./start.sh download first"

  local h
  for h in "${WORKER_HOSTS[@]}"; do
    info "rsync → $h:~/${WORKER_MODEL_LOCAL}/"
    remote_on "$h" "mkdir -p ~/${WORKER_MODEL_LOCAL} /var/tmp"
    local link_root_q
    if [[ -n "$WEIGHT_LINK_ROOT" ]]; then
      link_root_q=$(printf '%q' "$WEIGHT_LINK_ROOT")
      if [[ -n "${WORKER_PASS:-}" ]]; then
        remote_on "$h" "echo $(printf '%q' "$WORKER_PASS") | sudo -S bash -lc '
          mkdir -p $(printf '%q' "$WEIGHT_LINK_ROOT")
          chown -R ${WORKER_USER}:${WORKER_USER} $(printf '%q' "$(dirname "$WEIGHT_LINK_ROOT")")
        ' 2>/dev/null || true"
      fi
    else
      link_root_q='"$HOME/models/hf"'
    fi
    rsync -aH --info=progress2 \
      -e "$(ssh_rsync_e)" \
      "$MODEL_DIR/" \
      "${WORKER_USER}@${h}:~/${WORKER_MODEL_LOCAL}/"

    remote_on "$h" "
      set -e
      LINK_ROOT=${link_root_q}
      mkdir -p \"\$LINK_ROOT\" 2>/dev/null || true
      ln -sfn \"\$HOME/${WORKER_MODEL_LOCAL}\" \"\$LINK_ROOT/${WEIGHT_NAME}\" 2>/dev/null || true
      ln -sfn \"\$HOME/${WORKER_MODEL_LOCAL}\" $COMMON_MODEL
      test -f $COMMON_MODEL/config.json
      echo SYNC_OK_$h
    "
  done
  info "sync complete"
}

cmd_sync_dspark() {
  info "=== rsync DSpark draft → workers ==="
  ensure_ssh_keys
  [[ -f "$DSPARK_MODEL_DIR/model.safetensors" ]] \
    || die "no DSpark checkpoint at $DSPARK_MODEL_DIR"

  mkdir -p "$(dirname "$COMMON_DSPARK")"
  ln -sfn "$DSPARK_MODEL_DIR" "$COMMON_DSPARK"

  local h
  for h in "${WORKER_HOSTS[@]}"; do
    info "rsync DSpark → $h:~/${WORKER_DSPARK_LOCAL}/"
    remote_on "$h" "mkdir -p ~/${WORKER_DSPARK_LOCAL} /var/tmp"
    rsync -aH --info=progress2 \
      -e "$(ssh_rsync_e)" \
      "$DSPARK_MODEL_DIR/" \
      "${WORKER_USER}@${h}:~/${WORKER_DSPARK_LOCAL}/"
    remote_on "$h" "
      set -e
      ln -sfn \"\$HOME/${WORKER_DSPARK_LOCAL}\" $COMMON_DSPARK
      test -f $COMMON_DSPARK/model.safetensors
      echo DSPARK_SYNC_OK_$h
    "
  done
  info "DSpark sync complete"
}

cmd_mount() {
  info "=== SSHFS mount head checkpoint onto workers ==="
  ensure_ssh_keys
  local HEAD_USER="${HEAD_USER:-$USER}"
  local h
  for h in "${WORKER_HOSTS[@]}"; do
    if ! remote_ok_on "$h" "command -v sshfs"; then
      remote_on "$h" "sudo -n apt-get update -qq && sudo -n apt-get install -y sshfs" \
        || die "Install sshfs on $h"
    fi
    local link_root_q
    if [[ -n "$WEIGHT_LINK_ROOT" ]]; then
      link_root_q=$(printf '%q' "$WEIGHT_LINK_ROOT")
      if [[ -n "${WORKER_PASS:-}" ]]; then
        remote_on "$h" "echo $(printf '%q' "$WORKER_PASS") | sudo -S bash -lc '
          mkdir -p $(printf '%q' "$WEIGHT_LINK_ROOT")
          chown -R ${WORKER_USER}:${WORKER_USER} $(printf '%q' "$(dirname "$WEIGHT_LINK_ROOT")")
        ' 2>/dev/null || true"
      fi
    else
      link_root_q='"$HOME/models/hf"'
    fi
    remote_on "$h" "
      set -e
      KEY=\$HOME/.ssh/id_ed25519_shared
      HEAD='${HEAD_USER}@${HEAD_IP}'
      LINK_ROOT=${link_root_q}
      DST=\"\$LINK_ROOT/${WEIGHT_NAME}\"
      mkdir -p \"\$DST\"
      if ! mountpoint -q \"\$DST\"; then
        sshfs -o ro,noatime,reconnect,ServerAliveInterval=15,IdentityFile=\$KEY,IdentitiesOnly=yes \
          \"\$HEAD:$MODEL_DIR\" \"\$DST\"
      fi
      ln -sfn \"\$DST\" $COMMON_MODEL
      test -f $COMMON_MODEL/config.json
      echo MOUNT_OK
    " | tee "/tmp/glm52a-mount-$h.log"
    grep -q MOUNT_OK "/tmp/glm52a-mount-$h.log" || die "mount failed on $h"
  done
  ln -sfn "$MODEL_DIR" "$COMMON_MODEL"
  info "mount OK (SSHFS) — prefer ./start.sh sync for load speed"
}

cmd_ray() {
  info "=== start Ray cluster (3× docker · TP world) ==="
  [[ -f "$COMMON_MODEL/config.json" ]] || {
    ln -sfn "$MODEL_DIR" "$COMMON_MODEL"
    [[ -f "$COMMON_MODEL/config.json" ]] || die "missing $COMMON_MODEL — download+sync first"
  }
  if [[ "${ENABLE_DSPARK:-0}" == "1" ]]; then
    ln -sfn "$DSPARK_MODEL_DIR" "$COMMON_DSPARK"
    [[ -f "$COMMON_DSPARK/model.safetensors" ]] \
      || die "missing DSpark at $COMMON_DSPARK — download RedHatAI/GLM-5.2-speculator.dspark-preview"
    local need_ds=0 h
    for h in "${WORKER_HOSTS[@]}"; do
      if ! remote_ok_on "$h" "test -f $COMMON_DSPARK/model.safetensors"; then
        need_ds=1
      fi
    done
    if [[ "$need_ds" -eq 1 ]]; then
      cmd_sync_dspark
    fi
  fi
  ensure_ssh_keys

  local GID_HEAD GID1 GID2
  GID_HEAD=$(resolve_gid "$HEAD_IP")
  GID1=$(gid_index_remote "${WORKER_HOSTS[0]}" "$WORKER1_IP" | tr -d '\r' || true)
  GID2=$(gid_index_remote "${WORKER_HOSTS[1]}" "$WORKER2_IP" | tr -d '\r' || true)
  GID1="${GID1:-${NCCL_IB_GID_INDEX:-3}}"
  GID2="${GID2:-${NCCL_IB_GID_INDEX:-3}}"
  info "RoCEv2 GID indexes: head=$GID_HEAD w1=$GID1 w2=$GID2"

  docker rm -f "$HEAD_CTN" "$SERVE_CTN" >/dev/null 2>&1 || true
  local h
  for h in "${WORKER_HOSTS[@]}"; do
    remote_on "$h" "docker rm -f $WORKER_CTN $SERVE_CTN >/dev/null 2>&1 || true" || true
  done

  if [[ "${DROP_CACHES:-0}" == "1" ]]; then
    info "dropping page caches (sudo -n)..."
    sudo -n sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches' 2>/dev/null || true
    for h in "${WORKER_HOSTS[@]}"; do
      remote_on "$h" "sudo -n sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches' 2>/dev/null || true" || true
    done
  fi

  info "Starting Ray head container..."
  local -a head_args=()
  docker_common_args head_args
  docker_env_args head_args "$HEAD_IP" "$GID_HEAD"
  docker run -d --name "$HEAD_CTN" \
    --entrypoint bash \
    "${head_args[@]}" \
    "$IMAGE" \
    -lc "$(vllm_bash "mkdir -p '$OBJECT_SPILLING_DIR' && ray start --head \
      --node-ip-address=$HEAD_IP --port=$RAY_PORT \
      --object-store-memory=$OBJECT_STORE \
      --object-spilling-directory='$OBJECT_SPILLING_DIR' \
      --num-cpus=1 --num-gpus=1 \
      --include-dashboard=false --include-log-monitor=false \
      --disable-usage-stats --temp-dir=/tmp/ray-vllm-head --block")"

  sleep 5

  info "Starting Ray workers..."
  local idx=0
  local wip wgid
  for h in "${WORKER_HOSTS[@]}"; do
    wip="${WORKER_IPS[$idx]}"
    if [[ "$idx" -eq 0 ]]; then wgid="$GID1"; else wgid="$GID2"; fi

    remote_on "$h" "
      set -e
      test -f $COMMON_MODEL/config.json || { echo 'MISSING $COMMON_MODEL on $h — run sync/mount'; exit 1; }
      test -d /dev/infiniband || { echo 'MISSING /dev/infiniband on $h'; exit 1; }
      SRC=\$(readlink -f $COMMON_MODEL)
      test -d \"\$SRC\" || { echo 'model src not a dir: '\$SRC; exit 1; }
      DSPARK_VOL=''
      if [ '${ENABLE_DSPARK:-0}' = '1' ]; then
        test -f $COMMON_DSPARK/model.safetensors || { echo 'MISSING $COMMON_DSPARK on $h — run sync_dspark'; exit 1; }
        DSRC=\$(readlink -f $COMMON_DSPARK)
        DSPARK_VOL=\"-v \$DSRC:$DSPARK_CONTAINER_PATH:ro\"
      fi
      NCCL_VOL=''
      if [ -f \$HOME/nccl-2.30.7/libnccl.so.2.30.7 ]; then
        NCCL_VOL=\"-v \$HOME/nccl-2.30.7:$NCCL_CONTAINER_DIR:ro\"
      fi
      docker run -d --name $WORKER_CTN \
        --entrypoint bash \
        --network host --ipc host --privileged --cap-add IPC_LOCK --gpus all \
        --shm-size ${SHM_SIZE:-10gb} \
        --ulimit memlock=-1:-1 --ulimit stack=67108864 \
        --device /dev/infiniband:/dev/infiniband \
        -v \"\$SRC\":/models/1m:ro \$DSPARK_VOL \$NCCL_VOL \\
$(worker_docker_env_lines "$wip" "$wgid")
        $(printf '%q' "$IMAGE") \
        -lc \"cd /opt/vllm && source .venv/bin/activate && mkdir -p '$OBJECT_SPILLING_DIR' && ray start \
          --address=$HEAD_IP:$RAY_PORT --node-ip-address=$wip \
          --object-store-memory=$OBJECT_STORE \
          --object-spilling-directory='$OBJECT_SPILLING_DIR' \
          --num-cpus=1 --num-gpus=1 \
          --include-log-monitor=false --disable-usage-stats \
          --temp-dir=/tmp/ray-vllm-worker --block\"
    "
    idx=$((idx + 1))
  done

  info "Waiting for 3.0 GPU in Ray..."
  local i
  for i in $(seq 1 60); do
    if docker exec "$HEAD_CTN" bash -lc \
      "cd /opt/vllm && source .venv/bin/activate && ray status --address=$HEAD_IP:$RAY_PORT 2>/dev/null | grep -qE '[0-9.]+/3\\.0 GPU'"; then
      docker exec "$HEAD_CTN" bash -lc \
        "cd /opt/vllm && source .venv/bin/activate && ray status --address=$HEAD_IP:$RAY_PORT" | sed -n '1,40p'
      info "Ray cluster ready (3 GPUs)"
      return 0
    fi
    sleep 3
  done
  docker exec "$HEAD_CTN" bash -lc \
    "cd /opt/vllm && source .venv/bin/activate && ray status --address=$HEAD_IP:$RAY_PORT || true" || true
  die "Ray did not reach 3 GPUs — check worker containers / fabric"
}

_spec_config() {
  if [[ "${ENABLE_DSPARK:-0}" == "1" ]]; then
    # Match RedHat card shape. draft_tensor_parallel_size=1 is required so
    # SpeculativeConfig verify passes before runtime (draft still loads under
    # target TP; qwen3_dflash pads 64→96 heads via VLLM_GLM_TP_PAD).
    local model="${DSPARK_SPEC_MODEL:-$DSPARK_CONTAINER_PATH}"
    printf '{"model":"%s","num_speculative_tokens":%s,"method":"dspark","draft_tensor_parallel_size":1}' \
      "$model" "$DSPARK_SPEC_TOKENS"
  else
    printf '{"method":"deepseek_mtp","num_speculative_tokens":%s}' \
      "$MTP_SPEC_TOKENS"
  fi
}

_compilation_config() {
  # Env-driven CUDA graph mode. Capture sizes default to multiples of (1+k)*seqs when unset.
  local mode="${CUDAGRAPH_MODE:-PIECEWISE}"
  local fuse="${FUSE_GEMM_COMMS:-0}"
  local sizes="${CUDAGRAPH_CAPTURE_SIZES:-}"
  local k=0
  local seqs="${MAX_NUM_SEQS:-1}"
  if [[ "${ENABLE_DSPARK:-0}" == "1" ]]; then
    k="${DSPARK_SPEC_TOKENS:-0}"
  elif [[ "${ENABLE_MTP:-0}" == "1" ]]; then
    k="${MTP_SPEC_TOKENS:-0}"
  fi
  if [[ -z "$sizes" ]]; then
    local step=$((1 + k))
    local max=$((step * seqs * 6))
    [[ "$max" -lt "$step" ]] && max="$step"
    local s="$step" i
    sizes="$s"
    for ((i = 2; i <= 6; i++)); do
      s=$((step * i))
      [[ "$s" -gt "$max" ]] && break
      sizes+=",$s"
    done
  fi
  # FUSE_PASSES: comma list of fusion flags (no spaces). Examples:
  #   act | norm | attn | ar | rope_mla | act,norm
  # Do NOT include gemm/sp — Model Runner V2 rejects sequence parallelism.
  local fuse_passes="${FUSE_PASSES:-}"
  local -a pass_kv=()
  local tok
  IFS=',' read -ra _fuse_toks <<< "$fuse_passes"
  for tok in "${_fuse_toks[@]}"; do
    tok="${tok// /}"
    [[ -z "$tok" ]] && continue
    case "$tok" in
      act|fuse_act_quant) pass_kv+=('"fuse_act_quant":true') ;;
      norm|fuse_norm_quant) pass_kv+=('"fuse_norm_quant":true') ;;
      attn|fuse_attn_quant) pass_kv+=('"fuse_attn_quant":true') ;;
      ar|fuse_allreduce_rms|allreduce) pass_kv+=('"fuse_allreduce_rms":true') ;;
      rope_mla|fuse_rope_kvcache_cat_mla|rope) pass_kv+=('"fuse_rope_kvcache_cat_mla":true') ;;
      gemm|fuse_gemm_comms|sp|enable_sp)
        warn "FUSE_PASSES ignores '$tok' (blocked: Model Runner V2 / SP)"
        ;;
      *)
        warn "unknown FUSE_PASSES token '$tok' — ignoring"
        ;;
    esac
  done
  local pass="{}"
  if [[ "$fuse" == "1" ]]; then
    # NOTE (2026-07-22 A/B): fuse_gemm_comms auto-enables enable_sp → V2 hard-fail
    # or silent clear. Keep FUSE_GEMM_COMMS=0; use FUSE_PASSES for other fusions.
    warn "FUSE_GEMM_COMMS=1 is unsupported on this stack — ignoring (use FUSE_PASSES)"
  fi
  if [[ ${#pass_kv[@]} -gt 0 ]]; then
    local joined
    joined=$(IFS=,; echo "${pass_kv[*]}")
    pass="{$joined}"
  fi
  case "$mode" in
    PIECEWISE|FULL|FULL_AND_PIECEWISE) ;;
    *)
      warn "unknown CUDAGRAPH_MODE=$mode — falling back to PIECEWISE"
      mode=PIECEWISE
      ;;
  esac
  if [[ "$mode" == "PIECEWISE" ]]; then
    # Keep historical minimal config for PIECEWISE (vLLM fills capture sizes).
    printf '{"mode":3,"cudagraph_mode":"PIECEWISE"}'
  else
    printf '{"mode":3,"cudagraph_mode":"%s","cudagraph_capture_sizes":[%s],"pass_config":%s}' \
      "$mode" "$sizes" "$pass"
  fi
}

cmd_serve() {
  DOCTOR_STRICT=0 cmd_doctor || true

  if [[ ! -f "$MODEL_DIR/config.json" ]]; then
    cmd_download
  else
    ln -sfn "$MODEL_DIR" "$COMMON_MODEL"
  fi

  local need_sync=0
  local h
  for h in "${WORKER_HOSTS[@]}"; do
    if ! remote_ok_on "$h" "test -f $COMMON_MODEL/config.json"; then
      need_sync=1
      break
    fi
  done
  if [[ "$need_sync" -eq 1 ]]; then
    if [[ "${USE_SSHFS:-0}" == "1" ]]; then
      cmd_mount
    else
      cmd_sync
    fi
  fi

  if ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
    cmd_pull
  fi

  local img_arch host_arch
  img_arch=$(docker image inspect -f '{{.Architecture}}' "$IMAGE" 2>/dev/null || echo '?')
  host_arch=$(uname -m)
  case "$host_arch" in aarch64|arm64) host_arch=arm64 ;; x86_64|amd64) host_arch=amd64 ;; esac
  if [[ "$img_arch" != "$host_arch" && "${FORCE_SERVE:-0}" != "1" ]]; then
    die "refusing serve: image arch $img_arch != host $host_arch (FORCE_SERVE=1 to override)"
  fi

  if [[ "${SKIP_RAY:-0}" == "1" ]] && docker ps --format '{{.Names}}' | grep -qx "$HEAD_CTN"; then
    info "SKIP_RAY=1 — reusing existing Ray head ($HEAD_CTN)"
    if ! docker exec "$HEAD_CTN" bash -lc \
      "cd /opt/vllm && source .venv/bin/activate && ray status --address=$HEAD_IP:$RAY_PORT 2>/dev/null | grep -qE '[0-9.]+/3\\.0 GPU'"; then
      die "SKIP_RAY=1 but Ray does not show 3.0 GPU — run ./start.sh ray first"
    fi
  else
    cmd_ray
  fi
  # GCS can lag a few seconds after status reports ready
  sleep 5

  # Run the API server INSIDE the Ray head container.
  # A separate host-network container resolves Ray's get_node_ip_address() to the
  # LAN IP (e.g. 192.168.x) while cluster nodes are on fabric 10.0.0.x, so
  # ray.init fails with: No node info found matching attributes.
  info "=== launch vLLM serve inside $HEAD_CTN (TP=$TP_SIZE DCP=$DCP_SIZE $(spec_label) · $KV_CACHE_DTYPE · :$PORT) ==="
  docker rm -f "$SERVE_CTN" >/dev/null 2>&1 || true
  docker exec "$HEAD_CTN" bash -lc 'pkill -f "[v]llm serve" >/dev/null 2>&1 || true; rm -f /tmp/vllm-serve.log' || true

  local CC SPEC_FLAG="" ASYNC_FLAG="" HF_FLAG="" KV_MEM_FLAG=""
  CC=$(_compilation_config)
  if spec_enabled; then
    SPEC_FLAG="--speculative-config $(printf '%q' "$(_spec_config)")"
  fi
  if [[ "${ASYNC_SCHEDULING:-0}" == "1" ]]; then
    ASYNC_FLAG="--async-scheduling"
  fi
  # Pass JSON overrides via container env to avoid nested-quote corruption.
  if [[ -n "${HF_OVERRIDES:-}" ]]; then
    HF_FLAG='--hf-overrides "$HF_OVERRIDES"'
  fi
  if [[ -n "${KV_CACHE_MEMORY_BYTES:-}" ]]; then
    KV_MEM_FLAG="--kv-cache-memory-bytes $KV_CACHE_MEMORY_BYTES"
  fi

  docker exec -d \
    -e "RAY_ADDRESS=$HEAD_IP:$RAY_PORT" \
    -e "HF_OVERRIDES=${HF_OVERRIDES:-}" \
    "$HEAD_CTN" \
    bash -lc "$(vllm_bash "vllm serve /models/1m \
      --served-model-name $(printf '%q' "$SERVED_MODEL_NAME") \
      --host 0.0.0.0 --port $PORT \
      --trust-remote-code \
      --distributed-executor-backend ray \
      --tensor-parallel-size $TP_SIZE \
      --pipeline-parallel-size $PP_SIZE \
      --decode-context-parallel-size $DCP_SIZE \
      --dcp-comm-backend $DCP_COMM_BACKEND \
      --kv-cache-dtype $KV_CACHE_DTYPE \
      --gpu-memory-utilization $GPU_MEM_UTIL \
      --max-model-len $MAX_MODEL_LEN \
      --max-num-seqs $MAX_NUM_SEQS \
      --max-num-batched-tokens $MAX_NUM_BATCHED_TOKENS \
      --no-enable-flashinfer-autotune \
      --compilation-config $(printf '%q' "$CC") \
      --enable-auto-tool-choice --tool-call-parser glm47 \
      $SPEC_FLAG $ASYNC_FLAG $HF_FLAG $KV_MEM_FLAG >>/tmp/vllm-serve.log 2>&1")"

  info "vllm serve started in $HEAD_CTN — logging to $SERVE_LOG (container:/tmp/vllm-serve.log)"
  : >"$SERVE_LOG"
  if [[ -f "$ROOT/logs/glm52-aqlm-logtail.pid" ]]; then
    kill "$(cat "$ROOT/logs/glm52-aqlm-logtail.pid")" 2>/dev/null || true
  fi
  (
    while docker exec "$HEAD_CTN" test -f /tmp/vllm-serve.log 2>/dev/null; do
      docker exec "$HEAD_CTN" tail -n +1 -F /tmp/vllm-serve.log
      break
    done
  ) >>"$SERVE_LOG" 2>&1 &
  echo $! >"$ROOT/logs/glm52-aqlm-logtail.pid"

  local deadline=$((SECONDS + ${VLLM_ENGINE_READY_TIMEOUT_S:-3600}))
  while (( SECONDS < deadline )); do
    if curl -sf "http://127.0.0.1:${PORT}/v1/models" >/dev/null 2>&1; then
      info "Server is up on :$PORT"
      docker exec "$HEAD_CTN" bash -lc 'grep -E "Application startup complete|GPU KV cache|Maximum concurrency|error|Error" /tmp/vllm-serve.log' 2>/dev/null | tail -30 || true
      return 0
    fi
    if ! docker exec "$HEAD_CTN" bash -lc 'pgrep -f "[v]llm serve" >/dev/null'; then
      err "vllm serve exited inside $HEAD_CTN — last log lines:"
      docker exec "$HEAD_CTN" bash -lc 'tail -60 /tmp/vllm-serve.log' 2>/dev/null || true
      tail -60 "$SERVE_LOG" 2>/dev/null || true
      exit 1
    fi
    sleep 5
  done
  die "Timed out waiting for server. See: $SERVE_LOG / docker exec $HEAD_CTN tail /tmp/vllm-serve.log"
}

cmd_stop() {
  # Delegate to stop.sh (full multi-node teardown).
  exec "$ROOT/stop.sh" "$@"
}

cmd_status() {
  echo "=== status ==="
  docker ps --format 'table {{.Names}}\t{{.Status}}\t{{.Image}}' \
    | grep -E 'NAMES|glm52-aqlm' || warn "no aqlm containers on head"
  if docker ps --format '{{.Names}}' | grep -qx "$HEAD_CTN"; then
    docker exec "$HEAD_CTN" bash -lc \
      "cd /opt/vllm && source .venv/bin/activate && ray status --address=$HEAD_IP:$RAY_PORT" 2>/dev/null \
      | sed -n '1,35p' || warn "ray status failed"
  fi
  local h
  for h in "${WORKER_HOSTS[@]}"; do
    remote_on "$h" "docker ps --format '{{.Names}} {{.Status}}' | grep glm52-aqlm || echo 'no aqlm ctn'" \
      || warn "status SSH $h failed"
    remote_on "$h" "test -f $COMMON_MODEL/config.json && echo 'weights: OK' || echo 'weights: MISSING'" \
      || true
  done
  if curl -sf "http://127.0.0.1:${PORT}/v1/models" >/dev/null; then
    info "API up on :$PORT"
  else
    warn "API not responding on :$PORT"
  fi
  [[ -f "$SERVE_LOG" ]] && echo "---- log tail ----" && tail -20 "$SERVE_LOG"
}

cmd_smoke() {
  info "smoke test (thinking off)..."
  local out err
  out=$(mktemp)
  err=$(mktemp)
  if ! curl -sS --max-time 180 -o "$out" -w "%{http_code}" \
      "http://127.0.0.1:${PORT}/v1/chat/completions" \
      -H 'Content-Type: application/json' \
      -d "{\"model\":\"$SERVED_MODEL_NAME\",\"temperature\":0,\"max_tokens\":64,
           \"chat_template_kwargs\":{\"enable_thinking\":false},
           \"messages\":[{\"role\":\"user\",\"content\":\"What is 17*23? Then name the capital of Australia.\"}]}" \
      >"$err"; then
    warn "curl failed: $(cat "$err")"
    cat "$out" 2>/dev/null || true
    rm -f "$out" "$err"
    return 1
  fi
  local code
  code=$(cat "$err")
  if [[ "$code" != "200" ]]; then
    warn "HTTP $code"
    head -c 2000 "$out"; echo
    rm -f "$out" "$err"
    return 1
  fi
  python3 -c "import sys,json; r=json.load(open(sys.argv[1])); m=r['choices'][0]['message']; print(m.get('content')); print('tokens', r.get('usage'))" "$out"
  rm -f "$out" "$err"
}

# ─── main ─────────────────────────────────────────────────────────────────────
CMD="${1:-serve}"
shift || true
case "$CMD" in
  doctor)          cmd_doctor "$@";;
  build)           cmd_build "$@";;
  pull)            cmd_pull "$@";;
  download)        cmd_download "$@";;
  sync)            cmd_sync "$@";;
  sync_dspark)     cmd_sync_dspark "$@";;
  mount)           cmd_mount "$@";;
  ray)             cmd_ray "$@";;
  serve|start|"")  cmd_serve "$@";;
  stop|down)       cmd_stop "$@";;
  status)          cmd_status "$@";;
  smoke|test)      cmd_smoke "$@";;
  -h|--help|help)
    sed -n '2,45p' "$0"
    ;;
  *)
    die "Unknown command: $CMD (try: serve|doctor|build|pull|download|sync|sync_dspark|mount|ray|stop|status|smoke)"
    ;;
esac
