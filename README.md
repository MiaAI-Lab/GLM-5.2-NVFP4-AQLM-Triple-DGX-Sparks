# GLM-5.2 NVFP4+AQLM on 3× DGX Sparks • **VISION** · two KV recipes

<p align="center">
  <sub>by <a href="https://x.com/MiaAI_lab">Mia'a AI Lab</a></sub>
  <br><br>
  <a href="https://ko-fi.com/Z8Z3SPLOD" target="_blank" rel="noopener noreferrer" style="display:inline-block;margin:0 8px;vertical-align:middle;"><img src="https://storage.ko-fi.com/cdn/kofi6.png?v=6" alt="Buy Me a Coffee at ko-fi.com" height="28" style="height:28px;width:auto;vertical-align:middle;border:0;" /></a>
  <a href="https://x.com/MiaAI_lab" target="_blank" rel="noopener noreferrer" style="display:inline-block;margin:0 8px;vertical-align:middle;"><img src="https://img.shields.io/badge/Follow%20me%20on%20X-000000?style=for-the-badge&logo=x&logoColor=white" alt="Follow Mia on X" height="28" style="height:28px;width:auto;vertical-align:middle;border:0;" /></a>
</p>

**Release v4.5** — same **VISION** image (`:k12l1-vision`), two serve paths with **separate launchers and env files**:

| Path | Launcher | Env | KV cache | Context | Structured decode | Mixed decode |
|------|----------|-----|----------|---------|-------------------|--------------|
| **Max context (default)** | `./start.sh` | `.env` ← [`.env.example`](.env.example) | **`nvfp4_ds_mla`** | **~348k** (vision) / **~380k** (text) | **~21** tok/s | **~13.6–19** tok/s |
| **Coding speed** | `./start_fp8.sh` | `.env.fp8` ← [`.env.fp8.example`](.env.fp8.example) | **`fp8_ds_mla`** | **~235k** | **~25–26** tok/s | **~15.5–21** tok/s |

Same Docker image, weights, and cluster. Pick one path, don’t run both on the same `PORT`. Text-only 380k remains a **config swap** on the nvfp4 path (same image).

Serve [jarrelscy/GLM-5.2-NVFP4-AQLM-hybrid](https://huggingface.co/jarrelscy/GLM-5.2-NVFP4-AQLM-hybrid) (~272 GB text + ~1 GB **vision** on disk) with a **VISION-enabled** build of [jarrelscy/vllm-glm52-sm120](https://github.com/jarrelscy/vllm-glm52-sm120) on **three NVIDIA DGX Spark** nodes (GB10 / sm_121 / aarch64) over RoCE. **Text + image input**: the checkpoint's current revision adds a `glm5v` **VISION** build (MoonViT tower + patch-merger projector grafted onto the unchanged text backbone).

This is **not** stock vLLM. The hybrid checkpoint needs the fork’s `nvfp4_aqlm_hybrid` path and TP3 head/MoE padding (`VLLM_GLM_TP_PAD`); the **vision** tower additionally needs `--mm-encoder-tp-mode data` at TP3 (its 16 attention heads are not divisible by 3).

## Two serve paths (nvfp4 vs fp8 KV)

Both paths share the **same** image (`:k12l1-vision`), checkpoint, MTP-3, graphs, and MoE knobs. Only **KV cache dtype / pin / max context** and the **launcher + env file** differ.

| | **Path A — max context** | **Path B — coding speed** |
|--|--------------------------|---------------------------|
| **When to use** | Long agents, huge paste, max window | IDE / code gen; prefer tok/s over 300k+ ctx |
| **Launcher** | `./start.sh` | `./start_fp8.sh` |
| **Env file** | `.env` (from `.env.example`) | `.env.fp8` (from `.env.fp8.example`) |
| **KV dtype** | `nvfp4_ds_mla` | `fp8_ds_mla` |
| **KV pin** | **11 GiB** (vision) / 12 GiB (text-only) | **12 GiB** |
| **KV pool** | **354,496** (vision) / **386,688** (text) | **~240,640** (measured) |
| **`MAX_MODEL_LEN`** | **348160** / **380928** | **235392** |
| **`GPU_MEM_UTIL`** | 0.895 | 0.9 |
| **Structured decode** (code-like; warm≥5 c1) | **~21** tok/s | **~25–26** tok/s (~**+20%**) |
| **Mixed decode** (prose / chat) | **~13.6–19** tok/s | **~15.5–21** tok/s |
| **Short-prompt TTFT** | ~0.9–1.0 s | ~0.7–0.8 s |
| **40k long-ctx probe** | coherent | coherent |
| **Client context window** | 348160 (or 380928 text) | **235392** |

```bash
# Path A — max context (default)
cp .env.example .env          # first time; edit cluster secrets
./start.sh ray && ./start.sh serve

# Path B — coding speed (fp8 KV)
cp .env.fp8.example .env.fp8  # first time; edit cluster secrets
./start_fp8.sh ray && ./start_fp8.sh serve
```

`start_fp8.sh` is a full copy of `start.sh` that sources **`.env.fp8` only** (never `.env`). Same subcommands (`doctor`, `ray`, `serve`, `smoke`, …). Tear-down is still `./stop.sh` (shared containers).

**Decode note:** *structured* ≈ coding / tight format (primary for IDE use); *mixed* ≈ prose. Numbers are single-stream, `enable_thinking: false`, boot-matched on this fleet (2026-07-26).

### Shared stack (both paths)

| Metric | Value |
|--------|--------|
| Parallelism | TP3 · DCP1 · PP1 |
| Model | `Glm5vForConditionalGeneration` (glm5v: MoonViT + text backbone, 77 shards, NVFP4 hot + AQLM 2-bit cold hybrid MoE) |
| Weights revision | `HF_REVISION=53e0082e…` (**VISION**-era `main`) |
| **VISION** encoder | `MM_ENCODER_TP_MODE=data` (required at TP3 — 16 heads not divisible by 3) |
| Spec decode | MTP-3 (in-checkpoint) |
| CUDA graphs | FULL · capture `[4,8,12,16,20,24]` (incl. L1 draft size-1 capture fix) |
| Fusion | `FUSE_PASSES=ar,norm` |
| Attention/MoE | TP3 head pad **64→66** · MoE w2 lane-rows G=16 · SwiGLU-fused w13 |
| Cold-path knobs | `GLM_MOE_AQLM_CB=l1` · `GLM_MOE_AQLM_STREAM=1` · `GLM_NVFP4_STREAM=1` |
| Quant extras | fp8 W8A16 on o_proj/shared experts (`VLLM_DISABLE_FP8_W8A16=0`) |
| Indexer | `HF_OVERRIDES={"num_experts_per_tok":4,"index_topk_freq":8}` |
| API | `:8888` — OpenAI `/v1/chat/completions` **and** Anthropic `/v1/messages` |

**Text-only 380k variant** (nvfp4 path only; same image, wrapper dormant ≡ `:k12l1`): swap in the text config set, `KV_CACHE_MEMORY_BYTES=12884901888` (12 GiB), `MAX_MODEL_LEN=380928` → pool **386,688**. Use `./start.sh` + `.env`, not the fp8 path.

### **VISION** (glm5v) — the headline of **v4**, full text speed

The default **v4** recipe above **is** the **VISION** build: config is a `glm5v` wrapper (`text_config` + `vision_config`) served by the **`:k12l1-vision`** image (= `:latest`) — the v3 k12l1 text image plus the glm5v **vision** wrapper **and a config-override fix** (below). **VISION** vs. the old text-only serve is only these deltas:

```bash
HF_REVISION=53e0082eedebd806b63e19779c47905937d768ca   # VISION-era main; text-only pin is 2d2ee49…
MM_ENCODER_TP_MODE=data     # REQUIRED at TP3 — else boot dies: "16 is not divisible by 3" in kimi_k25_vit.py
KV_CACHE_MEMORY_BYTES=11811160064   # 11 GiB (was 12 GiB) — frees ~1 GiB for the replicated tower
MAX_MODEL_LEN=348160        # was 380928 — pool is now 354,496 tokens
IMAGE=ghcr.io/miaai-lab/glm-5.2-nvfp4-triple-dgx-sparks:k12l1-vision   # == :latest
# weights dir must carry the VISION config set (config.json / chat_template.jinja /
# model.safetensors.index.json from the vision revision — what ./start.sh download
# fetches at HF_REVISION=53e0082e…)
```

**The 2026-07-25 `:vision` build was ~20% slower on text decode — root-caused and fixed in v4.** Flat `--hf-overrides` keys land on the top-level `glm5v` wrapper config, and `num_experts_per_tok:4` never propagated to the nested `text_config` the MoE actually reads → the **vision** stack silently ran the MoE at **top-8 instead of top-4** (~2× routed-expert traffic; acceptance unchanged, hiding it). The fixed image adds one passthrough property. Measured on the fixed build + **VISION** config (boot-matched, warm≥5, r1+r2): structured **21.0/20.9**, mixed **13.7/13.6**, ttft **~0.9–1.0s** — text-recipe parity. Gates: single-image smoke exact (red square / white circle), 40k long-ctx probe coherent. **Do not use the old `:20260725-vision` pin** (top-8 bug).

Verified on this fleet (2026-07-26): `Multi-modal warmup completed`, KV pool 354,496 tokens, **image prompts** answered via both API shapes. Client-side gotchas (both bitten in practice): **mark the model vision-capable in your chat client** — clients like ZCode silently strip the attachment otherwise (`[Media omitted from provider request because the selected model does not support image input]`) and the model then truthfully says it can't see anything; and if you *do* enable thinking, **keep the thinking budget well below `max_tokens`** — e.g. ZCode's `budget_tokens: 32000` with `max_tokens: 32001` leaves ~1 token for the reply and surfaces as "model returned no content". Prefer thinking **off** (the server default) — GLM 5.2 tends to over-think.

Switching between **VISION** and text-only is a **config swap only** (same image): swap `config.json` / `chat_template.jinja` / `model.safetensors.index.json` on all 3 nodes + restart. Details: `dev/docs/GLM-5.2-Vision-report.md` + `dev/docs/GLM-5.2-Vision-Tech-Guide.md`; root-cause write-up and A/B numbers: `dev/docs/SPEED-IMPROVEMENTS.md` §0 "VISION-REGRESSION RESOLVED". Not covered here: the card's 950K + LMCache path (`UTIL=0.96`) — untested on this fleet.

### Disable `earlyoom` (highly recommended)

**Highly recommended:** disable `earlyoom` on **all three Sparks** before bring-up and while serving this stack.

At these KV pins (11–12 GiB) the machines sit near ~1 GiB free (often into swap). `earlyoom` (especially with a low free-mem threshold and a preference for `ray` / `python` / `vllm`) will **SIGTERM Ray workers mid CUDA-graph capture**. That looks like a GPU OOM but is the host killer. This fleet only held large-KV boots reliably after `earlyoom` was stopped.

```bash
# on every node (head + both workers)
sudo systemctl stop earlyoom
sudo systemctl disable earlyoom   # optional: keep it off across reboots
# verify
pgrep -a earlyoom || echo earlyoom_stopped
```

Re-enable later only if you drop the KV pin / context a lot, or raise earlyoom’s free-memory threshold so it won’t fire during capture.

## Requirements

- **3× DGX Spark** with Docker + NVIDIA Container Toolkit  
- RoCE / ConnectX fabric between nodes (Socket/TCP fallback possible but slower)  
- SSH from the head node to both workers  
- **≥ ~280 GB free NVMe per node** for the target checkpoint (plus draft if using DSpark)  
- Python 3 with `pexpect` on the head (`pip install pexpect`)  
- Hugging Face CLI / token to download the checkpoint  
- NCCL **≥ 2.30.7** on the host is strongly recommended for FULL graphs + TP on GB10+CX7 (mount via `NCCL_HOST_DIR`)

## Cluster layout

Run `./start.sh` on the **head**. Workers only need Docker, the image, synced weights, and SSH.

```text
head     10.0.0.1   ← you run start.sh / stop.sh here
worker1  10.0.0.2
worker2  10.0.0.3
```

Adjust IPs in `.env`.

### Same user on all Sparks (typical)

Most fleets use **one username and the same login on every node**. That is the default this repo expects:

```bash
# .env
WORKER_USER=spark          # same as `whoami` on the head
# WORKER_PASS=...          # only if you have no SSH key yet (optional)
SSH_IDENTITY=$HOME/.ssh/id_ed25519_shared
MODEL_DIR=$HOME/models/hf/GLM-5.2-NVFP4-AQLM-hybrid
VLLM_FORK_DIR=$HOME/src/vllm-glm52-sm120   # or wherever you clone the fork
NCCL_HOST_DIR=$HOME/nccl-2.30.7
```

Prefer **SSH keys** (passwordless) over `WORKER_PASS`. Generate a shared key, install the public key on all three nodes, and point `SSH_IDENTITY` at the private key.

```bash
ssh-keygen -t ed25519 -f ~/.ssh/id_ed25519_shared -N ''
for h in 10.0.0.1 10.0.0.2 10.0.0.3; do
  ssh-copy-id -i ~/.ssh/id_ed25519_shared.pub "$USER@$h"
done
```

### Different users on head vs workers (advanced)

If the head account differs from the worker account (uncommon), set:

```bash
WORKER_USER=otheruser
WORKER_PASS=...            # if key auth is not set up
WEIGHT_LINK_ROOT=/home/otheruser/models/hf   # only if you need a non-$HOME link path
```

By default, workers link weights under `$HOME/models/hf` on the remote account.

## Quick start

Choose a path first (see **[Two serve paths](#two-serve-paths-nvfp4-vs-fp8-kv)**):

| Path | Copy | Run with |
|------|------|----------|
| **A — max context** (~348k, ~21 structured tok/s) | `cp .env.example .env` | `./start.sh` |
| **B — coding speed** (~235k, ~25–26 structured tok/s) | `cp .env.fp8.example .env.fp8` | `./start_fp8.sh` |

```bash
git clone https://github.com/MiaAI-Lab/GLM-5.2-NVFP4-AQLM-Triple-DGX-Sparks.git glm52
cd glm52

# Path A (default) — or use .env.fp8.example → .env.fp8 for path B
cp .env.example .env
# edit IPs, WORKER_USER, paths, fabric NICs

pip install --user pexpect

# 1) Pull the known-good arm64/sm121 image (~39 GB) from GHCR, then distribute to workers
#    Package is public (anonymous pull works). Login optional (rate limits / GHCR quirks):
# echo "$GITHUB_TOKEN" | docker login ghcr.io -u USERNAME --password-stdin
docker pull ghcr.io/miaai-lab/glm-5.2-nvfp4-triple-dgx-sparks:latest
# .env.example already sets IMAGE to this tag; then:
./start.sh pull

# Alternative: build from the vLLM fork instead of pulling
# ./start.sh build && ./start.sh pull

# 2) Download + sync target weights
./start.sh download
./start.sh sync

# 3) Bring up Ray + vLLM
./start.sh ray
./start.sh serve
# Equivalent: ./start.sh   (default command is serve)
# Path B: use ./start_fp8.sh ray && ./start_fp8.sh serve instead

# 4) Probe (uses PORT from the active env, default 8888)
./start.sh smoke
curl -s "http://127.0.0.1:${PORT:-8888}/v1/models" | jq .

# Stop (shared for both paths)
./stop.sh
# UNMOUNT=1 ./stop.sh    # also drop SSHFS mounts if you used ./start.sh mount
```

`./start.sh` / `./start_fp8.sh` (with `serve`) will **doctor**, download/sync if weights are missing, **pull** the image if it is missing locally, start **Ray** (unless `SKIP_RAY=1`), then launch vLLM. Prefer the **GHCR image** above; `… build` is only needed if you rebuild from source.

### Docker image (GHCR)

| Tag | Notes |
|-----|--------|
| `ghcr.io/miaai-lab/glm-5.2-nvfp4-triple-dgx-sparks:latest` | **= `:k12l1-vision` — v4 default.** Superset image (2026-07-26, digest `f8f350d4…`): the k12l1 text recipe **plus** the glm5v **VISION** wrapper and the `num_experts_per_tok` override fix. Serves **both** config sets — text config (wrapper dormant ≡ `:k12l1` exactly) and **VISION** config — at full speed |
| `…:k12l1-vision` / `…:20260726-k12l1-vision` | Named + date pins of the **VISION** superset (= local tag `glm52-aqlm-sm121-baked:20260724-k12l1-vision`) |
| `…:vision` | Rolling **VISION** tag — **now = `:k12l1-vision`** (fixed). The previous 2026-07-25 build at `:20260725-vision` has the top-8 override bug (~20% slower text decode) — do not use |
| `…:k12l1` / `…:20260726-k12l1` | Named + date pins of the text-only k12l1 backport (= local tag `glm52-aqlm-sm121-baked:20260724-k12l1`, digest `da18eccd…`). Text config only — cannot load the glm5v config |
| `…:20260724` | Previous text-only build (2026-07-24 bake: fp8 W8A16 fix, pad-66 rule, MoE lane-rows, SwiGLU epilogue, M1/M2 hardening, nvfp4 FlashInfer patches). **Superseded by `:k12l1`** — same base minus K1/K2 + L1; kept as rollback pin |
| `…:pre-b4-speed` / `…:20260722` | Previous build (needs the fork re-apply list for the current recipe) |

> `:latest` = `:k12l1-vision` is the **v4** default pull and serves text **and VISION** (config swap only). `:k12l1` remains the minimal text-only (v3) pin; `:20260724` is the rollback pin.

Package and git repo are **public**. Anonymous `docker pull` works; `docker login ghcr.io` is optional.

See [CHANGELOG.md](./CHANGELOG.md) for the full release notes (v4.5 dual KV paths, v4 VISION, v3 k12l1).

### What's new in v4 (`:k12l1-vision`) — **VISION** support

**v4** makes **VISION** the default recipe. Image **`:k12l1-vision`** (= `:latest`) = v3 (`:k12l1`) **plus**:

1. **glm5v wrapper** — MoonViT tower + patch-merger projector on the unchanged text backbone (`Glm5vForConditionalGeneration`). Image input via OpenAI + Anthropic APIs.
2. **`num_experts_per_tok` passthrough fix** — flat `--hf-overrides` no longer get stuck on the top-level wrapper; top-4 MoE routing is restored (the 2026-07-25 `:vision` pin ran top-8 and lost ~20% text decode).
3. **Default serve knobs** — vision-era `HF_REVISION`, `MM_ENCODER_TP_MODE=data` at TP3, 11 GiB KV pin → **348k** ctx (354,496 pool). Text-only 380k remains a **config swap** on the same image.

Measured with vision enabled (boot-matched, warm≥5): structured **~21**, mixed **~13.6–19** tok/s — text-recipe parity. **Do not use `:20260725-vision`.**

### What's new in v3 (`:k12l1`)

v3 (`:k12l1`) is the 2026-07-24 baked image plus two kernel/graph backports developed on the serving fork’s SM121 branch — no other changes (the rest of the tree, FlashInfer, and all attention/MoE/quant code are byte-identical to `:20260724`). Superseded as the default by **v4** (`:k12l1-vision`):

1. **K1 — AQLM cold-path gather mem-path** (`aqlm_moe_v2.cu`). The 2-bit "cold" MoE experts spend ~92% of their bus traffic on random 16B codebook gathers. K1 routes codebook gathers through L1 (`GLM_MOE_AQLM_CB=l1`) and marks the code/activation streams evict-first (`GLM_MOE_AQLM_STREAM=1`). Microbench: **w13 +2.7%, w2 +22.5% sector throughput**, bit-exact vs the shipped kernel.
2. **K2 — evict-first on the NVFP4 hot weight stream** (`GLM_NVFP4_STREAM=1`, w13-only). Stops the ~210 MB/launch NVFP4 weight stream from evicting the 1 MB codebook from L2. Microbench: **w13 +7% sectors**, w2 flat, bit-exact.
3. **L1 — draft multi-step decode FULL-cudagraph capture** (`cudagraph_utils.py`). Capture lists sized for the target (4,8,12,… for MTP k=3) never produce a `decode_query_len=1` candidate, so MTP draft decode steps ran **eager forever**. The fix unions pure-decode shapes (`n_req × decode_query_len`) into the capture list; draft decode is now fully graphed at any k.

All three are **bit-exact and env-gated** — the kernels default to the old behavior unless the knobs are set (`.env.example` sets them). Measured end-to-end (2026-07-26, boot-matched, warm≥5, r1+r2): structured **20.8/21.0 tok/s** (tie with the previous best), mixed **14.0/14.0 tok/s** (best recorded on this fleet).

> If you pin an older image (`:20260724` or earlier), drop the three `GLM_MOE_AQLM_*` / `GLM_NVFP4_STREAM` knobs from your `.env` — they are ignored harmlessly, but you lose the cold-path gains.

> **2026-07-24:** the baked build lands the recipe numbers for text-only (fp8 W8A16, pad 66, lane-rows, SwiGLU-fused w13, `index_topk_freq=8`, pool 386,688) out of the box, no post-deploy patching. On the older `:20260722`/`pre-b4-speed` build, expect the previous recipe (pad 96, no fp8 W8A16, pool 248,896) unless the fork re-apply list is applied.

Set in `.env` (the current fleet recipe):

```bash
IMAGE=ghcr.io/miaai-lab/glm-5.2-nvfp4-triple-dgx-sparks:k12l1-vision
# (== :latest; local tag glm52-aqlm-sm121-baked:20260724-k12l1-vision on this fleet)
```

Then `./start.sh pull` copies that image to the workers (docker save/rsync/load). Default `pull` does **not** hit the registry on workers — it saves/loads from the head. Set `PULL_FROM_REGISTRY=1` only if every node can pull the same `IMAGE` ref itself.

## Configuration

Two env templates — **one active file per path** (never commit local secrets):

| Path | Template | Local file (gitignored) | Loaded by |
|------|----------|-------------------------|-----------|
| Max context (nvfp4) | [`.env.example`](.env.example) | `.env` | `./start.sh` |
| Coding speed (fp8) | [`.env.fp8.example`](.env.fp8.example) | `.env.fp8` | `./start_fp8.sh` |

Important knobs (both files; values differ by recipe):

| Variable | Purpose |
|----------|---------|
| `HEAD_IP` / `WORKER1_IP` / `WORKER2_IP` | Fabric / management IPs |
| `WORKER_USER` | SSH user on workers (**same as head for most people**) |
| `WORKER_PASS` | Optional password fallback for SSH/sudo |
| `SSH_IDENTITY` | Private key path |
| `IMAGE` | Docker image (`ghcr.io/...` or local tag) |
| `HF_REPO` / `MODEL_DIR` | Checkpoint source and local path |
| `TP_SIZE=3` `DCP_SIZE=1` | Keep DCP=1 on this sparse-MLA path |
| `KV_CACHE_DTYPE` | **`nvfp4_ds_mla`** (path A) or **`fp8_ds_mla`** (path B) |
| `KV_CACHE_MEMORY_BYTES` | Fixed KV budget (11 GiB / 12 GiB — see path table above) |
| `MAX_MODEL_LEN` | Per-request context cap; set ≤ pool for ≥1× concurrency |
| `GPU_MEM_UTIL` | Overall mem util (0.895 nvfp4 / 0.9 fp8); pool size is set by the pin |
| `ENABLE_MTP` / `MTP_SPEC_TOKENS` | In-checkpoint MTP speculative decode (default path) |
| `ENABLE_DSPARK` / `DSPARK_*` | External DSpark draft (see below); mutually exclusive with MTP |
| `CUDAGRAPH_MODE` / `CUDAGRAPH_CAPTURE_SIZES` | FULL graphs + capture list |
| `FUSE_PASSES` | e.g. `ar,norm` |
| `NCCL_*` / `IB_HCA` / `GLOO_SOCKET_IFNAME` | Fabric tuning |

### Path A recipe (MTP) — **nvfp4 max context** · `./start.sh`

```bash
IMAGE=ghcr.io/miaai-lab/glm-5.2-nvfp4-triple-dgx-sparks:k12l1-vision  # == :latest
HF_REVISION=53e0082eedebd806b63e19779c47905937d768ca
MM_ENCODER_TP_MODE=data
KV_CACHE_DTYPE=nvfp4_ds_mla
KV_CACHE_MEMORY_BYTES=11811160064     # 11 GiB
MAX_MODEL_LEN=348160                  # pool ~354,496 (~1.02×)
GPU_MEM_UTIL=0.895
MAX_NUM_SEQS=1
ENABLE_MTP=1
ENABLE_DSPARK=0
MTP_SPEC_TOKENS=3
CUDAGRAPH_MODE=FULL
FUSE_PASSES=ar,norm
VLLM_DISABLE_FP8_W8A16=0
HF_OVERRIDES={"num_experts_per_tok":4,"index_topk_freq":8}
```

After boot, confirm:

```text
GPU KV cache size: 354,496 tokens
Maximum concurrency for 348,160 tokens per request: 1.02x
```

### Path B recipe (MTP) — **fp8 coding speed** · `./start_fp8.sh`

```bash
# same IMAGE / HF_REVISION / MM_ENCODER_TP_MODE / MTP / graphs / HF_OVERRIDES as path A
KV_CACHE_DTYPE=fp8_ds_mla
KV_CACHE_MEMORY_BYTES=12884901888     # 12 GiB
MAX_MODEL_LEN=235392                  # pool ~240,640 (~1.02×)
GPU_MEM_UTIL=0.9
```

After boot, confirm:

```text
GPU KV cache size: 240,640 tokens   # approx; confirm on your boot
Maximum concurrency for 235,392 tokens per request: ≥1.0x
```

If FULL capture dies with worker SIGTERM, check `earlyoom` first (`systemctl stop earlyoom` on all nodes) before blaming GPU OOM — see **Disable earlyoom** above.

### Client settings (OpenAI-compatible UIs)

vLLM does **not** enforce a fixed completion length by itself. Chat apps send `max_tokens` / `max_completion_tokens`. If that budget is too small (especially with GLM **thinking** on), you get truncated replies such as *“Model stopped because it reached the maximum output token limit”*.

**Thinking / reasoning: leave it off.** GLM 5.2 tends to produce long thinking traces that burn output budget and slow replies; this stack defaults thinking **off** (`--default-chat-template-kwargs '{"enable_thinking":false}'` plus a matching chat-template default). Prefer staying with that for IDE / agent use. Only turn thinking on for hard multi-step problems, and keep any thinking budget **well below** `max_tokens`. To re-enable per request: `"chat_template_kwargs": {"enable_thinking": true}`.

Point the client at the head API and match **whichever path you booted**:

| Setting | Path A (nvfp4) | Path B (fp8) | Notes |
|---------|----------------|--------------|--------|
| Base URL | `http://<head-ip>:8888/v1` | same | `PORT` from the active env |
| Model id | `glm-5.2` | same | `SERVED_MODEL_NAME` |
| API key | any non-empty (e.g. `dummy`) | same | Auth is off on this stack |
| Context window | **348160** (text: 380928) | **235392** | Must match `MAX_MODEL_LEN` |
| Max output tokens | same as context | same as context | Keep `prompt + max_tokens ≤ MAX_MODEL_LEN`; thinking counts toward output |
| Reasoning / thinking | **off** (recommended) | same | Default off on this stack; enable only when needed |
| Image input | vision-capable | same | Clients that don't mark vision strip attachments |

Example **pi agent** entry for path A (348k); for path B set `contextWindow` / `maxTokens` to **235392** and rename accordingly:

```json
{
  "id": "glm-5.2",
  "name": "GLM 5.2 NVFP4+AQLM TP3 MTP · 348k + VISION",
  "reasoning": false,
  "input": ["text", "image"],
  "contextWindow": 348160,
  "maxTokens": 348160,
  "compat": {
    "supportsDeveloperRole": false,
    "supportsReasoningEffort": false,
    "maxTokensField": "max_tokens"
  }
}
```

Provider block: `baseUrl` `http://127.0.0.1:8888/v1`, `api` `openai-completions`, `auth` `none`. Reload/restart the client after changing these values.

### DSpark speculative decode (optional)

DSpark uses a **separate draft checkpoint**, not the in-checkpoint MTP block. TP3 is supported via the same `VLLM_GLM_TP_PAD` head-pad mechanism as the target (the DSpark draft pads 64→96; the target pads 64→66 via the backend-gated rule). `start.sh` forces `ENABLE_MTP=0` when `ENABLE_DSPARK=1`.

**Draft weights** — point `DSPARK_MODEL_DIR` at a local directory that contains `model.safetensors`:

| Source | On-disk size (approx.) | Notes |
|--------|------------------------|--------|
| [sapidlabs/Sparkulator-GLM-5.2](https://huggingface.co/sapidlabs/Sparkulator-GLM-5.2) | **~4.6 GiB** | W4A16; used in measured AQLM+TP3 DSpark runs |
| [RedHatAI/GLM-5.2-speculator.dspark-preview](https://huggingface.co/RedHatAI/GLM-5.2-speculator.dspark-preview) | **~7 GiB** | Upstream DSpark preview; larger resident footprint |

```bash
# download once (Sparkulator example)
hf download sapidlabs/Sparkulator-GLM-5.2 \
  --local-dir "$HOME/models/hf/Sparkulator-GLM-5.2"

# .env
ENABLE_DSPARK=1
ENABLE_MTP=0
DSPARK_SPEC_TOKENS=3       # K=3 measured best; K=7 collapsed mixed (~6 tok/s)
# REQUIRED — no safe default. start.sh's built-in default points at the
# RedHat bf16 preview; if the workers carry a different (e.g. W4A16
# Sparkulator) copy, serve dies at draft load: KeyError 'fc.weight_packed'.
# Head and workers MUST mount the same checkpoint.
DSPARK_MODEL_DIR=$HOME/models/hf/Sparkulator-GLM-5.2
COMMON_DSPARK=/var/tmp/glm52-dspark
# container mount path defaults to /models/dspark
# DSPARK_SPEC_MODEL=/models/dspark

# sync draft → workers, recreate Ray (mounts /models/dspark), serve
./start.sh sync_dspark
./start.sh ray          # required when enabling/disabling DSpark mounts
./start.sh serve
```

Changing `ENABLE_DSPARK` requires a fresh `./start.sh ray` (volume flags are set at container create time). `SKIP_RAY=1` alone is **not** enough after flipping DSpark on/off.

#### Context constraints with DSpark

The draft adds several GiB resident (about **4.6 GiB** for Sparkulator, more for the RedHat preview) on top of the target. At the MTP max-context envelope (~120 GiB / 121 GiB used, often swapping), that headroom is mostly gone. **Lower the KV pin back to ~8 GiB when running DSpark** (the 11–12 GiB / 348–380k pins leave no room for the draft).

| Goal | Suggested knobs | Expect |
|------|-----------------|--------|
| Proven DSpark band | `MAX_MODEL_LEN=100000`, pin **8 GiB** | Boots; about **~20 tok/s** decode on this stack |
| Long but stable | **120–150k**, pin 8 GiB (or 6 GiB if capture dies) | Best practical long-ctx + DSpark |
| Stretch | ~180k | Maybe; stop `earlyoom` first |
| MTP-class max (~380k) | — | **Not recommended** with DSpark — use MTP for max context |

DSpark is optional for draft experiments; prefer **MTP** for the default max-context recipe (memory headroom).

**Measured on the baked image (2026-07-24, Sparkulator W4A16, K=3, 100k, same-session warm c1):** structured **20.2** / mixed **12.2** tok/s — short-context parity with MTP-3 (20.5–21.0 / 13–15). But the 26k-context probe decode collapses to **~6 tok/s vs MTP's 17.6–20.4** (~3× slower): the draft runs full dense attention over the whole context every step. Acceptance length ~2.0–3.3. Net: no speed win over MTP at any context length, plus the ctx cap — DSpark stays a draft-research path, not a serving recipe.

```bash
# example: DSpark @ 150k (safer long-ctx try)
ENABLE_DSPARK=1
ENABLE_MTP=0
DSPARK_SPEC_TOKENS=3
DSPARK_MODEL_DIR=$HOME/models/hf/Sparkulator-GLM-5.2
KV_CACHE_DTYPE=nvfp4_ds_mla
KV_CACHE_MEMORY_BYTES=8589934592
MAX_MODEL_LEN=150000
MAX_NUM_SEQS=1
```

### Other variants

| Goal | Change |
|------|--------|
| Safer RAM | 8–10 GiB pin, maxlen ≈ new pool (either path) |
| Skip Ray recreate after serve-only edits | `SKIP_RAY=1 ./start.sh serve` (or `./start_fp8.sh serve`; not when toggling DSpark mounts) |

For the main **max-ctx vs coding-speed** choice, see **[Two serve paths](#two-serve-paths-nvfp4-vs-fp8-kv)** above.

## Commands

| Command | Action |
|---------|--------|
| **`./start.sh …`** | **Path A (nvfp4)** — loads **`.env`** |
| **`./start_fp8.sh …`** | **Path B (fp8)** — loads **`.env.fp8`** |
| `… doctor` | SSH, Docker, GPU, checkpoint checks |
| `… build` | Clone/build fork image (`Dockerfile.glm52-sm121`) |
| `… pull` | Distribute image to workers |
| `… download` | `hf download` target checkpoint on head |
| `… sync` | rsync target weights to workers |
| `… sync_dspark` | rsync DSpark draft to workers |
| `… ray` | Start Ray head + 2 worker containers |
| `… serve` | Launch `vllm serve` in the head container (default if no args) |
| `… status` | What’s running |
| `… smoke` | Short coherence probe against `PORT` |
| `./stop.sh` | Tear down serve + Ray containers (`UNMOUNT=1` also drops SSHFS mounts) |

## What is in this repo

Only the ops surface needed to run and stop the stack:

| File | Role |
|------|------|
| `start.sh` | Path A launcher (nvfp4 max context) |
| `start_fp8.sh` | Path B launcher (fp8 coding speed) |
| `stop.sh` | Shared teardown |
| `.env.example` | Path A template → copy to `.env` |
| `.env.fp8.example` | Path B template → copy to `.env.fp8` |
| `scripts/remote.py` | SSH helper |
| `README.md` / `CHANGELOG.md` | Docs |

Clone the vLLM fork separately (see `VLLM_FORK_*` in the env template), or pull the GHCR image. Weights, Docker images, and local `.env` / `.env.fp8` stay on your machines (gitignored).

Local experiment / port / handoff material (if present on a development checkout) lives under **`dev/`** — see `dev/README.md`. It is **not** required to serve and is gitignored.

## Known limits

- **3 Sparks / TP3** — target dims are padded (**64→66 heads** via the backend-gated rule, MoE intermediate 2048→2112); the DSpark draft pads 64→96. See fork `glm_tp_pad` / `qwen3_dflash`. A 4th Spark (TP4) is a different recipe (e.g. QuantTrio Int4–Int8Mix), not a drop-in.  
- **950K + LMCache** — the model card’s headline path (`UTIL=0.96`) is **not** this fleet’s recipe and is untested here.  
- **DCP &gt; 1** — not reliable on this sparse-MLA GB10 path; keep `DCP_SIZE=1`.  
- **Jarrelscy ~1M ctx** — needs TP4+DCP4 on 4× discrete GPUs; not this 3× DCP1 layout.  
- **DSpark vs max ctx** — draft costs several GiB; plan on **~100–150k** context, not the MTP ~348–380k ceiling.  
- **earlyoom** — **highly recommended off** on all nodes for this recipe; it will kill capture at large KV pins (see above).

## Credits

- [jarrelscy/GLM-5.2-NVFP4-AQLM-hybrid](https://huggingface.co/jarrelscy/GLM-5.2-NVFP4-AQLM-hybrid) — checkpoint (NVFP4 hot + AQLM 2-bit cold hybrid MoE)  
- [jarrelscy/vllm-glm52-sm120](https://github.com/jarrelscy/vllm-glm52-sm120) — serving fork (`glm52-sm120` branch; SM121 baked images on GHCR)  
- [sapidlabs/Sparkulator-GLM-5.2](https://huggingface.co/sapidlabs/Sparkulator-GLM-5.2) / [RedHatAI DSpark preview](https://huggingface.co/RedHatAI/GLM-5.2-speculator.dspark-preview) — optional draft  
- NVFP4 KV layout / Spark recipes inspired by community ports ([tonyd2wild](https://github.com/tonyd2wild/GLM-5.2-NVFP4-KV-4x-DGX-Spark-300kctx-42tok-s), danielwoz, CosmicRaisins, b12x, and related Spark GLM-5.2 work)

## License

Ops scripts and docs in this repository are released under the [MIT License](./LICENSE) (Copyright © 2026 Mia'a AI Lab).

Upstream model weights, the vLLM fork, and related kernel projects keep their own licenses.
