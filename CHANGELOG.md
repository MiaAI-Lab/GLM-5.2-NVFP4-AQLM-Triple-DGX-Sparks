# Changelog

All notable releases of this ops recipe (images, default serve knobs, and docs) are listed here.

## [v4.5] — 2026-07-27 — dual KV paths (nvfp4 max-ctx + fp8 coding speed)

**Headline:** two first-class serve recipes on the same **VISION** image (`:k12l1-vision` / `:latest`). No new Docker bake required.

### Added

- **Path A — max context (default):** `./start.sh` + `.env` (from `.env.example`)  
  - `KV_CACHE_DTYPE=nvfp4_ds_mla` · 11 GiB pin · `MAX_MODEL_LEN=348160` (vision) / 380928 (text-only config swap)  
  - Structured **~21** tok/s · mixed **~13.6–19** tok/s  
  - KV pool **354,496** (vision) / **386,688** (text)

- **Path B — coding speed:** `./start_fp8.sh` + `.env.fp8` (from `.env.fp8.example`)  
  - `KV_CACHE_DTYPE=fp8_ds_mla` · 12 GiB pin · `MAX_MODEL_LEN=235392` · `GPU_MEM_UTIL=0.9`  
  - Structured **~25** tok/s (~**+20%** vs path A) · mixed **~15.5–21** tok/s  
  - KV pool **~240,640** (measured); 40k long-ctx probe coherent  
  - `start_fp8.sh` is a full launcher copy that sources **`.env.fp8` only** (never `.env`)

- **Docs:** README leads with a side-by-side path table (launcher, env, KV dtype, context, decode tok/s); quick start, config recipes, client settings, and commands updated for both paths.

### Unchanged

- Image tag still **`:k12l1-vision`** (= `:latest`); same weights, MTP-3, graphs, MoE knobs, vision stack as v4.  
- Tear-down remains **`./stop.sh`** (shared). Do not run both serves on the same `PORT`.

```bash
# Path A — max context
cp .env.example .env && ./start.sh ray && ./start.sh serve

# Path B — coding speed
cp .env.fp8.example .env.fp8 && ./start_fp8.sh ray && ./start_fp8.sh serve
```

| | Path A (nvfp4) | Path B (fp8) |
|--|----------------|--------------|
| Launcher | `./start.sh` | `./start_fp8.sh` |
| Env | `.env` | `.env.fp8` |
| Context | ~348k / ~380k | ~235k |
| Structured tok/s | ~21 | ~25 |
| Mixed tok/s | ~13.6–19 | ~15.5–21 |

---

## [v4] — 2026-07-26 — **VISION** support

**Headline:** multimodal **VISION** is the default recipe. Image tag **`:k12l1-vision`** (= `:latest`).

### Added

- **glm5v vision stack** on the k12l1 base: MoonViT tower + patch-merger projector grafted onto the unchanged NVFP4+AQLM text backbone (`Glm5vForConditionalGeneration`).
- Default weights pin to vision-era `main` (`HF_REVISION=53e0082e…`); ~272 GB text + ~1 GB vision on disk.
- `MM_ENCODER_TP_MODE=data` at TP3 (required: vision tower has 16 attention heads, not divisible by 3).
- Image input via OpenAI `/v1/chat/completions` and Anthropic `/v1/messages`.
- Config-swap path: same image serves **VISION @ 348k** (11 GiB KV, pool 354,496) or **text-only @ 380k** (12 GiB, pool 386,688).

### Fixed

- **`num_experts_per_tok` passthrough on `Glm5vConfig`.** Flat `--hf-overrides` were written only on the top-level wrapper; the nested `text_config` kept the checkpoint default (top-8). The vision path silently ran ~2× routed-expert traffic and lost ~20% text decode. Fixed with a passthrough property (same pattern as `index_topk_freq`).
- **Do not use** the intermediate pin `:20260725-vision` (top-8 bug). Use `:k12l1-vision` / `:latest` / `:20260726-k12l1-vision`.

### Performance (vision enabled, boot-matched)

| | tok/s (approx.) |
|--|-----------------|
| Structured | **~21** |
| Mixed | **~13.6–19** (context-dependent) |

Parity with the text-only k12l1 recipe. Image smokes (red square / white circle) and long-ctx probes verified on this fleet.

### Image tags

| Tag | Role |
|-----|------|
| `:latest` / `:k12l1-vision` / `:20260726-k12l1-vision` | **v4 default** — text + VISION |
| `:vision` | Rolling alias → fixed k12l1-vision (not the buggy 20260725 build) |

---

## [v3] — 2026-07-26 — k12l1 cold-path + draft capture (`:k12l1`)

**Headline:** text-only bake with K1/K2 cold-path kernels and L1 FULL-cudagraph draft capture. Image tag **`:k12l1`**.

Built on the 2026-07-24 baked image (`:20260724`): same tree, FlashInfer, attention/MoE/quant code — only the three backports below differ.

### Added

1. **K1 — AQLM cold-path gather mem-path** (`aqlm_moe_v2.cu`)  
   2-bit “cold” MoE experts spend most bus traffic on random 16B codebook gathers. K1 routes codebook gathers through L1 (`GLM_MOE_AQLM_CB=l1`) and marks code/activation streams evict-first (`GLM_MOE_AQLM_STREAM=1`). Microbench: **w13 +2.7%, w2 +22.5%** sector throughput, bit-exact.

2. **K2 — NVFP4 hot weight stream** (`GLM_NVFP4_STREAM=1`, w13-only)  
   Stops the large NVFP4 weight stream from thrashing the 1 MB codebook out of L2. Microbench: **w13 +7%** sectors, w2 flat, bit-exact.

3. **L1 — draft multi-step decode FULL-cudagraph capture** (`cudagraph_utils.py`)  
   Capture lists sized for the target (4, 8, 12, … for MTP k=3) never produced a pure `decode_query_len=1` shape, so MTP draft steps stayed eager. L1 unions pure-decode shapes into the capture list; draft decode is fully graphed at any k.

All three knobs are **bit-exact and env-gated** (defaults preserve old behavior; `.env.example` enables them).

### Performance (text-only, boot-matched)

| | tok/s (approx.) |
|--|-----------------|
| Structured | **~20.8–21.0** |
| Mixed | **~14.0** (best recorded on this fleet that day) |

### Image tags

| Tag | Role |
|-----|------|
| `:k12l1` / `:20260726-k12l1` | **v3** text-only pin (cannot load glm5v config) |
| `:20260724` | Pre-k12l1 bake (fp8 W8A16, pad-66, lane-rows, SwiGLU-fused w13, …) — rollback |
| `:pre-b4-speed` / `:20260722` | Older build; needs fork re-apply for modern recipe numbers |

### Note

**v4** supersedes v3 as the documented default (`:latest` = `:k12l1-vision`). Text-only 380k on v4 is a config swap on the vision image; pin `:k12l1` only if you want a minimal text-only image without the glm5v wrapper.
