# Inkling 2 Small on Ampere (A100) — porting plan

Goal: serve `thinkingmachines/Inkling-2-Small` (NVFP4 checkpoint, unmodified) on **4 nodes x 2 A100-80GB
(8 GPUs, 640 GB)** for ChatHPC, live by **Monday 2026-11-16**. Hardware is available for ~3 weeks from then.
Deployment scripts, cluster gotchas, and benchmarks live in `../ChatHPC` (read its `AGENTS.md`: most of the
Inkling 1 lessons — node-local checkpoint staging, `/tmp` compile caches, `VLLM_ENGINE_READY_TIMEOUT_S`,
`apptainer exec` JSON quoting — apply here unchanged).

This branch (`inkling2-ampere`) starts from vllm-project/vllm PR #67325 (`d664cd1`, "[Model] Add Inkling 2
Small", WoosukKwon), which states **"Blackwell (SM100) only"**. Remote `avtc` has the Inkling 1 Ampere port
(`feature/inkling-w4a16-mixed-precision-ampere`), kept for reference (the Inkling 2 attention kernel was written
fresh). Push to remote `jakehlee` (`jakehlee/vllm`). GitHub's "N commits ahead of avtc/vllm:main" banner is
expected: the fork's parent is avtc, whose `main` is stale. The branch is upstream main `efe12305f0` + the PR +
our commits. On the cluster the repo root is `/scratch/chathpc-dev/vllm-inkling2-ampere` (so `VLLM_SRC` points there, and the
`vllm/` package is `/scratch/chathpc-dev/vllm-inkling2-ampere/vllm/`).

## Ground rules

- **Do not touch the Blackwell path.** Every Ampere change sits behind a capability check (SM < 100) or a
  separate code path, so the diff stays reviewable against the PR and rebases cleanly when the PR changes.
- **No requantization by default.** The routed experts are the only quantized weights; vLLM's Marlin MoE kernels
  run NVFP4 (`prepare_nvfp4_moe_layer_for_marlin`) and MXFP8 (`prepare_mxfp8_moe_layer_for_marlin`) weights as
  weight-only on SM80 with BF16 activations. Everything else in the checkpoint is BF16.
- **The image is the base.** Like Inkling 1, `vllm-inkling2-ampere.sif` = an official vLLM CI nightly plus this
  branch's `.py` files (`ChatHPC/inkling2/vllm-inkling2-ampere.def`). **Not** `vllm/vllm-openai:inkling-2-small`:
  it is built for `10.0+PTX` only, and its sm_80 Marlin coverage is partial (fp16 yes, bf16 no).
- **Every kernel replacement gets a test against the PR's torch reference** (`tests/models/inkling2/`), runnable
  on a single A100, before it goes into a full-model run. Full-model restarts cost 15+ minutes: change one
  variable per restart.

## Model facts (from `config.json` / `model.safetensors.index.json`)

- `Inkling2ForConditionalGeneration`, 594B total / 17B active, 48 layers: 40 sliding-window (1024) + 8 global
  (layers 5, 11, ..., 47). Hidden 4608, 48 Q / 8 KV heads, head dim 128, relative-bias attention (no RoPE),
  gated attention with a per-token sink, short convolutions, block attention residuals, logit softcap 30.
- MoE: 384 routed experts, top-8, no shared experts, expert intermediate 2304, gate bias, route scale 8,
  gate/up softcaps 4 / 25. Experts are **NVFP4 in layers 0-39** (`w13/w2_weight` packed `uint8`, e4m3 scales
  per 16, `_weight_amax` + `_weight_outer_scale_denominator`) and **MXFP8 in layers 40-47** (e4m3, ue8m0
  scales per 32).
- DSpark speculator (5 layers, RNN head, 7 tokens per step) in `mtp.safetensors` (3.4 GB). 23-layer ViT.
- Checkpoint: 36 files, **393.5 GB**, 1770 tensors.

## Parallelism: TP8 (= EP8), not DEP

The PR shards experts over the **TP group** (`moe.py`: "EP size = TP size", reduced with
`tensor_model_parallel_all_reduce`), splits attention heads over TP (48 Q / 8 KV heads, so TP <= 8), and asserts
PP == 1 (`model.py:494`). There is no DP x EP all-to-all path, so the layout is **TP8 across 4 nodes** with two
NCCL all-reduces per layer over InfiniBand (96 per token). Adding a DP x EP path is possible, but not this week.
TP8 also means attention, embeddings, and the LM head are sharded, not replicated, which helps memory:

## Memory budget at TP8 (per GPU)

| Per GPU | GB |
|---|---|
| Experts: 48 of 384 (layers 0-39 at ~0.56 B/param, 40-47 at ~1.03 B/param) | ~47 |
| BF16 non-expert weights, sharded 8 ways (ViT, DSpark, and small tensors replicated) | ~3-5 (estimate) |
| Left for KV + activations + Marlin workspace + CUDA graphs at 0.90 | **~20** |

KV per GPU is 1 KV head per layer at TP8. That's comfortable for 128k. Levers if it gets tight: higher
`--gpu-memory-utilization`, then text-only (no ViT), then **requantizing layers 40-47 from MXFP8 to NVFP4**
offline (~5.7 GB per GPU; a mechanical dequantize-and-requantize, not a calibration).

Speed: 96 cross-node all-reduces per token on InfiniBand will dominate decode at small batch, so expect
single-user speed well below Inkling 1. DSpark (7 drafted tokens per step) amortizes that, so it's high-value.
Note the gn13/gn14 IB measurement: NCCL ran NET/IB with GDR 0 across nodes (see ChatHPC memory).

## Blackwell-only code and its Ampere replacement

`vllm/models/inkling2/` is ~9.8k lines. Sorted by risk:

| Where | Blackwell dependency | Ampere replacement | State |
|---|---|---|---|
| `ops/attention.py`, `attention.py` | Prefill: vendored `tml_fa4` (`flash_fwd_sm100`). Decode/verify: Gluon kernels (TMEM, tcgen05, TMA) | `ops/attention_triton.py` (`rel_attention`): one Triton kernel for prefill, decode and verify, split-KV with a torch LSE merge. Reads the prologue's sheared rel logits directly, so `attention_prologue` and `gate_sink` stay shared. Picked off SM100 by `Inkling2Attention.use_triton`. Written fresh, not ported from avtc | **40/40 tests pass on A100**. Performance untuned |
| `moe.py`, `ops/trtllm_bmm.py`, `ops/moe.py` | FlashInfer `trtllm_gen` SM100 BMMs, NVFP4/MXFP8 activation quantization fused into routing and SwiGLU, SM100 row shuffle / scale swizzle at load | `moe_marlin.py` (`Inkling2MarlinExperts`): raw load, Marlin repack in `process_weights_after_loading`, torch routing (stable sort for low-index ties), `fused_marlin_moe` with a custom softcapped-SwiGLU activation. NVFP4 global scale per expert = `amax / outer_scale_denominator` | **13/13 tests pass on A100** (nightly-based image), incl. real full-size layers 0 (NVFP4) and 40 (MXFP8): the `amax / outer_scale_denominator` scale is confirmed |
| `ops/*.py` Triton kernels (~25 sites in 8 files) | `gdc_wait` / `gdc_launch_dependents` / `launch_pdl=True` (programmatic dependent launch): ptxas rejects `griddepcontrol` below sm_90 | `ops/pdl.py`: constexpr-gated wrappers and a `PDL` flag, on for SM90+ only | **Done**, compiles on A100 (attention tests exercise `gate_sink` and the prologue's neighbors) |
| `ops/linear.py` | FlashInfer CuTe DSL `mm_bf16` (<=32 rows) and a Triton TMA kernel (router, >=2048 rows) | cuBLAS for every batch; the plain Triton K-group kernel for the router | **Ran** in the truncated run (651030) |
| `quant_config.py` | `Inkling2QuantConfig.get_min_capability()` returned 100, so `vllm/config/vllm.py:897` rejects the checkpoint on Ampere before loading | Returns 80 | **Done** (`3460a29`); the unit tests never reached this check |
| `ops/allreduce.py`, `model.py` forward | NVLink multicast Lamport all-reduce; seq-parallel and fused conv-mix are gated on its workspace | `Inkling2AllReduceWorkspace.create` returns `None` off SM100, which also disables `seq_parallel` and `conv_mix` (`model.py:226-276`): unfused NCCL path, no `_C.inkling2_*` ops | **Ran** in the truncated run: TP2 used `CUSTOM`/`PYNCCL` all-reduce, no errors |
| `vision.py` | `tml_fa4.flash_attn_varlen_func` for the ViT attention (the other ViT kernels in `ops/vision.py` are plain Triton) | `_fa2_attention`: vLLM's FA2 varlen (`fa_version=2`, bidirectional, same `cu_seqlens`/`max_seqlen`), off SM100 | **3/3 `test_vision.py` pass on A100** (per-image SDPA reference, pixel gather, encoder CUDA-graph replay) |
| `dspark.py`, `ops/dspark.py` | FlashInfer `rmsnorm` / `fused_add_rmsnorm`, TRT-LLM fused all-reduce + norm, Gluon `block_decode_attention`, `_C.inkling2_dspark_argmax` over NVLink multicast | Off SM100 (`_SM100` in `dspark.py`): `ops.norm.rms_norm` + `_custom_ops.fused_add_rms_norm`; unfused NCCL all-reduce; `attention_triton.block_attention` (the Triton kernel's `DRAFT_BLOCK` mode); `gather_argmax` (per-shard argmax, all-gather of fp32 `(max, index)` pairs, lowest rank wins ties) | **17/17 `test_dspark_ampere.py` pass on A100** (the SM100 tests' cases ported, plus CUDA-graph replay) |
| `ops/{norm,attn_res,sconv,activation,vision}.py` | Plain Triton (`tldevice` intrinsics) | None expected beyond the PDL gate | **Ran** in the truncated run (compiled on sm_80, no errors) |

Numerics: the model was trained with NVFP4 *activations*. BF16 activations on Ampere are more precise but a
different path, so quality needs a check (ChatHPC `bench/effort-sweep.py`, a few MMMU/HLE samples via
`thinking-machines-lab/inkling-verify`), not an assumption.

## Schedule

| Day | Work | Done when |
|---|---|---|
| **Thu 11-12** | Checkpoint, images, MoE + attention fallbacks, PDL gate, A100 unit tests; DSpark + ViT fallbacks; truncated run | **Done**: 73/73 tests, truncated run answers all 5 request types |
| **Fri 11-13** (one node) | Truncated runs of each later toggle: `ENFORCE_EAGER=0`, `DSPARK=1`, `IMAGES=1`; production job scripts | CUDA graphs and DSpark (both samplers) **pass**; images pending |
| **Sat-Sun** (when gn11-14 are free) | Full model TP8 (`submit-tp8.sh`), eager, text only -> effort sweep -> CUDA graphs -> DSpark -> vision, one change per restart | Correct answers through the API |
| **Mon 11-16** | Front-end integration (gn14 job with nginx/OWUI/LiteLLM), LiteLLM entry, `bench/bench-concurrency.py`, go live | Public endpoint serves it |

**Fallback (track A)**: if no-go, scale the existing Inkling-Small AWQ DEP setup to the 4 new nodes for
Monday, and keep porting over the three weeks. Keep an eye on `avtc/vllm` and cyankiwi in case someone else
publishes an Ampere port.

## Cluster scripts (`../ChatHPC/inkling2/`, committed by the user; run from the ChatHPC repo root)

| Script | What it does |
|---|---|
| `download.sh` | Checkpoint into the GPFS HF cache (done) |
| `vllm-inkling2-ampere.def` | **The image**: CI nightly `be47189f` + this branch's `.py`. Rebuild after every `git pull` of code: `apptainer build --fakeroot /scratch/chathpc-dev/vllm-inkling2-ampere.sif vllm-inkling2-ampere.def` |
| `check-ampere.sh` + `check_ampere.py` | Image sanity: archs, ops, imports, PDL probe, real experts through Marlin |
| `diag-marlin.sh` + `diag_marlin.py` | Marlin per (activation dtype, weight format), `CUDA_LAUNCH_BLOCKING=1` |
| `test-ampere.sh` | `VLLM_SRC=/scratch/chathpc-dev/vllm-inkling2-ampere sbatch inkling2/test-ampere.sh`: MoE + attention unit tests. Binds `VLLM_SRC/vllm/models/inkling2` over the image's copy, so it tests uncommitted edits without a rebuild |
| `stage-checkpoint.sh` | Sourced helper: 16-way parallel copy of the checkpoint to `/tmp/$USER-models/Inkling-2-Small` (~3 min; checks free space: an old Inkling 1 copy may still sit in `/tmp/$USER-models`) |
| `truncated-gn13.sh` | First 12 layers (global layers 5 and 11), TP2 on gn13; then 5 request types (+ an image request with `IMAGES>0`) against port 3011. Same toggles as `tp8-node.sh`: `ENFORCE_EAGER=0`, `DSPARK=1` (`DSPARK_SAMPLE=greedy` for the argmax path), `IMAGES=1`. Serves a symlink view of the staged checkpoint with a rewritten `config.json` (`num_hidden_layers`, and DSpark `target_layer_ids` remapped to the last 7 truncated layers + 12): `--hf-overrides` doesn't reach the draft config. `NUM_LAYERS` >= 7 |
| `engine.sh` | Sourced by both TP8 jobs: the one vLLM engine definition (TP8 args, toggles `ENFORCE_EAGER` / `DSPARK` / `IMAGES` / `IMAGE`, default image `vllm-inkling2-ampere-dspark.sif`), node IPs `192.168.12.41`-`.44` and ranks (gn14 0, gn11 1, gn12 2, gn13 3), checkpoint staging, workers waiting on gn14:29560. Flip a default here once TP8 confirms it |
| `tp8-node.sh` + `submit-tp8.sh` | Bring-up: TP8 head on gn14 (API on port 3012, no front end) + headless gn11-13, reservation `ChatHPC_pilot_warmup`. `tp8-node.sh` is also the headless worker for production |
| `prod-gn14.sh` + `submit-prod.sh` | Production: TP8 head on gn14 (API on 3010, `inkling2-small:128k`, API key and OTLP as before) + the whole front end (nginx, OWUI, LiteLLM, Grafana, TEI, Docling, Kokoro, PR-Agent, from the Inkling 1 DEP job), workers via `tp8-node.sh`. **Not yet run.** Stop the current gn14 job first |
| `litellm.yml` | Bound into the production LiteLLM as `config_inkling2.yml`: `inkling2-small:128k`, with `inkling-small:128k`, `gemma4:31b-128k`, `laguna-s:118b-128k` as aliases |

Cluster paths: repo root `/scratch/chathpc-dev/vllm-inkling2-ampere` (`git pull` there), images in
`/scratch/chathpc-dev/`, logs in `/home/jakelee/out/<jobid>.{out,err}`.

## Status (2026-11-12, late night) — handoff

Branch head: see `git log` (everything is pushed to `jakehlee/inkling2-ampere`). The cluster image
`/scratch/chathpc-dev/vllm-inkling2-ampere.sif` was built from `3460a29` (no DSpark/ViT fallbacks) and is the
known-good baseline for the truncated run. **Rebuild from the current head to a new file**
(`vllm-inkling2-ampere-dspark.sif`) and keep the old one for A/B if something regresses.

Done:
- [x] Branch from PR #67325; Blackwell dependency inventory (table above)
- [x] Checkpoint on GPFS (`/scratch/chathpc-dev/vllm/huggingface/hub`, snapshot `e242a97e`, 393.5 GB, 36 files)
- [x] `check-ampere.sh` (gn11): checkpoint layout confirmed (layer 0 NVFP4 `uint8` [4608, 2304] w13 with e4m3
  scales, `amax` ~0.08-0.12, denominator 2688; layer 40 MXFP8 e4m3 [4608, 4608] w13, `uint8` scales)
- [x] PDL gate (`ops/pdl.py`): ptxas rejects `griddepcontrol` on sm_80
- [x] Image: `inkling-2-small` is built for `TORCH_CUDA_ARCH_LIST=10.0+PTX` and lacks bf16 sm_80 Marlin. CI
  nightly `nightly-be47189f37f6162991366ff6b91539f6e9d1f432` has all 6 Marlin cases passing (`diag-marlin.sh`)
- [x] `get_min_capability` 100 -> 80 (`3460a29`)
- [x] DSpark draft + ViT fallbacks (`4d8e614`; see the table)
- [x] **73/73 unit tests on gn13** (`test-ampere.sh` at `345a36b`): Marlin MoE 13 (incl. real layers 0 and 40),
  Triton attention 40 (still passing after the `DRAFT_BLOCK` edit), DSpark 17, ViT 3
- [x] **Truncated run** (`truncated-gn13.sh`, job 651030, 12 layers, TP2, eager, text only, old image): all 5
  request types returned 200, no errors. Weights loaded in 18.6 s from local SSD (staging 276 s); **41.36 GiB per
  GPU** for 12 layers at TP2, which matches the expert estimate (~47 GiB per GPU at TP8 for all 48); 24 GiB KV
  (120k tokens); ~25 tok/s single-stream eager. Output is gibberish, as expected with 36 layers missing
- [x] gn11-14 IPs confirmed: `192.168.12.41`-`.44` (as in `tp8-node.sh`)

Next. Steps 1-4 need one node; the TP8 steps need all four.
1. [x] Image rebuilt from the DSpark/ViT head: `/scratch/chathpc-dev/vllm-inkling2-ampere-dspark.sif` (now the
   default in every script). The truncated runs below ran on gn12, so gn12 (like gn13) already has the
   checkpoint staged in `/tmp/jakelee-models`
2. [x] `ENFORCE_EAGER=0` truncated (651038): **breakable CUDA graphs captured** (5 PIECEWISE + 4 FULL sizes,
   4 s, 0.09 GiB), all 5 request types pass, no errors. No speed gain at TP2 x 12 layers (25.3 tok/s vs 25 eager:
   one request, and the Triton attention runs outside graphs); the real comparison is at TP8
3. [x] `DSPARK=1` truncated, probabilistic (651043) and greedy / `gather_argmax` (651044): draft loads
   (`Inkling2DSparkModel`, +1.7 GiB per GPU at TP2), all 5 request types pass, `SpecDecoding metrics` logged.
   Acceptance 0.1% (1 of ~1500 drafted), **expected**: the draft reads features of a 12-layer target it was
   never trained on. With nothing accepted, decode was 18-20 tok/s vs 25 without DSpark (7 drafted tokens
   per step cost ~25% here). Real acceptance needs the full model
4. [ ] `IMAGES=1` truncated (651045): processor, ViT on FA2, embedding merge, one image request. Ran; log not
   yet reviewed
5. [ ] TP8 on gn11-14 (`inkling2/submit-tp8.sh`), eager, text only: check worker logs first
6. [ ] Full model quality: effort sweep (`bench/effort-sweep.py`, answer 4.22 km/s) through port 3012
7. [ ] TP8 `ENFORCE_EAGER=0`, then `DSPARK=1` (check acceptance in `SpecDecoding metrics`), then `IMAGES=4`,
   one change per restart
8. [ ] Front end: **written** (`prod-gn14.sh`, `submit-prod.sh`, `litellm.yml`; see the scripts table), not yet
   run. Once TP8 bring-up passes: stop the current gn14 job, `inkling2/submit-prod.sh`, check OWUI, then
   `bench/bench-concurrency.py --model inkling2-small:128k` against the Inkling 1 table in ChatHPC `AGENTS.md`

Known risks:
- DSpark at TP8: each greedy draft step all-gathers across nodes (7 per draft); probabilistic sampling
  all-gathers full-vocab logits per step. Both are small next to the target's 96 all-reduces per token,
  but if acceptance is good and speed is not, compute the draft logits replicated instead
- `_custom_ops.fused_add_rms_norm` may round the residual differently from FlashInfer's: only acceptance, not
  correctness, can move
