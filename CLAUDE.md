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
| `moe.py`, `ops/trtllm_bmm.py`, `ops/moe.py` | FlashInfer `trtllm_gen` SM100 BMMs, NVFP4/MXFP8 activation quantization fused into routing and SwiGLU, SM100 row shuffle / scale swizzle at load | `moe_marlin.py` (`Inkling2MarlinExperts`): raw load, Marlin repack in `process_weights_after_loading`, torch routing (stable sort for low-index ties), `fused_marlin_moe` with a custom softcapped-SwiGLU activation. NVFP4 global scale per expert = `amax / outer_scale_denominator` | Written. **Blocked on the image** (no bf16 sm_80 Marlin in `inkling-2-small`); rerun on the nightly-based image |
| `ops/*.py` Triton kernels (~25 sites in 8 files) | `gdc_wait` / `gdc_launch_dependents` / `launch_pdl=True` (programmatic dependent launch): ptxas rejects `griddepcontrol` below sm_90 | `ops/pdl.py`: constexpr-gated wrappers and a `PDL` flag, on for SM90+ only | **Done**, compiles on A100 (attention tests exercise `gate_sink` and the prologue's neighbors) |
| `ops/linear.py` | FlashInfer CuTe DSL `mm_bf16` (<=32 rows) and a Triton TMA kernel (router, >=2048 rows) | cuBLAS for every batch; the plain Triton K-group kernel for the router | Written, untested |
| `ops/allreduce.py`, `model.py` forward | NVLink multicast Lamport all-reduce; seq-parallel and fused conv-mix are gated on its workspace | `Inkling2AllReduceWorkspace.create` returns `None` off SM100, which also disables `seq_parallel` and `conv_mix` (`model.py:226-276`): unfused NCCL path, no `_C.inkling2_*` ops | Nothing to write; verify in the truncated-model run |
| `vision.py` | `from vllm.third_party.tml_fa4 import flash_attn_varlen_func` for the ViT | SDPA / FA2 varlen fallback, or serve **text-only** first | **Not started**. Imports fine (checked), but would fail on the first image request |
| `dspark.py`, `ops/dspark.py` | FlashInfer `rmsnorm` / `fused_add_rmsnorm`, TRT-LLM fused all-reduce + norm, `_C.inkling2_dspark_argmax` over NVLink multicast | Torch/`ops.norm` RMSNorms, unfused all-reduce, torch argmax over `all_gather` | **Not started**. Needed only for speculative decoding; serve without DSpark first |
| `ops/{norm,attn_res,sconv,activation,vision}.py` | Plain Triton (`tldevice` intrinsics) | None expected beyond the PDL gate | `rms_norm` verified on A100; the rest gets covered by the truncated-model run |

Numerics: the model was trained with NVFP4 *activations*. BF16 activations on Ampere are more precise but a
different path, so quality needs a check (ChatHPC `bench/effort-sweep.py`, a few MMMU/HLE samples via
`thinking-machines-lab/inkling-verify`), not an assumption.

## Schedule

| Day | Work | Done when |
|---|---|---|
| **Thu 11-12** | Checkpoint + image, `check-ampere.sh`, MoE and attention fallbacks, PDL gate, A100 tests | **Done except the MoE tests**, which wait on the nightly-based image |
| **Fri 11-13** | Build `vllm-inkling2-ampere.sif` (nightly `be47189f` + branch). `diag-marlin.sh` on the plain nightly (all 6 cases should pass), then `test-ampere.sh` (MoE incl. real layers 0 and 40). Truncated model (`--hf-overrides` ~6 layers incl. global layer 5) on 1-2 A100s, eager, text only, logits vs a dequantized torch reference | MoE tests pass; truncated-model logits match |
| **Sat 11-14** | Full model **TP8** on the 4 nodes, eager, text only, no DSpark. Effort sweep for quality | **Go/no-go (Sat night)**: correct answers through the API |
| **Sun 11-15** | CUDA graphs, then DSpark fallbacks + DSpark, then the ViT fallback (one change per restart) | Faster, still correct |
| **Mon 11-16** | Scripts in `../ChatHPC` (`llm-prod-inkling2-*.sh`), LiteLLM entry, benchmark with `bench/bench-concurrency.py`, go live | Public endpoint serves it |

**Fallback (track A)**: if no-go, scale the existing Inkling-Small AWQ DEP setup to the 4 new nodes for
Monday, and keep porting over the three weeks. Keep an eye on `avtc/vllm` and cyankiwi in case someone else
publishes an Ampere port.

## Status (2026-11-12, evening)

Branch head: `69555f2` on `jakehlee/inkling2-ampere`. Cluster scripts: `../ChatHPC/inkling2/` (not yet
committed to ChatHPC; copied to the cluster by hand).

Done:
- [x] Branch from PR #67325; Blackwell dependency inventory (table above)
- [x] Checkpoint downloaded to `/scratch/chathpc-dev/vllm/huggingface/hub` (snapshot `e242a97e`, 393.5 GB, 36 files)
- [x] `check-ampere.sh` on gn11 (A100 80GB PCIe) against `inkling-2-small`: vllm `0.1.dev25249+gb974b306f`,
  torch 2.14 / CUDA 13.0; Marlin ops registered; every inkling2 module imports; Triton `rms_norm` correct on sm80.
  Checkpoint layout confirmed: layer 0 NVFP4 (`uint8` [4608, 2304] w13, e4m3 scales, `amax` ~0.08-0.12,
  denominator 2688), layer 40 MXFP8 (e4m3 [4608, 4608] w13, `uint8` scales)
- [x] PDL gate (`ops/pdl.py`): ptxas rejects `griddepcontrol` on sm_80
- [x] Attention (`ops/attention_triton.py`): **40/40** `test_attention_triton.py` on gn13
- [x] Root cause of `no kernel image is available` (`diag-marlin.sh`): fp16 Marlin works, bf16 Marlin missing.
  `inkling-2-small` is a local build for `TORCH_CUDA_ARCH_LIST=10.0+PTX` (from its registry config). Use CI
  nightly `nightly-be47189f37f6162991366ff6b91539f6e9d1f432` (`7.5 8.0 8.6 8.9 9.0 10.0 12.0`) as the base:
  `ChatHPC/inkling2/vllm-inkling2-ampere.def`
- [x] Routing tie-break (stable sort) and test-fixture teardown fixed

In progress:
- [x] Nightly `be47189f` pulled as `/scratch/chathpc-dev/vllm-openai_nightly-be47189f.sif`;
  `/scratch/chathpc-dev/vllm-inkling2-ampere.sif` built from the `.def`
- [x] `diag-marlin.sh` on the plain nightly: **all 6 Marlin cases pass**, bf16 included (rel. error ~2e-3);
  481 sm_80 Marlin kernels in each of `_C` and `_moe_C` (vs 226 and 481 fp16-only in `inkling-2-small`)
- [ ] `test-ampere.sh` on the new image: MoE tests (random weights, EP=4 emulation, real layers 0 and 40),
  attention still 40/40

Next:
- [ ] Truncated-model run (~6 layers incl. global layer 5) on 1-2 A100s: verifies `linear`, the unfused
  all-reduce path, `sconv`/`attn_res`/`norm`, and weight loading end to end
- [ ] Serving scripts for TP8 over 4 x 2 A100 (node-local checkpoint staging, `/tmp` compile caches, IB
  NCCL settings from the Inkling 1 scripts)
- [ ] Full model, eager, text only, no DSpark -> effort sweep -> CUDA graphs
- [ ] DSpark fallbacks (`dspark.py`, `ops/dspark.py`)
- [ ] ViT attention fallback (`vision.py`), or ship text-only
- [ ] Quality check vs the Blackwell numbers (NVFP4 activations there, BF16 here)
