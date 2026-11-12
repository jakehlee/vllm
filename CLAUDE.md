# Inkling 2 Small on Ampere (A100) — porting plan

Goal: serve `thinkingmachines/Inkling-2-Small` (NVFP4 checkpoint, unmodified) on **4 nodes x 2 A100-80GB
(8 GPUs, 640 GB)** for ChatHPC, live by **Monday 2026-11-16**. Hardware is available for ~3 weeks from then.
Deployment scripts, cluster gotchas, and benchmarks live in `../ChatHPC` (read its `AGENTS.md`: most of the
Inkling 1 lessons — node-local checkpoint staging, `/tmp` compile caches, `VLLM_ENGINE_READY_TIMEOUT_S`,
`apptainer exec` JSON quoting — apply here unchanged).

This branch (`inkling2-ampere`) starts from vllm-project/vllm PR #67325 (`d664cd1`, "[Model] Add Inkling 2
Small", WoosukKwon), which states **"Blackwell (SM100) only"**. Remote `avtc` has the Inkling 1 Ampere port
(`feature/inkling-w4a16-mixed-precision-ampere`), whose Triton relative-attention fallback we reuse. Push to
`jakehlee` (`jakehlee/vllm`, same fork network).

## Ground rules

- **Do not touch the Blackwell path.** Every Ampere change sits behind a capability check (SM < 100) or a
  separate code path, so the diff stays reviewable against the PR and rebases cleanly when the PR changes.
- **No requantization by default.** The routed experts are the only quantized weights; vLLM's Marlin MoE kernels
  run NVFP4 (`prepare_nvfp4_moe_layer_for_marlin`) and MXFP8 (`prepare_mxfp8_moe_layer_for_marlin`) weights as
  weight-only on SM80 with BF16 activations. Everything else in the checkpoint is BF16.
- **The image is the base.** Like Inkling 1, build `vllm-inkling2-ampere.sif` = `vllm/vllm-openai:inkling-2-small`
  plus the changed `.py` files from this branch, **if** that image ships SM80 kernels (checked in step 1). A
  source build is the fallback and costs most of a day.
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

| Where | Blackwell dependency | Ampere plan |
|---|---|---|
| `ops/attention.py` (1.3k lines), `attention.py` | Prefill: vendored `tml_fa4` (`flash_fwd_sm100`). Decode/verify: Gluon kernels using `gl.nvidia.blackwell` (TMEM, tcgen05 MMA, TMA) | Port avtc's `vllm/models/inkling/nvidia/ops/triton_rel_attention.py` (757 lines, already works on SM80). Inkling 2 adds the sink gate, log-scaling tau, FP32 output, and pre-sheared rel bias: diff against the PR's torch reference in `tests/models/inkling2/test_attention.py`. **Biggest risk.** |
| `moe.py`, `ops/trtllm_bmm.py`, `ops/moe.py` | Expert GEMMs are FlashInfer `trtllm_gen` SM100 BMMs, with NVFP4/MXFP8 activation quantization fused into `route` and `swiglu_quantize`. The loader shuffles rows and swizzles scales into the SM100 layout | **Written, untested**: `moe_marlin.py` (`Inkling2MarlinExperts`, picked in `Inkling2MoE` off SM100). Raw row-major load, Marlin repack in `process_weights_after_loading`, torch routing, `fused_marlin_moe` with a custom softcapped-SwiGLU activation (routing weight applied there, `mul_topk_weights_in_gemm2=False`). NVFP4 global scale per expert = `amax / outer_scale_denominator`. Test: `tests/models/inkling2/test_moe_marlin.py` (`INKLING2_CKPT=<snapshot>` adds real layers 0 and 40) |
| `ops/dspark.py` | `torch.ops._C.inkling2_dspark_argmax` and multicast, built only for `10.0a/10.3a` | Torch argmax fallback. (`conv_mix`/`seq_parallel` ops are unreachable off SM100, see the next row) |
| `ops/allreduce.py`, `model.py` forward | NVLink multicast Lamport all-reduce; seq-parallel and fused conv-mix are gated on the workspace | `Inkling2AllReduceWorkspace.create` returns `None` off SM100, which also turns off `seq_parallel` and `conv_mix` (`model.py:226-276`), so the forward takes the unfused NCCL path and never calls the `_C.inkling2_*` conv-mix ops. Verify only |
| `ops/linear.py` | FlashInfer `mm_bf16` and a Triton TMA (`TensorDescriptor`) kernel | `F.linear` / `torch.mm` on SM80 |
| `dspark.py` | FlashInfer TRT-LLM fused all-reduce + norm | Unfused all-reduce + `rms_norm` |
| `ops/{norm,attn_res,sconv,activation,mxfp8,nvfp4,vision}.py` | Plain Triton (`tldevice` intrinsics) | Expected to work on SM80; confirm with tests. `nvfp4`/`mxfp8` activation quantization is dropped on the Ampere path anyway |

Numerics: the model was trained with NVFP4 *activations*. BF16 activations on Ampere are more precise but a
different path, so quality needs a check (ChatHPC `bench/effort-sweep.py`, a few MMMU/HLE samples via
`thinking-machines-lab/inkling-verify`), not an assumption.

## Schedule

| Day | Work | Done when |
|---|---|---|
| **Thu 11-12** | (1) `../ChatHPC/inkling2/download.sh` (checkpoint + image `.sif`), then `sbatch inkling2/check-ampere.sh` on one A100: sm80 cubins in `_C`/`_moe_C`, Marlin ops present, which inkling2 modules import, and real layer-0 NVFP4 / layer-40 MXFP8 experts through Marlin vs a dequantized reference. Branch pushed to `jakehlee/vllm` | Image verdict: overlay vs source build; NVFP4 global scale = `w_amax / outer_scale_denominator` confirmed |
| **Fri 11-13** | Ampere MoE (Marlin loader + forward), `linear`, `conv_mix`, `seq_parallel` off, DSpark fallbacks. Port the PR tests to SM80 | `test_moe`, `test_core_ops` pass on 1 A100 |
| **Sat 11-14** | Attention port. Truncated model (`--hf-overrides` ~6 layers including global layer 5, a few experts' worth of memory) on 1-2 A100s, eager, text only, logits vs a dequantized torch reference | **Go/no-go (Sat night)**: attention + MoE match references |
| **Sun 11-15** | Full model DEP8 on the 4 nodes, eager, text only -> effort sweep -> CUDA graphs -> DSpark -> vision (one per restart) | Correct answers through the API |
| **Mon 11-16** | Scripts in `../ChatHPC` (`llm-prod-inkling2-*.sh`), LiteLLM entry, benchmark with `bench/bench-concurrency.py`, go live | Public endpoint serves it |

**Fallback (track A)**: if no-go, scale the existing Inkling-Small AWQ DEP setup to the 4 new nodes for
Monday, and keep porting over the three weeks. Keep an eye on `avtc/vllm` and cyankiwi in case someone else
publishes an Ampere port.

## Status

- [x] Branch from PR #67325, `avtc` remote added
- [x] Blackwell dependency inventory (table above)
- [ ] Checkpoint downloaded to `/scratch/chathpc-dev/vllm/huggingface/hub`
- [ ] `vllm-openai_inkling-2-small.sif` pulled; `check-ampere.sh` results recorded here
  - The image Dockerfile builds `TORCH_CUDA_ARCH_LIST='7.5 8.0 ... 12.0'` (CUDA 13.2), so sm80 kernels are
    expected; the check confirms it.
- [ ] Ampere MoE: written (`moe_marlin.py`), needs `pytest tests/models/inkling2/test_moe_marlin.py` on an A100
- [ ] Ampere attention
- [ ] Remaining fallbacks
- [ ] Truncated-model logits match
- [ ] Full model serving on 8 GPUs
