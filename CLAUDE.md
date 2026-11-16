# Inkling 2 Small on Ampere (8x A100-80GB)

This branch (`inkling2-ampere`) runs `thinkingmachines/Inkling-2-Small` (the NVFP4 checkpoint, unmodified) on
**A100s: 4 nodes x 2 A100-80GB (8 GPUs, 640 GB), TP8 across nodes over InfiniBand**. It starts from
vllm-project/vllm PR #67325 (`d664cd1`, "[Model] Add Inkling 2 Small"), which is **Blackwell (SM100) only**,
and adds an SM80 path for every Blackwell-only kernel. The branch is upstream main `efe12305f0` + the PR + the
`[Inkling2]` commits; push to remote `jakehlee` (`jakehlee/vllm`, a fork of avtc/vllm, so GitHub's "ahead of
avtc:main" banner is expected).

One deployment of this branch (Slurm job scripts, front end, cluster specifics) lives in a separate repo,
`../ChatHPC` (branch `inkling`, see its `AGENTS.md`); nothing here depends on it.

## Ground rules

- **Do not touch the Blackwell path.** Every Ampere change sits behind a capability check (SM < 100) or a
  separate code path, so the diff stays reviewable against the PR and rebases cleanly when the PR changes.
- **No requantization.** The routed experts are the only quantized weights; vLLM's Marlin MoE kernels run NVFP4
  (`prepare_nvfp4_moe_layer_for_marlin`) and MXFP8 (`prepare_mxfp8_moe_layer_for_marlin`) weights as
  weight-only on SM80 with BF16 activations. Everything else in the checkpoint is BF16.
- **Every kernel replacement gets a test against the PR's torch reference** (`tests/models/inkling2/`),
  runnable on a single A100, before it goes into a full-model run. Full-model restarts cost 15+ minutes.

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

## Parallelism: TP8 (= EP8)

The PR shards experts over the **TP group** (`moe.py`: "EP size = TP size", reduced with
`tensor_model_parallel_all_reduce`), splits attention heads over TP (8 KV heads, so TP must divide 8), and
asserts PP == 1 (`model.py`). There is no DP x EP all-to-all path, so 8 A100s means **TP8 across 4 nodes**,
with two NCCL all-reduces per layer (96 per token) over InfiniBand. Consequences:
- 3 nodes (TP6) can't run, and 2 nodes (TP4, ~94 GiB of experts per GPU) can't hold the full model.
- Decode is bound by the cross-node all-reduces, not compute, so batching is cheap and DSpark is the big
  single-stream win.

## Memory at TP8 (per GPU)

| Per GPU | GiB |
|---|---|
| Experts: 48 of 384 (layers 0-39 at ~0.56 B/param, 40-47 at ~1.03 B/param) | ~47 |
| BF16 non-expert weights, sharded 8 ways (ViT, DSpark, and small tensors replicated) | ~3-5 |
| KV cache at `--gpu-memory-utilization 0.95`, `--max-num-batched-tokens 4096`, 128k context | the rest |

`--max-num-batched-tokens 8192` does **not** fit one 128k request (27.95 GiB of KV needed vs 22.5 available at
0.95): with async scheduling, each sliding-window layer reserves the window + 2 x the batched-token budget per
request. If memory gets tight: text-only (no ViT), then requantizing layers 40-47 from MXFP8 to NVFP4 offline
(~5.7 GB per GPU; mechanical, not a calibration).

## Blackwell-only code and its Ampere replacement

`vllm/models/inkling2/` is ~9.8k lines; the Ampere changes are ~1.5k (`git diff d664cd1 -- vllm tests`).

| Where | Blackwell dependency | Ampere replacement | Tests on A100 |
|---|---|---|---|
| `ops/attention.py`, `attention.py` | Prefill: vendored `tml_fa4` (`flash_fwd_sm100`). Decode/verify: Gluon kernels (TMEM, tcgen05, TMA) | `ops/attention_triton.py` (`rel_attention`): one Triton kernel for prefill, decode and verify, split-KV with a torch LSE merge. Reads the prologue's sheared rel logits directly, so `attention_prologue` and `gate_sink` stay shared. Picked off SM100 by `Inkling2Attention.use_triton`. Performance untuned | `test_attention_triton.py` 40/40 |
| `moe.py`, `ops/trtllm_bmm.py`, `ops/moe.py` | FlashInfer `trtllm_gen` SM100 BMMs, NVFP4/MXFP8 activation quantization fused into routing and SwiGLU, SM100 row shuffle / scale swizzle at load | `moe_marlin.py` (`Inkling2MarlinExperts`): raw load, Marlin repack in `process_weights_after_loading`, torch routing (stable sort, so ties go to the lower expert index), `fused_marlin_moe` with a custom softcapped-SwiGLU activation. NVFP4 global scale per expert = `amax / outer_scale_denominator` | `test_moe_marlin.py` 13/13, incl. real full-size layers 0 (NVFP4) and 40 (MXFP8) |
| `ops/*.py` Triton kernels (~25 sites in 8 files) | `gdc_wait` / `gdc_launch_dependents` / `launch_pdl=True` (programmatic dependent launch): ptxas rejects `griddepcontrol` below sm_90 | `ops/pdl.py`: constexpr-gated wrappers and a `PDL` flag, on for SM90+ only | compiles on sm_80 (exercised by the attention tests) |
| `ops/linear.py` | FlashInfer CuTe DSL `mm_bf16` (<=32 rows) and a Triton TMA kernel (router, >=2048 rows) | cuBLAS for every batch; the plain Triton K-group kernel for the router | full model |
| `quant_config.py` | `get_min_capability()` returned 100, so `vllm/config/vllm.py` rejects the checkpoint on Ampere before loading | Returns 80 | full model |
| `ops/allreduce.py`, `model.py` forward | NVLink multicast Lamport all-reduce; seq-parallel and fused conv-mix are gated on its workspace | `Inkling2AllReduceWorkspace.create` returns `None` off SM100, which also disables `seq_parallel` and `conv_mix`: the unfused NCCL path, no `_C.inkling2_*` ops | full model |
| `vision.py` | `tml_fa4.flash_attn_varlen_func` for the ViT attention (the other ViT kernels in `ops/vision.py` are plain Triton) | `_fa2_attention`: vLLM's FA2 varlen (`fa_version=2`, bidirectional, same `cu_seqlens` / `max_seqlen`), off SM100 | `test_vision.py` 3/3 |
| `dspark.py`, `ops/dspark.py` | FlashInfer `rmsnorm` / `fused_add_rmsnorm`, TRT-LLM fused all-reduce + norm, Gluon `block_decode_attention`, `_C.inkling2_dspark_argmax` over NVLink multicast | Off SM100 (`_SM100` in `dspark.py`): `ops.norm.rms_norm` + `_custom_ops.fused_add_rms_norm`; unfused NCCL all-reduce; `attention_triton.block_attention` (the Triton kernel's `DRAFT_BLOCK` mode); `gather_argmax` (per-shard argmax, all-gather of fp32 `(max, index)` pairs, lowest rank wins ties) | `test_dspark_ampere.py` 17/17 (the SM100 tests' cases, plus CUDA-graph replay) |
| `ops/{norm,attn_res,sconv,activation,vision}.py` | Plain Triton (`tldevice` intrinsics) | Only the PDL gate | full model |

Checkpoint layout confirmed on real weights: layer 0 NVFP4 `uint8` [4608, 2304] w13 with e4m3 scales, `amax`
~0.08-0.12, denominator 2688; layer 40 MXFP8 e4m3 [4608, 4608] w13, `uint8` scales.

## Image

Build on an **official vLLM CI nightly** (`vllm/vllm-openai:nightly-be47189f37f6162991366ff6b91539f6e9d1f432`,
4 Python-only commits before the PR's base, built for `TORCH_CUDA_ARCH_LIST="7.5 8.0 8.6 8.9 9.0 10.0 12.0"`)
and copy this branch's `vllm/**/*.py` (plus `vllm/models/inkling2/ops/*.cu`, which the SM100 path JIT-compiles)
over the installed package. The PR's C++ additions are SM100-only kernels, unreachable on Ampere, so the
nightly's compiled extensions match. **Not** `vllm/vllm-openai:inkling-2-small`: it is built for `10.0+PTX`
only, and its sm_80 Marlin coverage is partial (fp16 kernels run, bf16 ones don't).
`docker/Dockerfile.inkling2-ampere` does exactly this (`docker build -f docker/Dockerfile.inkling2-ampere
-t vllm-inkling2-ampere .` from the repo root; `apptainer build ... docker-daemon://...` for Apptainer).

## Tests (one A100)

```
python3 -m pytest -v tests/models/inkling2/test_moe_marlin.py tests/models/inkling2/test_attention_triton.py \
    tests/models/inkling2/test_dspark_ampere.py tests/models/inkling2/test_vision.py
```

73/73 pass. `test_moe_marlin.py`'s real-layer cases need the checkpoint in the HF cache.

## Serving (4 nodes x 2 A100)

The same `vllm serve` on every node; node rank 0 serves the API, the others add `--headless`:

```
vllm serve <checkpoint> --tensor-parallel-size 8 --nnodes 4 --node-rank <0-3> \
    --master-addr <rank-0 IP> --master-port 29560 --no-numa-bind \
    --max-model-len 131072 --max-num-batched-tokens 4096 --max-num-seqs 16 --gpu-memory-utilization 0.95 \
    --limit-mm-per-prompt.image 4 --limit-mm-per-prompt.audio 0 \
    --enable-chunked-prefill --enable-prefix-caching --async-scheduling \
    --speculative-config '{"method": "dspark", "num_speculative_tokens": 7, "draft_sample_method": "probabilistic"}' \
    --reasoning-parser inkling --tool-call-parser inkling --enable-auto-tool-choice \
    --default-chat-template-kwargs '{"reasoning_effort": 0.7}'      # rank 0 only
```

- `--no-numa-bind` on every node (per the vLLM recipe; headless nodes fail with "Executable was changed during
  NUMA binding context" otherwise). Set `VLLM_HOST_IP`, `GLOO_SOCKET_IFNAME` and `NCCL_SOCKET_IFNAME` to the
  InfiniBand interface on each node.
- Put the checkpoint on node-local SSD: vLLM reads shards one at a time, so from a network filesystem at
  ~225 MB/s per reader loading takes ~30 min (~20 s from local SSD).
- `VLLM_ENGINE_READY_TIMEOUT_S=3600`: the default 600 s is too short for a 4-node start.
- Point `TRITON_CACHE_DIR` and `VLLM_CACHE_ROOT` at local disk: several nodes compiling into a shared NFS cache
  hit `OSError [Errno 116] Stale file handle`.
- CUDA graphs: vLLM auto-enables `VLLM_USE_BREAKABLE_CUDAGRAPH` for Inkling 2. Don't set
  `cudagraph_mode: piecewise` (the same breakable path; without breakable graphs it's forced to NONE).
- Reasoning effort is continuous in [0, 0.99]; numeric values only via `chat_template_kwargs.reasoning_effort`
  (the top-level `reasoning_effort` field takes names only). At effort 0 the model tends not to call tools.

## Results (4 x 2 A100-80GB, NCCL NET/IB across nodes, GPUDirect RDMA off, 2026-11-15)

Single stream, one change at a time:

| Config | tok/s |
|---|---|
| `--enforce-eager` | ~12 |
| Breakable CUDA graphs (7 PIECEWISE + 5 FULL sizes) | 43.9 |
| + DSpark, 7 tokens, probabilistic | 118-180 on technical text, ~90 on prose |

DSpark acceptance 34-60% (mean acceptance length 3.4-5.2, position 1 at 0.71-0.90). Drafting and verification
cost ~20% per step (~34.5 steps/s vs ~44 without DSpark). Greedy drafting (`gather_argmax`) passes on a
truncated model but is untried on the full model.

Concurrency (graphs + DSpark, effort 0.7, 1024 max tokens, straight at the API server):

| Concurrent | 1 | 2 | 4 | 8 | 16 |
|---|---|---|---|---|---|
| Aggregate tok/s | 90 | 143 | 212 | 277 | 353 |
| Per-request tok/s | 91 | 76 | 56 | 37 | 23 |

TTFT p50 stayed <= 1.5 s.

Quality: the model was trained with NVFP4 *activations*; this path runs BF16 activations, so it is close but
not bit-identical. Checked so far: short factual and arithmetic answers, coherent long-form text, a correct
image description, DSpark acceptance in line with a well-matched target, and document Q&A via tool calls. Not
yet run: a formal eval (e.g. MMMU / HLE samples via `thinking-machines-lab/inkling-verify`).

## Known limits

- **Images are not resized**: one token per 40x40 px tile, and the encoder cache is
  `--max-num-batched-tokens`, so at 4096 an image over ~6.5 MP is rejected with a 400. Resize client-side.
- **Attention is untuned** Triton; prefill-heavy loads may want block-size tuning.
- **DSpark at TP8**: each greedy draft step all-gathers across nodes (7 per draft); probabilistic sampling
  all-gathers full-vocab logits per step. The measured overhead is small next to the target's 96 all-reduces.
- `_custom_ops.fused_add_rms_norm` may round the residual differently from FlashInfer's; that can only move
  DSpark acceptance, not the target's output.
