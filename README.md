<!-- markdownlint-disable MD001 MD041 -->
> [!IMPORTANT]
> **This is the `inkling2-ampere` branch of a personal fork, not upstream vLLM.** It runs
> [Inkling 2 Small](https://huggingface.co/thinkingmachines/Inkling-2-Small) on **Ampere (A100) GPUs**, built on
> [vllm-project/vllm#67325](https://github.com/vllm-project/vllm/pull/67325), whose kernels are Blackwell (SM100)
> only. Not intended for upstream merge as-is. The upstream vLLM README follows the section below.

## Inkling 2 Small on 8x A100-80GB

The released NVFP4/MXFP8 checkpoint is served **unmodified**: the routed experts run as weight-only Marlin GEMMs
with BF16 activations, and every other Blackwell-only kernel (attention, DSpark drafting, the ViT, fused
all-reduces, PDL) has an SM80 fallback selected by GPU capability. The Blackwell code paths are untouched.

- **Hardware:** 8 A100-80GB GPUs (tested as 4 nodes x 2 GPUs over InfiniBand), **TP8**. The model's experts are
  sharded over the TP group and it asserts PP = 1, so it needs exactly 8 GPUs: TP must divide the 8 KV heads, and
  4 GPUs (TP4) can't hold the full model. A single 8-GPU node or 2 x 4 should work the same way but is untested.
- **Speed (4 x 2 A100, InfiniBand):** ~12 tok/s eager, 44 tok/s with CUDA graphs, **90-180 tok/s with DSpark**
  speculative decoding, single stream; 353 tok/s aggregate at 16 concurrent requests (23 tok/s each).
- **Context:** 128k, text + images (images aren't resized: about 6.5 MP max at the settings below).
- **Numerics:** the model was trained with NVFP4 activations; this path runs BF16 activations. Answers, image
  descriptions and DSpark acceptance (34-60%) look right, but no formal eval has been run yet.

**Image:** a vLLM CI nightly plus this branch's Python files; nothing is compiled
([`docker/Dockerfile.inkling2-ampere`](docker/Dockerfile.inkling2-ampere)):

```bash
docker build -f docker/Dockerfile.inkling2-ampere -t vllm-inkling2-ampere .
```

**Serve:** the same command on every node, rank 0 first; ranks 1-3 replace the last line with `--headless`.

```bash
vllm serve thinkingmachines/Inkling-2-Small --tensor-parallel-size 8 --nnodes 4 --node-rank <0-3> \
    --master-addr <rank-0 IP> --master-port 29560 --no-numa-bind \
    --max-model-len 131072 --max-num-batched-tokens 4096 --max-num-seqs 16 --gpu-memory-utilization 0.95 \
    --limit-mm-per-prompt.image 4 --limit-mm-per-prompt.audio 0 \
    --enable-chunked-prefill --enable-prefix-caching --async-scheduling \
    --speculative-config '{"method": "dspark", "num_speculative_tokens": 7, "draft_sample_method": "probabilistic"}' \
    --reasoning-parser inkling --tool-call-parser inkling --enable-auto-tool-choice
```

Put the 393.5 GB checkpoint on node-local SSD, set `VLLM_ENGINE_READY_TIMEOUT_S=3600`, and point
`NCCL_SOCKET_IFNAME` / `GLOO_SOCKET_IFNAME` / `VLLM_HOST_IP` at the InfiniBand interface.
**[`CLAUDE.md`](CLAUDE.md)** has the details: each kernel replacement and its tests (73 pass on one A100), the
memory budget, deployment notes, and the full results.

---

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/vllm-project/vllm/main/docs/assets/logos/vllm-logo-text-dark.png">
    <img alt="vLLM" src="https://raw.githubusercontent.com/vllm-project/vllm/main/docs/assets/logos/vllm-logo-text-light.png" width=55%>
  </picture>
</p>

<h3 align="center">
Easy, fast, and cheap LLM serving for everyone
</h3>

<p align="center">
| <a href="https://docs.vllm.ai"><b>Documentation</b></a> | <a href="https://blog.vllm.ai/"><b>Blog</b></a> | <a href="https://arxiv.org/abs/2309.06180"><b>Paper</b></a> | <a href="https://x.com/vllm_project"><b>Twitter/X</b></a> | <a href="https://discuss.vllm.ai"><b>User Forum</b></a> | <a href="https://slack.vllm.ai"><b>Developer Slack</b></a> |
</p>

🔥 We have built a vLLM website to help you get started with vLLM. Please visit [vllm.ai](https://vllm.ai) to learn more.
For events, please visit [vllm.ai/events](https://vllm.ai/events) to join us.

---

## About

vLLM is a fast and easy-to-use library for LLM inference and serving.

Originally developed in the [Sky Computing Lab](https://sky.cs.berkeley.edu) at UC Berkeley, vLLM has grown into one of the most active open-source AI projects built and maintained by a diverse community of many dozens of academic institutions and companies from over 2000 contributors.

vLLM is fast with:

- State-of-the-art serving throughput
- Efficient management of attention key and value memory with [**PagedAttention**](https://blog.vllm.ai/2023/06/20/vllm.html)
- Continuous batching of incoming requests, chunked prefill, prefix caching
- Fast and flexible model execution with piecewise and full CUDA/HIP graphs
- Quantization: FP8, MXFP8/MXFP4, NVFP4, INT8, INT4, GPTQ/AWQ, GGUF, compressed-tensors, ModelOpt, TorchAO, and [more](https://docs.vllm.ai/en/latest/features/quantization/index.html)
- Optimized attention kernels including FlashAttention, FlashInfer, TRTLLM-GEN, FlashMLA, and Triton
- Optimized GEMM/MoE kernels for various precisions using CUTLASS, TRTLLM-GEN, CuTeDSL
- Speculative decoding including n-gram, suffix, EAGLE, DFlash
- Automatic kernel generation and graph-level transformations using torch.compile
- Disaggregated prefill, decode, and encode

vLLM is flexible and easy to use with:

- Seamless integration with popular Hugging Face models
- High-throughput serving with various decoding algorithms, including *parallel sampling*, *beam search*, and more
- Tensor, pipeline, data, expert, and context parallelism for distributed inference
- Streaming outputs
- Generation of structured outputs using xgrammar or guidance
- Tool calling and reasoning parsers
- OpenAI-compatible API server, plus Anthropic Messages API and gRPC support
- Efficient multi-LoRA support for dense and MoE layers
- Support for NVIDIA GPUs, AMD GPUs, Intel GPUs, and x86/ARM/PowerPC CPUs. Additionally, diverse hardware plugins such as Google TPUs, Intel Gaudi, IBM Spyre, Huawei Ascend, Rebellions NPU, Apple Silicon, MetaX GPU, Iluvatar CoreX GPU, and more.

vLLM seamlessly supports 200+ model architectures on Hugging Face, including:

- Decoder-only LLMs (e.g., Llama, Qwen, Gemma)
- Mixture-of-Expert LLMs (e.g., Mixtral, DeepSeek-V3, Qwen-MoE, GPT-OSS)
- Hybrid attention and state-space models (e.g., Mamba, Qwen3.5)
- Multi-modal models (e.g., LLaVA, Qwen-VL, Pixtral)
- Embedding and retrieval models (e.g., E5-Mistral, GTE, ColBERT)
- Reward and classification models (e.g., Qwen-Math)

Find the full list of supported models [here](https://docs.vllm.ai/en/latest/models/supported_models.html).

## Getting Started

Install vLLM with [`uv`](https://docs.astral.sh/uv/) (recommended) or `pip`:

```bash
uv pip install vllm
vllm download-kernels  # precompiled kernels for faster startup
```

Or [build from source](https://docs.vllm.ai/en/latest/getting_started/installation/gpu/index.html#build-wheel-from-source) for development.

Visit our [documentation](https://docs.vllm.ai/en/latest/) to learn more.

- [Installation](https://docs.vllm.ai/en/latest/getting_started/installation.html)
- [Quickstart](https://docs.vllm.ai/en/latest/getting_started/quickstart.html)
- [List of Supported Models](https://docs.vllm.ai/en/latest/models/supported_models.html)

## Contributing

We welcome and value any contributions and collaborations.
Please check out [Contributing to vLLM](https://docs.vllm.ai/en/latest/contributing/index.html) for how to get involved.

## Citation

If you use vLLM for your research, please cite our [paper](https://arxiv.org/abs/2309.06180):

```bibtex
@inproceedings{kwon2023efficient,
  title={Efficient Memory Management for Large Language Model Serving with PagedAttention},
  author={Woosuk Kwon and Zhuohan Li and Siyuan Zhuang and Ying Sheng and Lianmin Zheng and Cody Hao Yu and Joseph E. Gonzalez and Hao Zhang and Ion Stoica},
  booktitle={Proceedings of the ACM SIGOPS 29th Symposium on Operating Systems Principles},
  year={2023}
}
```

## Contact Us

<!-- --8<-- [start:contact-us] -->
- For technical questions and feature requests, please use GitHub [Issues](https://github.com/vllm-project/vllm/issues)
- For discussing with fellow users, please use the [vLLM Forum](https://discuss.vllm.ai)
- For coordinating contributions and development, please use [Slack](https://slack.vllm.ai)
- For security disclosures, please use GitHub's [Security Advisories](https://github.com/vllm-project/vllm/security/advisories) feature
- For collaborations and partnerships, please contact us at [collaboration@vllm.ai](mailto:collaboration@vllm.ai)
<!-- --8<-- [end:contact-us] -->

## Media Kit

- If you wish to use vLLM's logo, please refer to [our media kit repo](https://github.com/vllm-project/media-kit)
