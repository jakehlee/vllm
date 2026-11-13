# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The Inkling2 DSpark draft's non-SM100 path against torch references: the
Triton block attention, the all-gather argmax merge, the vLLM RMSNorms, and
the portable draft kernels compiling and matching on this GPU.

Self-contained (test_dspark.py skips its whole module off SM100), so it also
runs as a lone file against an installed vLLM.
"""

import os
import tempfile
from types import SimpleNamespace

import pytest
import torch

from vllm.platforms import current_platform

if not current_platform.is_cuda() or current_platform.is_device_capability_family(100):
    pytest.skip("the non-SM100 DSpark path", allow_module_level=True)

from vllm.models.inkling2.ops.attention_triton import block_attention  # noqa: E402
from vllm.models.inkling2.ops.dspark import (  # noqa: E402
    add_softcap,
    markov_gate,
    merge_argmax_pairs,
    mix_rmsnorm,
    qk_norm_rope_,
    shard_argmax_pairs,
    silu_and_mul,
)

DEVICE = "cuda"


@pytest.fixture
def dist_init():
    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.distributed import (
        cleanup_dist_env_and_memory,
        init_distributed_environment,
        initialize_model_parallel,
    )

    fd, path = tempfile.mkstemp()
    os.close(fd)
    with set_current_vllm_config(VllmConfig()):
        init_distributed_environment(1, 0, f"file://{path}", 0, "nccl")
        initialize_model_parallel(1, 1)
        yield
    cleanup_dist_env_and_memory()
    if os.path.exists(path):
        os.remove(path)


@pytest.mark.parametrize("is_local", [True, False])
@pytest.mark.parametrize("num_splits", [1, 3, 200])
@pytest.mark.parametrize("num_heads,num_kv_heads", [(6, 2), (8, 1)])
def test_block_attention_anchored_window(is_local, num_splits, num_heads, num_kv_heads):
    """test_dspark.test_block_attention_anchored_window on the Triton kernel:
    every query of a block sees the whole block and, for local layers, the
    `window` positions before it; a padding request (no keys) comes out 0.
    200 splits is clamped (more splits than key blocks leaves some empty)."""
    g = torch.Generator().manual_seed(854)
    block, window, head_dim, page = 8, 64, 128, 16
    context_lens, num_pages = [5, 70, 150, 300, 1100], 80

    def rand(*shape):
        return torch.randn(*shape, generator=g).bfloat16().to(DEVICE)

    num_reqs = len(context_lens) + 1
    # The model's [blocks, block_size, kv_heads, (K | V)] cache layout.
    kv = rand(num_reqs * num_pages, page, num_kv_heads, 2 * head_dim)
    key_cache, value_cache = kv.split(head_dim, dim=-1)
    # Shuffled pages: the kernel must go through the block table.
    perm = torch.randperm(num_reqs * num_pages, generator=g)
    block_table = perm.int().to(DEVICE).view(num_reqs, num_pages)
    seq_lens = [n + block for n in context_lens] + [0]
    seq_lens = torch.tensor(seq_lens, dtype=torch.int32, device=DEVICE)
    q = rand(block * num_reqs, num_heads, head_dim)
    out = torch.full((q.shape[0], num_heads * head_dim), float("nan"), device=DEVICE)
    out = out.bfloat16()
    block_attention(
        q,
        key_cache,
        value_cache,
        out,
        block_table=block_table,
        cache_seqlens=seq_lens,
        cu_seqlens_q=torch.arange(num_reqs + 1, device=DEVICE).int() * block,
        block=block,
        window=window + block if is_local else 0,
        num_splits=num_splits,
    )
    out = out.view(-1, num_heads, head_dim)

    group = num_heads // num_kv_heads
    for i, n in enumerate(context_lens):
        start = max(0, n - window) if is_local else 0
        pages = block_table[i].long()
        keys = key_cache[pages].flatten(0, 1)[start : n + block]
        vals = value_cache[pages].flatten(0, 1)[start : n + block]
        keys = keys.float().repeat_interleave(group, 1)
        vals = vals.float().repeat_interleave(group, 1)
        rows = slice(i * block, (i + 1) * block)
        scores = torch.einsum("qhd,khd->hqk", q[rows].float(), keys) * head_dim**-0.5
        ref = torch.einsum("hqk,khd->qhd", scores.softmax(-1), vals)
        torch.testing.assert_close(out[rows].float(), ref, atol=2e-2, rtol=2e-2)
    assert (out[len(context_lens) * block :] == 0).all()


@torch.inference_mode()
def test_block_attention_cuda_graph_replay():
    """Captured once, replayed with longer KV lengths (the draft's FULL
    graph): the same output as eager."""
    g = torch.Generator().manual_seed(855)
    block, heads, kv_heads, head_dim, page, pages = 8, 6, 1, 128, 16, 40
    kv = torch.randn(2 * pages, page, kv_heads, 2 * head_dim, generator=g)
    key_cache, value_cache = kv.bfloat16().to(DEVICE).split(head_dim, dim=-1)
    block_table = torch.arange(2 * pages, dtype=torch.int32, device=DEVICE)
    block_table = block_table.view(2, pages)
    seq_lens = torch.tensor([20, 30], dtype=torch.int32, device=DEVICE)
    q = torch.randn(2 * block, heads, head_dim, generator=g).bfloat16().to(DEVICE)
    cu = torch.arange(3, device=DEVICE).int() * block
    out = torch.empty(2 * block, heads * head_dim, dtype=torch.bfloat16, device=DEVICE)

    def run(dst):
        block_attention(
            q,
            key_cache,
            value_cache,
            dst,
            block_table=block_table,
            cache_seqlens=seq_lens,
            cu_seqlens_q=cu,
            block=block,
            window=0,
            num_splits=4,
        )

    run(out)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run(out)
    for lens in ([100, 37], [600, 9], [640, 640]):
        seq_lens.copy_(torch.tensor(lens))
        graph.replay()
        eager = torch.empty_like(out)
        run(eager)
        assert torch.equal(out, eager)


def test_gather_argmax_matches_full_vocab():
    """Per-shard (max, index) pairs merged rank-major equal the first argmax
    of the full softcapped logits: ties across and within shards, padding
    columns past `valid` holding the largest value, NaN, -0.0 maxima."""
    g = torch.Generator().manual_seed(853)
    world, cols, vocab, cap = 4, 50256, 200058, 30.0
    rows = 9
    base = torch.randn(rows, world * cols, generator=g).bfloat16().to(DEVICE)
    bias = torch.randn(rows, world * cols, generator=g).bfloat16().to(DEVICE)
    planted = [5, cols, cols + 3, cols + 900, 2 * cols + 1, 3 * cols - 1]
    base[:3, planted], bias[:3, planted] = 90.0, 0.0
    base[0, [5, cols + 3, cols + 900, 3 * cols - 1]] = 0.0  # Across shards.
    base[1, [5, cols, 2 * cols + 1, 3 * cols - 1]] = 0.0  # Within a shard.
    base[3, vocab:] = 90.0  # Padding.
    base[4], bias[4] = 1.0, 0.0  # All tied.
    base[5], bias[5] = -0.0, -0.0
    base[6], bias[6] = -1.0, 0.0
    base[6, 2 * cols + 7], bias[6, 2 * cols + 7] = -0.0, -0.0
    base[7] = float("nan")  # All NaN: index 0.
    base[8, : 3 * cols] = float("nan")

    valid = [min(cols, vocab - r * cols) for r in range(world)]
    pairs = torch.cat(
        [
            shard_argmax_pairs(
                base[:, r * cols : (r + 1) * cols].contiguous(),
                bias[:, r * cols : (r + 1) * cols].contiguous(),
                cap,
                valid[r],
                r * cols,
            )
            for r in range(world)
        ]
    )
    out = merge_argmax_pairs(pairs, rows)

    logits = add_softcap(base, bias, cap).float()[:, :vocab]
    logits[logits.isnan()] = float("-inf")
    expected = logits.argmax(-1)
    assert out.tolist() == expected.tolist()
    assert expected[:3].tolist() == [cols, cols + 3, 5]


def test_rms_norms_match_torch(dist_init):
    """The draft's `_RMSNorm` off SM100: plain and fused-add (in place)."""
    from vllm.models.inkling2.dspark import _RMSNorm

    g = torch.Generator().manual_seed(856)
    norm = _RMSNorm(4608, 1e-6).to(DEVICE).bfloat16()
    norm.weight.data = (0.5 + torch.rand(4608, generator=g)).bfloat16().to(DEVICE)
    x = torch.randn(17, 4608, generator=g).bfloat16().to(DEVICE)
    residual = torch.randn(17, 4608, generator=g).bfloat16().to(DEVICE)

    def ref(t):
        t = t.float()
        t = t * torch.rsqrt(t.square().mean(-1, keepdim=True) + 1e-6)
        return (t * norm.weight.float()).bfloat16()

    torch.testing.assert_close(norm(x), ref(x), atol=1e-2, rtol=1e-2)
    total = (x.float() + residual.float()).bfloat16()
    out = norm(x.clone(), residual)
    torch.testing.assert_close(residual, total, atol=1e-2, rtol=1e-2)
    torch.testing.assert_close(out, ref(total), atol=2e-2, rtol=2e-2)


def test_portable_draft_kernels():
    """The draft's Triton kernels, which the SM100 path shares, compile on
    this GPU and match torch."""
    g = torch.Generator().manual_seed(857)

    def rand(*shape, scale=1.0):
        return (torch.randn(*shape, generator=g) * scale).bfloat16().to(DEVICE)

    features = rand(17, 8, 4608)
    alpha = torch.randn(8, generator=g).softmax(-1).bfloat16().to(DEVICE)
    weight = (0.5 + torch.rand(4608, generator=g)).bfloat16().to(DEVICE)
    mixed = (features.float() * alpha.float()[:, None]).sum(1)
    ref = mixed * torch.rsqrt(mixed.square().mean(-1, keepdim=True) + 1e-6)
    out = mix_rmsnorm(features, alpha, weight, 1e-6)
    torch.testing.assert_close(
        out, (ref * weight.float()).bfloat16(), rtol=1e-2, atol=0
    )

    x = rand(5, 2 * 3072)
    gate, up = x.float().chunk(2, -1)
    ref = (torch.nn.functional.silu(gate) * up).bfloat16()
    torch.testing.assert_close(silu_and_mul(x), ref, atol=1e-2, rtol=1e-2)

    base, bias = rand(3, 5000, scale=20.0), rand(3, 5000)
    ref = ((base + bias).float() / 30.0).tanh().mul(30.0).bfloat16()
    torch.testing.assert_close(add_softcap(base, bias, 30.0), ref, atol=0, rtol=1e-2)

    raw = torch.randn(64, 3 * 32, generator=g).mul(4).to(DEVICE)
    state = rand(64, 32)
    gate_raw, cand_raw, out_raw = raw.bfloat16().chunk(3, dim=-1)
    gate = torch.sigmoid(gate_raw)
    new_state, latent = markov_gate(raw, state)
    assert torch.equal(new_state, gate * state + (1.0 - gate) * torch.tanh(cand_raw))
    assert torch.equal(latent, torch.tanh(out_raw))

    # qk_norm_rope_: per-head RMSNorm (rounded) then neox RoPE, in place.
    heads_q, heads_k, d = 4, 1, 128
    qkv = rand(6, (heads_q + 2 * heads_k) * d)
    positions = torch.tensor([0, 1, 7, 100, 1023, 5], device=DEVICE)
    inv = 10000.0 ** (-torch.arange(0, d, 2, dtype=torch.float32) / d)
    angles = torch.arange(2048, dtype=torch.float32)[:, None] * inv
    cos_sin = torch.cat([angles.cos(), angles.sin()], -1).to(DEVICE)
    q_w, k_w = rand(d, scale=0.1) + 1, rand(d, scale=0.1) + 1
    heads = qkv[:, : (heads_q + heads_k) * d].view(6, -1, d).float()
    w = torch.cat([q_w[None].expand(heads_q, d), k_w[None].expand(heads_k, d)])
    normed = heads * torch.rsqrt(heads.square().mean(-1, keepdim=True) + 1e-6)
    normed = (normed * w.float()).bfloat16().float()
    x1, x2 = normed.chunk(2, -1)
    cos, sin = cos_sin[positions, None, : d // 2], cos_sin[positions, None, d // 2 :]
    ref = torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], -1).bfloat16()
    v_before = qkv[:, (heads_q + heads_k) * d :].clone()
    qk_norm_rope_(qkv, positions, q_w, k_w, cos_sin, heads_q, heads_k, 1e-6)
    out = qkv[:, : (heads_q + heads_k) * d].view(6, -1, d)
    torch.testing.assert_close(out, ref, atol=2e-2, rtol=2e-2)
    assert torch.equal(qkv[:, (heads_q + heads_k) * d :], v_before)


def test_markov_head_step(dist_init):
    """`Inkling2MarkovHead.context` / `step` (fp32 `addmm(out_dtype=...)`)
    against an fp32 torch recurrence."""
    from vllm.models.inkling2.dspark import Inkling2MarkovHead

    g = torch.Generator().manual_seed(858)
    width, rank, vocab, reqs, steps = 256, 32, 512, 3, 7

    def normal(*shape, scale=1.0):
        return (torch.randn(*shape, generator=g) * scale).bfloat16().to(DEVICE)

    config = SimpleNamespace(
        hidden_size=width, markov_rank=rank, padded_vocab_size=vocab, rms_norm_eps=1e-6
    )
    with torch.device(DEVICE):
        head = Inkling2MarkovHead(config).bfloat16()
    head.markov_w1.weight.data = normal(vocab, rank, scale=0.3)
    head.rnn_proj.weight.data = normal(3 * rank, 2 * rank + 2 * width, scale=0.05)
    head.rnn_proj.bias.data = normal(3 * rank, scale=0.05)
    head.hidden_out.weight.data = normal(width, rank, scale=0.03)
    head.hidden_out.bias.data = normal(width, scale=0.03)
    lm_head = normal(vocab, width, scale=0.03)
    head.fold(lm_head)
    head.rnn_bias = head.rnn_proj.bias.float()

    hidden = normal(reqs * steps, width)
    f_rnn = normal(reqs, width)
    prev = torch.randint(0, vocab, (reqs,), generator=g).to(DEVICE)
    context = head.context(hidden, f_rnn)
    assert context.shape == (steps, reqs, 3 * rank) and context.dtype == torch.float32
    state = torch.zeros(reqs, rank, dtype=torch.bfloat16, device=DEVICE)
    state, bias = head.step(state, prev, context[0])

    w = head.rnn_proj.weight.float()
    h0 = hidden.view(reqs, steps, width)[:, 0].float()
    inputs = torch.cat(
        [
            torch.zeros(reqs, rank, device=DEVICE),
            head.markov_w1(prev).float(),
            h0,
            f_rnn.float(),
        ],
        -1,
    )
    raw = inputs @ w.T + head.rnn_bias
    torch.testing.assert_close(
        context[0] + inputs[:, : 2 * rank] @ w[:, : 2 * rank].T,
        raw,
        atol=2e-2,
        rtol=2e-2,
    )
    gate_raw, cand_raw, out_raw = raw.chunk(3, -1)
    gate = torch.sigmoid(gate_raw)
    ref_state = (1 - gate) * torch.tanh(cand_raw)
    torch.testing.assert_close(state.float(), ref_state, atol=3e-2, rtol=3e-2)
    latent = torch.tanh(out_raw)
    ref_bias = latent @ head.fold_weight.float().T + head.fold_bias.float()
    torch.testing.assert_close(bias.float(), ref_bias, atol=3e-2, rtol=3e-2)
