# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The portable Triton relative attention (non-SM100) against the fp32
reference of test_attention.py, on the sheared logits the prologue writes.

The reference helpers are copied rather than imported: test_attention.py
imports the Inkling 1 FA4 module, which needs SM100 toolchains.
"""

import pytest
import torch

from vllm.platforms import current_platform

if not current_platform.is_cuda():
    pytest.skip("needs CUDA", allow_module_level=True)

from vllm.models.inkling2.ops.attention_triton import rel_attention  # noqa: E402

HEAD_DIM = 128
REL_EXTENT = 1024
SCENARIOS = {
    "prefill": [(1500, 1500), (37, 37), (300, 300)],
    "chunked_prefill": [(200, 1700), (64, 64), (513, 2100)],
    "decode": [(1, 1), (1, 900), (1, 3000), (1, 1025)],
    "verify": [(8, 8), (8, 1500), (3, 2100)],
    "mixed": [(1, 900), (4, 3000), (300, 300), (200, 1700)],
}


def shear_rel_logits(rel_logits, seqs, tile, is_local):
    """Copy of test_attention.shear_rel_logits."""
    tokens, heads, extent = rel_logits.shape
    out = rel_logits.new_full((tokens + 128, heads, extent + 256), float("nan"))
    cols = torch.arange(extent + 256, device=rel_logits.device)
    start = 0
    for q_len, kv_len in seqs:
        idx = torch.arange(q_len, device=rel_logits.device)
        pos = kv_len - q_len + idx
        tile_end = ((idx // tile + 1) * tile).clamp(max=q_len)
        n_max = (kv_len - q_len + tile_end + 127) // 128
        dist = pos[:, None] - (cols[None] + (n_max[:, None] - 10) * 128)
        valid = (dist >= 0) & (dist < extent)
        rel = rel_logits[start : start + q_len]
        vals = rel.gather(2, dist.clamp(0, extent - 1)[:, None].expand(-1, heads, -1))
        pad = torch.where((dist < 0) | is_local, float("-inf"), 0.0).to(
            rel_logits.dtype
        )
        out[start : start + q_len] = torch.where(valid[:, None], vals, pad[:, None])
        start += q_len
    return out


def _ref_attention(
    q, key_cache, value_cache, rel_logits, seqs, block_table, scale, window_left
):
    """Copy of test_attention._ref_attention."""
    num_heads = q.shape[1]
    num_kv_heads = key_cache.shape[2]
    out = torch.empty(q.shape, dtype=torch.float32, device=q.device)
    lse = torch.empty(q.shape[:2], dtype=torch.float32, device=q.device)
    start = 0
    for i, (q_len, kv_len) in enumerate(seqs):
        blocks = block_table[i].long()
        k, v = (
            cache[blocks]
            .flatten(0, 1)[:kv_len]
            .float()
            .repeat_interleave(num_heads // num_kv_heads, dim=1)
            for cache in (key_cache, value_cache)
        )
        qpos = torch.arange(kv_len - q_len, kv_len, device=q.device)
        dist = qpos[:, None] - torch.arange(kv_len, device=q.device)[None]
        rel = rel_logits[start : start + q_len].float().transpose(0, 1)
        idx = dist.clamp(0, REL_EXTENT - 1).expand(num_heads, -1, -1)
        bias = rel.gather(2, idx).masked_fill(~((dist >= 0) & (dist < REL_EXTENT)), 0)
        qk = torch.einsum("qhd,khd->hqk", q[start : start + q_len].float(), k)
        scores = (qk + bias / scale) * scale
        mask = dist < 0
        if window_left is not None:
            mask |= dist > window_left
        scores.masked_fill_(mask, float("-inf"))
        lse[start : start + q_len] = scores.logsumexp(-1).T
        out[start : start + q_len] = torch.einsum("hqk,khd->qhd", scores.softmax(-1), v)
        start += q_len
    return out, lse


@pytest.mark.parametrize(
    "num_heads,num_kv_heads,block_size,window_left",
    [
        (6, 1, 16, None),  # TP8 shard of a full layer
        (6, 1, 16, REL_EXTENT - 1),  # TP8 shard of a window layer
        (12, 2, 128, None),
        (48, 8, 16, REL_EXTENT - 1),
    ],
)
@pytest.mark.parametrize("scenario", list(SCENARIOS))
@pytest.mark.parametrize("padding", [0, 5])
@torch.inference_mode()
def test_triton_rel_attention(
    num_heads, num_kv_heads, block_size, window_left, scenario, padding
):
    torch.manual_seed(0)
    device = "cuda"
    seqs = SCENARIOS[scenario]
    q_lens = [q_len for q_len, _ in seqs]
    num_tokens = sum(q_lens)
    max_blocks = -(-max(kv_len for _, kv_len in seqs) // block_size)
    num_blocks = len(seqs) * max_blocks + 1

    # The model's [blocks, block_size, kv_heads, (K | V)] cache, split like
    # Inkling2Attention._split_kv_cache.
    kv = torch.randn(num_blocks, block_size, num_kv_heads, 2 * HEAD_DIM, device=device)
    kv[..., :HEAD_DIM] = torch.nn.functional.normalize(kv[..., :HEAD_DIM], dim=-1) * 2
    key_cache, value_cache = kv.bfloat16().split(HEAD_DIM, dim=-1)
    block_table = (
        (torch.randperm(num_blocks - 1, device=device) + 1)[: len(seqs) * max_blocks]
        .view(len(seqs), max_blocks)
        .int()
    )
    # CUDA-graph padding rows past the real tokens hold garbage.
    q = torch.randn(num_tokens + padding, num_heads, HEAD_DIM, device=device)
    q = (torch.nn.functional.normalize(q, dim=-1) * 3).bfloat16()
    rel_logits = torch.randn(
        num_tokens, num_heads, REL_EXTENT, device=device, dtype=torch.bfloat16
    )
    rel_in = shear_rel_logits(rel_logits, seqs, 128, window_left is not None)
    rel_in = torch.cat([rel_in, rel_in.new_full((padding, *rel_in.shape[1:]), 7.0)])
    cu_seqlens_q = torch.tensor(
        [0, *torch.tensor(q_lens).cumsum(0).tolist()], dtype=torch.int32, device=device
    )
    seq_lens = torch.tensor([kv_len for _, kv_len in seqs], dtype=torch.int32, device=device)
    scale = HEAD_DIM**-0.5

    out = torch.full(q.shape, float("nan"), device=device)
    # The model passes the transpose of an (H, T) buffer.
    lse = torch.full(q.shape[1::-1], float("nan"), device=device).t()
    rel_attention(
        q,
        key_cache,
        value_cache,
        rel_in,
        block_table=block_table,
        seq_lens=seq_lens,
        cu_seqlens_q=cu_seqlens_q,
        max_query_len=max(q_lens),
        max_seq_len=max(kv for _, kv in seqs),
        softmax_scale=scale,
        window_left=0 if window_left is None else window_left,
        out=out,
        lse=lse,
    )
    ref_out, ref_lse = _ref_attention(
        q[:num_tokens], key_cache, value_cache, rel_logits, seqs, block_table, scale,
        window_left,
    )
    torch.testing.assert_close(out[:num_tokens], ref_out, atol=1e-2, rtol=1e-2)
    torch.testing.assert_close(lse[:num_tokens], ref_lse, atol=1e-3, rtol=1e-4)
    assert (out[num_tokens:] == 0).all() and (lse[num_tokens:] == 0).all()
