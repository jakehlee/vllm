# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Inkling2 relative attention for GPUs without tml_fa4 / the Gluon decode
kernels (Ampere, Hopper): one portable Triton kernel for prefill, decode and
verify tokens alike, split-KV for long decodes.

It reads the relative logits in the sheared layout `attention_prologue`
writes for tml_fa4, so the prologue is shared with the SM100 path: token `t`
at query index `i` of a request with `q_len` queries and `kv_len` keys has
`n_max = ceil((kv_len - q_len + min((i // 128 + 1) * 128, q_len)) / 128)`, and
column `c` of its `(E + 256)`-wide row holds key `c + n_max * 128 - (E + 256)`.
Keys left of the columns are past the extent (bias 0); keys right of them
are causally masked anyway. Scores are scale-after-add, as in tml_fa4:

    score(i, j) = q_i . k_j * scale + rel(i, j)

and the fp32 output and natural-log LSE feed `gate_sink` like tml_fa4's.
"""

import functools

import torch

from vllm.triton_utils import tl, triton

@functools.cache
def _num_sms(device_index: int) -> int:
    return torch.cuda.get_device_properties(device_index).multi_processor_count


@triton.jit
def _rel_attention_kernel(
    Q,  # [T, H, D] bf16 (clipped)
    K_CACHE,  # [num_blocks, page, HKV, D] (strided)
    V_CACHE,
    REL,  # [T + 128, H, E + 256] sheared relative logits
    BLOCK_TABLE,
    SEQ_LENS,  # [R] keys per request, including this step's
    CU_SEQLENS_Q,  # [R + 1]
    O_PART,  # [S, T, H, D] fp32
    LSE_PART,  # [S, T, H] fp32 natural-log LSE (-inf: no keys)
    scale,
    stride_qt,
    stride_qh,
    stride_kb,
    stride_kn,
    stride_kh,
    stride_vb,
    stride_vn,
    stride_vh,
    stride_rt,
    stride_rh,
    stride_bt,
    T,
    H: tl.constexpr,
    GROUP: tl.constexpr,
    D: tl.constexpr,
    E: tl.constexpr,
    PAGE: tl.constexpr,
    WINDOW_LEFT: tl.constexpr,  # 0: full attention
    NUM_SPLITS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    req = tl.program_id(0)
    head = tl.program_id(1)
    m_block = tl.program_id(2) // NUM_SPLITS
    split = tl.program_id(2) % NUM_SPLITS
    kv_head = head // GROUP
    E_PAD: tl.constexpr = E + 256

    q_start = tl.load(CU_SEQLENS_Q + req)
    q_len = tl.load(CU_SEQLENS_Q + req + 1) - q_start
    if m_block * BLOCK_M >= q_len:
        return
    kv_len = tl.load(SEQ_LENS + req)
    past = kv_len - q_len

    offs_m = m_block * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, D)
    offs_n = tl.arange(0, BLOCK_N)
    m_mask = offs_m < q_len
    pos = past + offs_m  # absolute position of each query
    tok = (q_start + offs_m).to(tl.int64)

    q = tl.load(
        Q + tok[:, None] * stride_qt + head * stride_qh + offs_d[None, :],
        m_mask[:, None],
        other=0.0,
    )
    # First key of each query's sheared row (see the module docstring).
    tile_end = tl.minimum((offs_m // 128 + 1) * 128, q_len)
    col0 = tl.cdiv(past + tile_end, 128) * 128 - E_PAD
    rel_row = REL + tok * stride_rt + head * stride_rh

    # Keys this query block can see, split into NUM_SPLITS contiguous chunks.
    last_pos = past + tl.minimum((m_block + 1) * BLOCK_M, q_len) - 1
    lo = 0
    if WINDOW_LEFT > 0:
        lo = tl.maximum(past + m_block * BLOCK_M - WINDOW_LEFT, 0)
    lo = lo // BLOCK_N * BLOCK_N
    num_n = tl.cdiv(last_pos + 1 - lo, BLOCK_N)
    per_split = tl.cdiv(num_n, NUM_SPLITS)
    n_begin = lo + split * per_split * BLOCK_N
    n_end = tl.minimum(n_begin + per_split * BLOCK_N, last_pos + 1)

    m_i = tl.full([BLOCK_M], float("-inf"), tl.float32)
    l_i = tl.zeros([BLOCK_M], tl.float32)
    acc = tl.zeros([BLOCK_M, D], tl.float32)
    # Scores in log2 units: exp2 instead of exp.
    qk_scale = scale * 1.4426950408889634
    for n0 in range(n_begin, n_end, BLOCK_N):
        k_pos = n0 + offs_n
        n_mask = k_pos < n_end
        page = tl.load(
            BLOCK_TABLE + req * stride_bt + k_pos // PAGE, n_mask, other=0
        ).to(tl.int64)
        slot = k_pos % PAGE
        k = tl.load(
            K_CACHE
            + page[:, None] * stride_kb
            + slot[:, None] * stride_kn
            + kv_head * stride_kh
            + offs_d[None, :],
            n_mask[:, None],
            other=0.0,
        )
        v = tl.load(
            V_CACHE
            + page[:, None] * stride_vb
            + slot[:, None] * stride_vn
            + kv_head * stride_vh
            + offs_d[None, :],
            n_mask[:, None],
            other=0.0,
        )
        s = tl.dot(q, tl.trans(k)) * qk_scale

        col = k_pos[None, :] - col0[:, None]
        in_cols = (col >= 0) & (col < E_PAD) & m_mask[:, None] & n_mask[None, :]
        rel = tl.load(rel_row[:, None] + col, in_cols, other=0.0).to(tl.float32)
        dist = pos[:, None] - k_pos[None, :]
        visible = (dist >= 0) & n_mask[None, :] & m_mask[:, None]
        if WINDOW_LEFT > 0:
            visible = visible & (dist <= WINDOW_LEFT)
        s = tl.where(visible, s + rel * 1.4426950408889634, float("-inf"))

        m_new = tl.maximum(m_i, tl.max(s, 1))
        # Rows with no visible key yet keep m = -inf; shift them by 0.
        m_safe = tl.where(m_new == float("-inf"), 0.0, m_new)
        alpha = tl.exp2(m_i - m_safe)
        p = tl.exp2(s - m_safe[:, None])
        l_i = l_i * alpha + tl.sum(p, 1)
        acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
        m_i = m_new

    has_keys = l_i > 0
    out = acc / tl.where(has_keys, l_i, 1.0)[:, None]
    log2_l = tl.log2(tl.where(has_keys, l_i, 1.0))
    lse = tl.where(has_keys, (m_i + log2_l) / 1.4426950408889634, float("-inf"))
    part = split * T + tok
    tl.store(
        O_PART + (part[:, None] * H + head) * D + offs_d[None, :], out, m_mask[:, None]
    )
    tl.store(LSE_PART + part * H + head, lse, m_mask)


def rel_attention(
    q: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    rel_logits: torch.Tensor,
    *,
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    max_query_len: int,
    max_seq_len: int,
    softmax_scale: float,
    window_left: int,
    out: torch.Tensor,
    lse: torch.Tensor,
) -> None:
    """fp32 `(T, H, D)` attention `out` and natural-log `(T, H)` `lse` of
    the first `cu_seqlens_q[-1]` tokens of `q`; later rows (CUDA-graph
    padding) come out 0. `window_left` 0 is a full layer. The host lengths
    only size the launch."""
    num_tokens, num_heads, head_dim = q.shape
    page, num_kv_heads = key_cache.shape[1], key_cache.shape[2]
    extent = rel_logits.shape[2] - 256
    num_reqs = seq_lens.shape[0]
    assert q.stride(2) == 1 and key_cache.stride(3) == value_cache.stride(3) == 1
    assert rel_logits.stride(2) == 1 and rel_logits.shape[1] == num_heads
    assert out.shape == q.shape and lse.shape == (num_tokens, num_heads)

    block_m = 16 if max_query_len <= 16 else 64
    block_n = 64
    m_blocks = triton.cdiv(max_query_len, block_m)
    # Split-KV only pays when the query blocks alone leave SMs idle.
    num_splits = 1
    work = num_reqs * num_heads * m_blocks
    sms = _num_sms(q.device.index or 0)
    if work < sms:
        visible = max_seq_len if window_left == 0 else window_left + max_query_len
        num_splits = max(1, min(sms * 2 // work, triton.cdiv(visible, 4 * block_n), 64))

    # `lse` may be a transposed view (the caller's (H, T) buffer).
    o_part = (
        out.view(1, *out.shape)
        if num_splits == 1 and out.is_contiguous()
        else q.new_empty((num_splits, *q.shape), dtype=torch.float32)
    )
    lse_part = q.new_empty((num_splits, num_tokens, num_heads), dtype=torch.float32)
    # Padding rows are never written.
    o_part.zero_()
    lse_part.fill_(float("-inf"))

    _rel_attention_kernel[(num_reqs, num_heads, m_blocks * num_splits)](
        q,
        key_cache,
        value_cache,
        rel_logits,
        block_table,
        seq_lens,
        cu_seqlens_q,
        o_part,
        lse_part,
        softmax_scale,
        q.stride(0),
        q.stride(1),
        *key_cache.stride()[:3],
        *value_cache.stride()[:3],
        rel_logits.stride(0),
        rel_logits.stride(1),
        block_table.stride(0),
        num_tokens,
        H=num_heads,
        GROUP=num_heads // num_kv_heads,
        D=head_dim,
        E=extent,
        PAGE=page,
        WINDOW_LEFT=window_left,
        NUM_SPLITS=num_splits,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        num_warps=4,
        num_stages=2,
    )
    if num_splits > 1:
        total = torch.logsumexp(lse_part, 0)
        weights = torch.exp(lse_part - total).nan_to_num_(0.0)
        out.copy_((o_part * weights[..., None]).sum(0))
        lse.copy_(total)
    else:
        if o_part.data_ptr() != out.data_ptr():
            out.copy_(o_part[0])
        lse.copy_(lse_part[0])
    # gate_sink reads the LSE of every row: padding rows (and requests with
    # no keys) get 0, with a zero output.
    lse.masked_fill_(lse == float("-inf"), 0.0)
