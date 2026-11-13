# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Inkling2 Marlin experts (non-SM100) vs. dequantized torch references.

Runs on one Ampere or Hopper GPU. With INKLING2_CKPT set to a checkpoint
snapshot, also checks real NVFP4 (layer 0) and MXFP8 (layer 40) experts.
Self-contained (its own `dist_init`), so it also runs as a lone file against an
installed vLLM, outside the repo's tests/ tree.
"""

import json
import os
import tempfile
from types import SimpleNamespace

import pytest
import torch

from vllm.platforms import current_platform

if not current_platform.is_cuda() or current_platform.is_device_capability_family(100):
    pytest.skip("Marlin experts are the non-SM100 path", allow_module_level=True)

import vllm.models.inkling2.moe_marlin as moe_marlin  # noqa: E402
from vllm.models.inkling2.moe import Inkling2MoE  # noqa: E402
from vllm.models.inkling2.moe_marlin import select_experts  # noqa: E402

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
    # The FileStore may already have deleted it.
    if os.path.exists(path):
        os.remove(path)


E2M1 = [0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6]

MOE_CONFIG = SimpleNamespace(
    hidden_size=512,
    intermediate_size=256,
    n_routed_experts=16,
    n_shared_experts=0,
    num_experts_per_tok=8,
    route_scale=8.0,
    mlp_gate_softcap=4.0,
    mlp_up_softcap=25.0,
    use_gate_bias=True,
)


def dequant_nvfp4(values: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    lut = torch.tensor(E2M1, device=values.device)
    v = torch.stack([lut[(values & 0xF).long()], lut[(values >> 4).long()]], -1)
    return v.flatten(-2) * scales.float().repeat_interleave(16, -1)


def dequant_mxfp8(values: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    return values.float() * torch.exp2(scales.float() - 127).repeat_interleave(32, -1)


def _random_moe_weights(fmt: str, cfg=MOE_CONFIG) -> dict[str, torch.Tensor]:
    torch.manual_seed(0)
    e, h, i = cfg.n_routed_experts, cfg.hidden_size, cfg.intermediate_size
    weights = {
        "gate.weight": (torch.randn(e, h, device="cuda") * 0.05).bfloat16(),
        "gate.bias": torch.randn(e, device="cuda") * 0.01,
    }
    for name, n, k in (("w13", 2 * i, h), ("w2", h, i)):
        prefix = f"experts.{name}_weight"
        if fmt == "nvfp4":
            values = torch.randint(0, 256, (e, n, k // 2), device="cuda").to(torch.uint8)
            scales = (torch.rand(e, n, k // 16, device="cuda") + 0.5).to(torch.float8_e4m3fn)
            weights[f"{prefix}_amax"] = torch.rand(e, device="cuda") * 100 + 100
            weights[f"{prefix}_outer_scale_denominator"] = torch.tensor(
                [2688.0], device="cuda"
            )
        else:
            values = torch.randn(e, n, k, device="cuda").to(torch.float8_e4m3fn)
            scales = torch.randint(120, 128, (e, n, k // 32), device="cuda").to(torch.uint8)
        weights[prefix], weights[f"{prefix}_scale"] = values, scales
    return weights


def _dequantized(weights: dict[str, torch.Tensor], name: str, e: int) -> torch.Tensor:
    """fp64 [N, K] weights of expert `e` (one at a time: a full layer of fp64
    experts would not fit)."""
    prefix = f"experts.{name}_weight"
    if f"{prefix}_amax" in weights:
        w = dequant_nvfp4(weights[prefix][e], weights[f"{prefix}_scale"][e]).double()
        scale = weights[f"{prefix}_amax"][e] / weights[f"{prefix}_outer_scale_denominator"]
        return w * scale.double()
    return dequant_mxfp8(weights[prefix][e], weights[f"{prefix}_scale"][e]).double()


def _build_moe(fmt: str, weights: dict[str, torch.Tensor], cfg=MOE_CONFIG) -> Inkling2MoE:
    with torch.device("cuda"):
        moe = Inkling2MoE(cfg, fmt)
    assert moe.marlin
    for name, param in moe.named_parameters():
        param.weight_loader(param, weights[name])
    moe.experts.process_weights_after_loading()
    return moe


def _moe_reference(weights: dict[str, torch.Tensor], x: torch.Tensor, cfg=MOE_CONFIG):
    """Per-token loop over dequantized experts with bf16 activations, rounding
    where the Marlin path rounds (gate-up output, SwiGLU output, each expert's
    down output)."""
    routing, ids = select_experts(
        x,
        weights["gate.weight"],
        weights["gate.bias"],
        cfg.num_experts_per_tok,
        cfg.route_scale,
    )
    out = torch.zeros(x.shape, dtype=torch.float64, device=x.device)
    for e in ids.unique().tolist():
        w13, w2 = _dequantized(weights, "w13", e), _dequantized(weights, "w2", e)
        for t, slot in (ids == e).nonzero().tolist():
            gate_up = (x[t].double() @ w13.T).bfloat16().double()
            gate, up = gate_up[0::2], gate_up[1::2]
            act = (
                cfg.mlp_gate_softcap
                * torch.tanh(gate / cfg.mlp_gate_softcap)
                * torch.sigmoid(gate)
                * cfg.mlp_up_softcap
                * torch.tanh(up / cfg.mlp_up_softcap)
                * routing[t, slot]
            )
            out[t] += (act.bfloat16().double() @ w2.T).bfloat16().double()
    return out


def _assert_close(out: torch.Tensor, ref: torch.Tensor) -> None:
    scale = ref.abs().max().item()
    assert scale > 0.1
    assert not out.isnan().any()
    rel = ((out.double() - ref).norm() / ref.norm()).item()
    assert rel < 1e-2, f"relative error {rel:.2e}"


def test_routing_breaks_ties_low():
    torch.manual_seed(0)
    weight = (torch.randn(16, 512, device="cuda") * 0.05).bfloat16()
    bias = torch.randn(16, device="cuda") * 0.01
    weight[[2, 3]] = weight[1].clone()
    bias[[1, 2, 3]] = 1.0
    x = torch.randn(7, 512, device="cuda").bfloat16()
    routing, ids = select_experts(x, weight, bias, 8, 8.0)
    assert (ids[:, :3] == torch.tensor([1, 2, 3], device="cuda", dtype=torch.int32)).all()
    torch.testing.assert_close(routing.sum(-1), torch.full((7,), 8.0, device="cuda"))


@pytest.mark.parametrize("fmt", ["nvfp4", "mxfp8"])
@pytest.mark.parametrize("tokens", [1, 5, 33, 300])
def test_moe_layer_matches_reference(dist_init, fmt: str, tokens: int):
    weights = _random_moe_weights(fmt)
    moe = _build_moe(fmt, weights)
    x = torch.randn(tokens, 512, device="cuda").bfloat16()
    _assert_close(moe(x), _moe_reference(weights, x))


@pytest.mark.parametrize("fmt", ["nvfp4", "mxfp8"])
def test_moe_expert_parallel_matches_single_rank(dist_init, monkeypatch, fmt: str):
    """EP=4, emulated on one GPU, sums to the EP=1 output."""
    weights = _random_moe_weights(fmt)
    x = torch.randn(64, 512, device="cuda").bfloat16()
    expected = _build_moe(fmt, weights)(x)

    partials: list[torch.Tensor] = []
    monkeypatch.setattr(moe_marlin, "get_tensor_model_parallel_world_size", lambda: 4)
    monkeypatch.setattr(
        moe_marlin, "tensor_model_parallel_all_reduce", lambda p: partials.append(p) or p
    )
    for rank in range(4):
        monkeypatch.setattr(moe_marlin, "get_tensor_model_parallel_rank", lambda r=rank: r)
        _build_moe(fmt, weights)(x)

    assert len(partials) == 4 and all(not p.isnan().any() for p in partials)
    actual = sum(p.float() for p in partials)
    scale = expected.abs().max().item()
    torch.testing.assert_close(actual, expected.float(), atol=4 * 2**-8 * scale, rtol=2e-2)


@pytest.mark.skipif("INKLING2_CKPT" not in os.environ, reason="set INKLING2_CKPT")
@pytest.mark.parametrize("layer", [0, 40])
def test_checkpoint_layer_matches_reference(dist_init, layer: int):
    """A real layer at full size (384 experts, hidden 4608, intermediate
    2304); NVFP4 in layer 0, MXFP8 in layer 40."""
    from safetensors import safe_open

    ckpt = os.environ["INKLING2_CKPT"]
    weight_map = json.load(open(os.path.join(ckpt, "model.safetensors.index.json")))
    prefix = f"model.llm.layers.{layer}.mlp."
    weights = {}
    for name, file in weight_map["weight_map"].items():
        if name.startswith(prefix):
            with safe_open(os.path.join(ckpt, file), "pt", device="cuda") as f:
                weights[name.removeprefix(prefix)] = f.get_tensor(name)
    cfg = SimpleNamespace(
        **{
            **vars(MOE_CONFIG),
            "hidden_size": 4608,
            "intermediate_size": 2304,
            "n_routed_experts": 384,
        }
    )
    fmt = "nvfp4" if "experts.w13_weight_amax" in weights else "mxfp8"
    assert fmt == ("nvfp4" if layer < 40 else "mxfp8")
    moe = _build_moe(fmt, weights, cfg)
    torch.manual_seed(0)
    x = torch.randn(4, 4608, device="cuda").bfloat16()
    _assert_close(moe(x), _moe_reference(weights, x, cfg))
