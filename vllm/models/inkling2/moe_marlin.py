# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Inkling2 routed experts on GPUs without the SM100 expert kernels (Ampere,
Hopper): weight-only NVFP4 / MXFP8 Marlin GEMMs with bf16 activations.

The SM100 path (`moe.Inkling2Experts`) quantizes the activations to the expert
format (NVFP4 / MXFP8, as the model was trained); here they stay bf16, the only
numerical difference. Everything else follows the SM100 path: sigmoid + bias
top-k selection with renormalized log-sigmoid routing weights, the softcapped
SwiGLU with the routing weight applied before the down projection, one bf16
rounding of each rank's combine, then the all-reduce over the EP (= TP) group.
"""

from types import SimpleNamespace

import torch
from torch import nn

from vllm.distributed import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_reduce,
)
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.experts.marlin_moe import fused_marlin_moe
from vllm.model_executor.layers.quantization.utils.marlin_utils_fp4 import (
    prepare_nvfp4_moe_layer_for_marlin,
)
from vllm.model_executor.layers.quantization.utils.marlin_utils_fp8 import (
    prepare_mxfp8_moe_layer_for_marlin,
)
from vllm.model_executor.utils import replace_parameter, set_weight_attrs
from vllm.platforms import current_platform
from vllm.scalar_type import scalar_types


def use_marlin_experts() -> bool:
    """The SM100 expert kernels (FlashInfer trtllm-gen) need Blackwell."""
    return current_platform.is_cuda() and not current_platform.is_device_capability_family(
        100
    )


def select_experts(
    x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, topk: int, route_scale: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """fp32 routing weights and int32 ids [T, topk]: top-k of sigmoid(logits) +
    bias, weighted by the renormalized sigmoid of the selected logits. The
    logits are fp32, as in the SM100 router (`ops.linear.linear_k_groups`)."""
    logits = x.float() @ weight.float().T
    _, ids = torch.topk(logits.sigmoid() + bias, topk, dim=-1, sorted=True)
    selected = torch.nn.functional.logsigmoid(logits.gather(1, ids))
    return selected.softmax(-1) * route_scale, ids.to(torch.int32)


class Inkling2MarlinExperts(nn.Module):
    """Same parameters and checkpoint names as `moe.Inkling2Experts`, loaded
    row-major (no SM100 row shuffle or scale swizzle) and repacked for Marlin
    in `process_weights_after_loading`."""

    def __init__(self, config, quant_format: str) -> None:
        super().__init__()
        self.ep_rank = get_tensor_model_parallel_rank()
        self.ep_size = get_tensor_model_parallel_world_size()
        assert config.n_routed_experts % self.ep_size == 0
        self.global_num_experts = config.n_routed_experts
        self.num_experts = config.n_routed_experts // self.ep_size
        self.topk = config.num_experts_per_tok
        self.gate_cap = config.mlp_gate_softcap
        self.up_cap = config.mlp_up_softcap
        self.nvfp4 = quant_format == "nvfp4"
        hidden, inter = config.hidden_size, config.intermediate_size
        self.hidden_size, self.intermediate_size = hidden, inter

        pack, group = (2, 16) if self.nvfp4 else (1, 32)
        weight_dtype = torch.uint8 if self.nvfp4 else torch.float8_e4m3fn
        scale_dtype = torch.float8_e4m3fn if self.nvfp4 else torch.uint8
        # w13 rows interleave gate and up: row 2i is gate i, row 2i + 1 is up i.
        for name, (n, k) in zip(("w13", "w2"), ((2 * inter, hidden), (hidden, inter))):
            self._register(f"{name}_weight", (n, k // pack), weight_dtype)
            self._register(f"{name}_weight_scale", (n, k // group), scale_dtype)
            if self.nvfp4:
                self._register(f"{name}_weight_amax", (), torch.float32)
                self._register(
                    f"{name}_weight_outer_scale_denominator",
                    (1,),
                    torch.float32,
                    per_expert=False,
                )

        expert_map = None
        if self.ep_size > 1:
            start = self.ep_rank * self.num_experts
            expert_map = torch.full((self.global_num_experts,), -1, dtype=torch.int32)
            expert_map[start : start + self.num_experts] = torch.arange(
                self.num_experts, dtype=torch.int32
            )
        self.register_buffer("expert_map", expert_map, persistent=False)
        self.w13_global_scale: torch.Tensor | None = None
        self.w2_global_scale: torch.Tensor | None = None

    def _register(
        self, name: str, shape: tuple[int, ...], dtype: torch.dtype, per_expert: bool = True
    ) -> None:
        if per_expert:
            shape = (self.num_experts, *shape)
        param = nn.Parameter(torch.empty(shape, dtype=dtype), requires_grad=False)
        loader = self._load_expert_shard if per_expert else self._load_full
        set_weight_attrs(param, {"weight_loader": loader})
        self.register_parameter(name, param)

    def _load_expert_shard(self, param: nn.Parameter, loaded: torch.Tensor) -> None:
        start = self.ep_rank * self.num_experts
        self._load_full(param, loaded[start : start + self.num_experts])

    @staticmethod
    def _load_full(param: nn.Parameter, loaded: torch.Tensor) -> None:
        assert loaded.shape == param.shape and loaded.dtype == param.dtype, (
            f"{loaded.shape} {loaded.dtype} != {param.shape} {param.dtype}"
        )
        param.data.copy_(loaded)

    def process_weights_after_loading(self) -> None:
        if self.nvfp4:
            # The per-expert global weight scale is amax / outer_scale_denominator
            # (the weight half of the SM100 down-projection row scale,
            # `ops.moe._row_scales`).
            scales = [
                getattr(self, f"{name}_weight_amax")
                / getattr(self, f"{name}_weight_outer_scale_denominator")
                for name in ("w13", "w2")
            ]
            layer = SimpleNamespace(
                num_experts=self.num_experts,
                hidden_size=self.hidden_size,
                intermediate_size_per_partition=self.intermediate_size,
                params_dtype=torch.bfloat16,
            )
            w13, w13_scale, w13_gs, w2, w2_scale, w2_gs = (
                prepare_nvfp4_moe_layer_for_marlin(
                    layer,
                    self.w13_weight.data,
                    self.w13_weight_scale.data,
                    scales[0],
                    self.w2_weight.data,
                    self.w2_weight_scale.data,
                    scales[1],
                    is_act_and_mul=True,
                )
            )
            self.w13_global_scale = w13_gs.float()
            self.w2_global_scale = w2_gs.float()
            for name in ("w13", "w2"):
                delattr(self, f"{name}_weight_amax")
                delattr(self, f"{name}_weight_outer_scale_denominator")
        else:
            w13, w2, w13_scale, w2_scale = prepare_mxfp8_moe_layer_for_marlin(
                self,
                self.w13_weight.data,
                self.w2_weight.data,
                self.w13_weight_scale.data,
                self.w2_weight_scale.data,
            )
        replace_parameter(self, "w13_weight", w13)
        replace_parameter(self, "w2_weight", w2)
        replace_parameter(self, "w13_weight_scale", w13_scale)
        replace_parameter(self, "w2_weight_scale", w2_scale)

    def _swiglu(self, topk_weights: torch.Tensor):
        """The softcapped SwiGLU times the routing weight, rounded once to bf16
        (`ops.activation._situ`); Marlin's gate-up output rows are
        token-major, (token, slot)."""
        gate_cap, up_cap = self.gate_cap, self.up_cap
        gammas = topk_weights.reshape(-1, 1)

        def activation(_, out: torch.Tensor, gate_up: torch.Tensor, **kwargs) -> None:
            gate, up = gate_up[:, 0::2].float(), gate_up[:, 1::2].float()
            y = gate_cap * torch.tanh(gate / gate_cap) * torch.sigmoid(gate)
            y = y * (up_cap * torch.tanh(up / up_cap)) * gammas[: gate.shape[0]]
            out.copy_(y)

        return activation

    def forward(
        self, x: torch.Tensor, topk_weights: torch.Tensor, topk_ids: torch.Tensor
    ) -> torch.Tensor:
        out = fused_marlin_moe(
            x,
            self.w13_weight,
            self.w2_weight,
            None,
            None,
            self.w13_weight_scale,
            self.w2_weight_scale,
            topk_weights,
            topk_ids,
            (scalar_types.float4_e2m1f if self.nvfp4 else scalar_types.float8_e4m3fn).id,
            global_num_experts=self.global_num_experts,
            activation=MoEActivation.SILU,
            activation_func=self._swiglu(topk_weights),
            expert_map=self.expert_map,
            global_scale1=self.w13_global_scale,
            global_scale2=self.w2_global_scale,
            # The routing weight is already in the SwiGLU output.
            mul_topk_weights_in_gemm2=False,
        )
        return out if self.ep_size == 1 else tensor_model_parallel_all_reduce(out)
