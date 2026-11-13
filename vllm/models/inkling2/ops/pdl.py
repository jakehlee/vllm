# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Programmatic dependent launch (PDL), gated to the GPUs that have it.

`griddepcontrol` needs SM90+: ptxas rejects it for SM80, so on Ampere the
kernels compile without it and launch without `launch_pdl`. PDL only lets a
kernel start before the previous one finishes; without it, kernels just run
in stream order.
"""

from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton

PDL = current_platform.is_cuda() and current_platform.has_device_capability(90)
_USE_PDL = tl.constexpr(PDL)


@triton.jit
def gdc_wait():
    if _USE_PDL:
        tl.extra.cuda.gdc_wait()


@triton.jit
def gdc_launch_dependents():
    if _USE_PDL:
        tl.extra.cuda.gdc_launch_dependents()
