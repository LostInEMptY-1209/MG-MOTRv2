# ------------------------------------------------------------------------
# Copyright (c) 2022 megvii-research. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from Deformable DETR (https://github.com/fundamentalvision/Deformable-DETR)
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------


from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import warnings
import math

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.init import xavier_uniform_, constant_

try:
    from ..functions import MSDeformAttnFunction
except ImportError:
    MSDeformAttnFunction = None

def forward_pytorch(
    self,
    query,                # [B, Q, C]
    reference_points,     # [B, Q, n_levels, 2]
    input_flatten,        # [B, sum(HW), C]
    spatial_shapes,       # [n_levels, 2]
    level_start_index     # [n_levels]
):
    """
    Pure PyTorch fallback of Multi-Scale Deformable Attention
    """
    B, Q, C = query.shape
    n_levels = spatial_shapes.shape[0]
    n_heads = self.n_heads
    n_points = self.n_points
    head_dim = C // n_heads

    # value projection
    value = self.value_proj(input_flatten)
    value = value.view(B, -1, n_heads, head_dim)

    # sampling offsets & attention weights
    sampling_offsets = self.sampling_offsets(query)
    sampling_offsets = sampling_offsets.view(
        B, Q, n_heads, n_levels, n_points, 2
    )

    attn_weights = self.attention_weights(query)
    attn_weights = attn_weights.view(
        B, Q, n_heads, n_levels, n_points
    )
    attn_weights = attn_weights.softmax(-1)

    output = torch.zeros(
        B, Q, n_heads, head_dim,
        device=query.device
    )

    for lvl in range(n_levels):
        H, W = spatial_shapes[lvl]
        start = level_start_index[lvl]
        end = start + H * W

        # [B, H*W, n_heads, head_dim]
        value_l = value[:, start:end].view(
            B, H, W, n_heads, head_dim
        ).permute(0, 3, 4, 1, 2)  # [B, n_heads, head_dim, H, W]

        # normalize reference points to [-1, 1]
        ref = reference_points[:, :, lvl].unsqueeze(2)
        offset = sampling_offsets[:, :, :, lvl]

        sampling_loc = ref + offset / torch.tensor(
            [W, H], device=query.device
        )

        sampling_grid = sampling_loc * 2 - 1
        sampling_grid = sampling_grid.view(
            B * n_heads, Q * n_points, 1, 2
        )

        value_l = value_l.reshape(
            B * n_heads, head_dim, H, W
        )

        sampled = torch.nn.functional.grid_sample(
            value_l,
            sampling_grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=False
        )

        sampled = sampled.view(
            B, n_heads, head_dim, Q, n_points
        )

        attn = attn_weights[:, :, :, lvl].permute(0, 2, 1, 3)
        output += (sampled * attn.unsqueeze(2)).sum(-1)

    output = output.permute(0, 2, 1, 3).reshape(B, Q, C)
    output = self.output_proj(output)

    return output


def _is_power_of_2(n):
    if (not isinstance(n, int)) or (n < 0):
        raise ValueError("invalid input for _is_power_of_2: {} (type: {})".format(n, type(n)))
    return (n & (n-1) == 0) and n != 0


class MSDeformAttn(nn.Module):
    def __init__(self, d_model=256, n_levels=4, n_heads=8, n_points=4, sigmoid_attn=False):
        """
        Multi-Scale Deformable Attention Module
        :param d_model      hidden dimension
        :param n_levels     number of feature levels
        :param n_heads      number of attention heads
        :param n_points     number of sampling points per attention head per feature level
        """
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError('d_model must be divisible by n_heads, but got {} and {}'.format(d_model, n_heads))
        _d_per_head = d_model // n_heads
        # you'd better set _d_per_head to a power of 2 which is more efficient in our CUDA implementation
        if not _is_power_of_2(_d_per_head):
            warnings.warn("You'd better set d_model in MSDeformAttn to make the dimension of each attention head a power of 2 "
                          "which is more efficient in our CUDA implementation.")

        self.im2col_step = 64
        self.sigmoid_attn = sigmoid_attn

        self.d_model = d_model
        self.n_levels = n_levels
        self.n_heads = n_heads
        self.n_points = n_points

        self.sampling_offsets = nn.Linear(d_model, n_heads * n_levels * n_points * 2)
        self.attention_weights = nn.Linear(d_model, n_heads * n_levels * n_points)
        self.value_proj = nn.Linear(d_model, d_model)
        self.output_proj = nn.Linear(d_model, d_model)

        self._reset_parameters()

    def _reset_parameters(self):
        constant_(self.sampling_offsets.weight.data, 0.)
        thetas = torch.arange(self.n_heads, dtype=torch.float32) * (2.0 * math.pi / self.n_heads)
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        grid_init = (grid_init / grid_init.abs().max(-1, keepdim=True)[0]).view(self.n_heads, 1, 1, 2).repeat(1, self.n_levels, self.n_points, 1)
        for i in range(self.n_points):
            grid_init[:, :, i, :] *= i + 1
        with torch.no_grad():
            self.sampling_offsets.bias = nn.Parameter(grid_init.view(-1))
        constant_(self.attention_weights.weight.data, 0.)
        constant_(self.attention_weights.bias.data, 0.)
        xavier_uniform_(self.value_proj.weight.data)
        constant_(self.value_proj.bias.data, 0.)
        xavier_uniform_(self.output_proj.weight.data)
        constant_(self.output_proj.bias.data, 0.)

    def forward(
            self,
            query,
            reference_points,
            input_flatten,
            spatial_shapes,
            level_start_index
    ):
        if MSDeformAttnFunction is not None:
            return MSDeformAttnFunction.apply(
                query,
                reference_points,
                input_flatten,
                spatial_shapes,
                level_start_index,
                self.sampling_offsets.weight,
                self.attention_weights.weight,
                self.value_proj.weight,
                self.output_proj.weight,
                self.n_heads,
                self.n_points
            )
        else:
            # 🔥 PyTorch fallback
            return self.forward_pytorch(
                query,
                reference_points,
                input_flatten,
                spatial_shapes,
                level_start_index
            )

