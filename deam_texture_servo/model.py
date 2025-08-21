# SPDX-License-Identifier: MIT
from __future__ import annotations

import math
from collections import OrderedDict
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import timm  # using a ViT backbone with features_only=True

# Use the local utils (TempModule etc.)
from .utils_dynamicConv import TempModule

# Input image size (kept from the original code)
image_size = (1080 // 2, 1920 // 2)

# ---------------------------------------------------------------------
# Small utility blocks
# ---------------------------------------------------------------------
def conv_1x1_bn(
    inp: int,
    oup: int,
    conv_layer: type[nn.Conv2d] = nn.Conv2d,
    norm_layer: type[nn.BatchNorm2d] = nn.BatchNorm2d,
    nlin_layer: type[nn.Module] = nn.ReLU,
) -> nn.Sequential:
    return nn.Sequential(
        conv_layer(inp, oup, 1, 1, 0, bias=False),
        norm_layer(oup),
        nlin_layer(),
    )


def conv_2d(
    inp: int,
    oup: int,
    kernel_size: int = 3,
    stride: int = 1,
    padding: int = 0,
    groups: int = 1,
    bias: bool = False,
    norm: bool = True,
    act: bool = True,
) -> nn.Sequential:
    conv = nn.Sequential()
    conv.add_module("conv", nn.Conv2d(inp, oup, kernel_size, stride, padding, bias=bias, groups=groups))
    if norm:
        conv.add_module("BatchNorm2d", nn.BatchNorm2d(oup))
    if act:
        conv.add_module("Activation", nn.GELU())
    return conv


# ---------------------------------------------------------------------
# Linear cross attention
# ---------------------------------------------------------------------
class LinearCrossAttention(nn.Module):
    def __init__(self, embed_dim: int, attn_dropout: float = 0.0):
        super().__init__()
        self.q_proj = conv_2d(embed_dim, 1, kernel_size=1, bias=True, norm=False, act=False)
        self.k_proj = conv_2d(embed_dim, embed_dim, kernel_size=1, bias=True, norm=False, act=False)
        self.v_proj = conv_2d(embed_dim, embed_dim, kernel_size=1, bias=True, norm=False, act=False)
        self.attn_dropout = nn.Dropout(attn_dropout)
        self.out_proj = conv_2d(embed_dim, embed_dim, kernel_size=1, bias=True, norm=False, act=False)
        self.embed_dim = embed_dim

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        q = self.q_proj(x2)
        k = self.k_proj(x1)
        v = self.v_proj(x1)

        context_score = F.softmax(q, dim=-1)
        context_score = self.attn_dropout(context_score)

        context_vector = k * context_score
        context_vector = torch.sum(context_vector, dim=-1, keepdim=True)

        out = F.relu(v) * context_vector.expand_as(v)
        out = self.out_proj(out)
        return out


class LinearAttnFFN(nn.Module):
    def __init__(self, embed_dim: int, ffn_latent_dim: int, dropout: float = 0.0, attn_dropout: float = 0.0):
        super().__init__()
        self.g_norm = nn.GroupNorm(num_channels=embed_dim, eps=1e-5, affine=True, num_groups=1)
        self.linear_atten = LinearCrossAttention(embed_dim, attn_dropout)
        self.drop = nn.Dropout(dropout)

        self.pre_norm_ffn = nn.Sequential(
            nn.GroupNorm(num_channels=embed_dim, eps=1e-5, affine=True, num_groups=1),
            conv_2d(embed_dim, ffn_latent_dim, kernel_size=1, stride=1, bias=True, norm=False, act=True),
            nn.Dropout(dropout),
            conv_2d(ffn_latent_dim, embed_dim, kernel_size=1, stride=1, bias=True, norm=False, act=False),
            nn.Dropout(dropout),
        )

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        res_x1 = x1
        x1 = self.g_norm(x1)
        x1 = self.linear_atten(x1, x2)
        x1 = self.drop(x1)
        x1 = x1 + res_x1
        x1 = x1 + self.pre_norm_ffn(x1)
        return x1


# ---------------------------------------------------------------------
# Mobile CrossViT Block v2
# ---------------------------------------------------------------------
class MobileCrossViTBlockv2(nn.Module):
    def __init__(self, inp: int, attn_dim: int, ffn_multiplier: float, attn_blocks: int, patch_size: Tuple[int, int]):
        super().__init__()
        self.patch_h, self.patch_w = patch_size

        # Local representations for the two streams
        self.local_rep_x1 = nn.Sequential()
        self.local_rep_x1.add_module("conv_1x1", conv_2d(inp, attn_dim, kernel_size=1, stride=1, norm=False, act=False))

        self.local_rep_x2 = nn.Sequential()
        self.local_rep_x2.add_module("conv_1x1", conv_2d(inp, attn_dim, kernel_size=1, stride=1, norm=False, act=False))

        # Global representation via linear attention + FFN
        ffn_dims = [int((ffn_multiplier * attn_dim) // 16 * 16)] * attn_blocks
        self.linear_attn_x1 = nn.ModuleList(
            [LinearAttnFFN(attn_dim, ffn_dims[i], attn_dropout=0.0, dropout=0.0) for i in range(attn_blocks)]
        )
        self.linear_attn_x2 = nn.ModuleList(
            [LinearAttnFFN(attn_dim, ffn_dims[i], attn_dropout=0.0, dropout=0.0) for i in range(attn_blocks)]
        )

        self.g_norm_x1 = nn.GroupNorm(num_channels=attn_dim, eps=1e-5, affine=True, num_groups=1)
        self.g_norm_x2 = nn.GroupNorm(num_channels=attn_dim, eps=1e-5, affine=True, num_groups=1)

        self.conv_proj2_x1 = conv_2d(attn_dim * 2, inp, kernel_size=1, stride=1, padding=0, act=True)
        self.conv_proj2_x2 = conv_2d(attn_dim * 2, inp, kernel_size=1, stride=1, padding=0, act=True)

    def unfolding_pytorch(self, feature_map: torch.Tensor) -> Tuple[torch.Tensor, Tuple[int, int]]:
        b, c, img_h, img_w = feature_map.shape
        # [B, C, H, W] -> [B, C, P, N]
        patches = F.unfold(
            feature_map,
            kernel_size=(self.patch_h, self.patch_w),
            stride=(self.patch_h, self.patch_w),
        )
        patches = patches.reshape(b, c, self.patch_h * self.patch_w, -1)
        return patches, (img_h, img_w)

    def folding_pytorch(self, patches: torch.Tensor, output_size: Tuple[int, int]) -> torch.Tensor:
        b, in_dim, patch_size, n_patches = patches.shape
        # [B, C, P, N] -> [B, C, H, W]
        patches = patches.reshape(b, in_dim * patch_size, n_patches)
        feature_map = F.fold(
            patches,
            output_size=output_size,
            kernel_size=(self.patch_h, self.patch_w),
            stride=(self.patch_h, self.patch_w),
        )
        return feature_map

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x1 = self.local_rep_x1(x1)
        x2 = self.local_rep_x2(x2)

        x1, output_size = self.unfolding_pytorch(x1)
        x2, _ = self.unfolding_pytorch(x2)

        for i in range(len(self.linear_attn_x1)):
            x1 = self.linear_attn_x1[i](x1, x2)
            x2 = self.linear_attn_x2[i](x2, x1)

        x1 = self.g_norm_x1(x1)
        x2 = self.g_norm_x2(x2)

        x1 = self.folding_pytorch(patches=x1, output_size=output_size)
        x2 = self.folding_pytorch(patches=x2, output_size=output_size)

        diffx1 = x1 - x2
        diffx2 = x2 - x1

        x1 = torch.cat((x1, diffx1), dim=1)
        x2 = torch.cat((x2, diffx2), dim=1)

        x1 = self.conv_proj2_x1(x1)
        x2 = self.conv_proj2_x2(x2)
        return x1, x2


# ---------------------------------------------------------------------
# EfficientNet-like MBConv pieces
# ---------------------------------------------------------------------
class ConvBNAct(nn.Sequential):
    """Convolution → Normalization → Activation"""

    def __init__(
        self,
        in_channel: int,
        out_channel: int,
        kernel_size: int,
        stride: int,
        groups: int,
        norm_layer: nn.Module,
        act: type[nn.Module],
        conv_layer: type[nn.Conv2d] = nn.Conv2d,
    ):
        super().__init__(
            conv_layer(in_channel, out_channel, kernel_size, stride=stride, padding=(kernel_size - 1) // 2, groups=groups, bias=False),
            norm_layer,
            act(),
        )


class StochasticDepth(nn.Module):
    """Stochastic Depth (a.k.a. DropPath)."""

    def __init__(self, prob: float, mode: str):
        super().__init__()
        self.prob = prob
        self.survival = 1.0 - prob
        self.mode = mode

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.prob == 0.0 or not self.training:
            return x
        shape = [x.size(0)] + [1] * (x.ndim - 1) if self.mode == "row" else [1]
        return x * torch.empty(shape, device=x.device).bernoulli_(self.survival).div_(self.survival)


class Hswish(nn.Module):
    def __init__(self, inplace: bool = True):
        super().__init__()
        self.inplace = inplace

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * F.relu6(x + 3.0, inplace=self.inplace) / 6.0


class MBConvConfig:
    """Configuration holder for MBConv blocks (kept API-compatible with the original)."""

    def __init__(
        self,
        expand_ratio: float,
        kernel: int,
        stride: int,
        in_ch: int,
        out_ch: int,
        layers: int,
        use_se: bool,
        fused: bool,
        act=nn.SiLU,
    ):
        self.expand_ratio = expand_ratio
        self.kernel = kernel
        self.stride = stride
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.num_layers = layers

        if act == "SI":
            self.act = nn.SiLU
        elif act == "GE":
            self.act = nn.GELU
        elif act == "HS":
            self.act = Hswish
        else:
            raise NotImplementedError

        self.use_se = use_se
        self.fused = fused

    @staticmethod
    def adjust_channels(channel: int, factor: float, divisible: int = 8) -> int:
        new_channel = channel * factor
        divisible_channel = max(divisible, (int(new_channel + divisible / 2) // divisible) * divisible)
        if divisible_channel < 0.9 * new_channel:
            divisible_channel += divisible
        return divisible_channel


# ---------------------- DC ---------------------
class ECAAttention(nn.Module):
    """ECA (Efficient Channel Attention) as lightweight channel attention."""

    def __init__(self, in_channels: int, k_size: int = 3):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.avg_pool(x)
        y = y.view(x.size(0), -1).unsqueeze(1)
        y = self.conv(y)
        y = self.sigmoid(y)
        y = y.view(x.size(0), x.size(1), 1, 1)
        return x * y.expand_as(x)


class AttentionLayer(nn.Module):
    """Generates mixture weights over dynamic kernels via global pooling."""

    def __init__(self, c_dim: int, hidden_dim: int, nof_kernels: int):
        super().__init__()
        self.global_pooling = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten())
        self.to_scores = nn.Sequential(
            nn.Linear(c_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, nof_kernels),
        )

    def forward(self, x: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
        out = self.global_pooling(x)
        scores = self.to_scores(out)
        return F.softmax(scores / temperature, dim=-1)


class DynamicConvolution(TempModule):
    """Dynamic depthwise convolution: per-sample aggregation of K candidate kernels."""

    def __init__(
        self,
        nof_kernels: int,
        reduce: int,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.groups = groups
        self.conv_args = {"stride": stride, "padding": padding, "dilation": dilation}
        self.nof_kernels = nof_kernels
        self.attention = AttentionLayer(in_channels, max(1, in_channels // reduce), nof_kernels)

        self.kernel_size = (kernel_size, kernel_size)
        self.kernels_weights = nn.Parameter(
            torch.Tensor(nof_kernels, out_channels, in_channels // self.groups, *self.kernel_size), requires_grad=True
        )
        if bias:
            self.kernels_bias = nn.Parameter(torch.Tensor(nof_kernels, out_channels), requires_grad=True)
        else:
            self.register_parameter("kernels_bias", None)
        self.initialize_parameters()

    def initialize_parameters(self) -> None:
        for i_kernel in range(self.nof_kernels):
            init.kaiming_uniform_(self.kernels_weights[i_kernel], a=math.sqrt(5))
        if self.kernels_bias is not None:
            bound = 1 / math.sqrt(self.kernels_weights[0, 0].numel())
            nn.init.uniform_(self.kernels_bias, -bound, bound)

    def forward(self, x: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
        bsz = x.shape[0]
        alphas = self.attention(x, temperature)

        # Aggregate candidate kernels with attention weights
        agg_weights = torch.sum(
            torch.mul(self.kernels_weights.unsqueeze(0), alphas.view(bsz, -1, 1, 1, 1, 1)), dim=1
        )
        agg_weights = agg_weights.view(-1, *agg_weights.shape[-3:])  # (B*out_c, in_c, k, k)

        if self.kernels_bias is not None:
            agg_bias = torch.sum(torch.mul(self.kernels_bias.unsqueeze(0), alphas.view(bsz, -1, 1)), dim=1)
            agg_bias = agg_bias.view(-1)
        else:
            agg_bias = None

        # Group trick: apply per-sample convs in a single call
        x_grouped = x.view(1, -1, *x.shape[-2:])  # (1, B*in_c, H, W)
        out = F.conv2d(x_grouped, agg_weights, agg_bias, groups=self.groups * bsz, **self.conv_args)
        out = out.view(bsz, -1, *out.shape[-2:])
        return out


class DCAB(nn.Module):
    """Channel attention (ECA) + Dynamic depthwise convolution + LN + GELU with residual."""

    def __init__(self, in_channels: int):
        super().__init__()
        self.in_channels = in_channels
        self.channel_att = ECAAttention(in_channels)
        self.dynamic_conv = DynamicConvolution(
            nof_kernels=12,
            reduce=1,
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=3,
            padding=1,
            groups=in_channels,
            bias=False,
        )
        self.norm = nn.LayerNorm(in_channels, eps=1e-6)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        original_x = x
        x = self.channel_att(x)
        out = self.dynamic_conv(x)

        # LayerNorm expects channel-last; convert back after normalization
        out = out.permute(0, 2, 3, 1).contiguous()
        out = self.norm(out)
        out = out.permute(0, 3, 1, 2).contiguous()

        out = self.act(out)
        out = out + original_x
        return out


class MBConv(nn.Module):
    """EfficientNet-like MBConv block with optional DCAB."""

    def __init__(self, c: MBConvConfig, sd_prob: float = 0.0, layer_scale_init_value: float = 1e-6):
        super().__init__()
        inter_channel = c.adjust_channels(c.in_ch, c.expand_ratio)
        block = []

        if c.expand_ratio == 1:
            # Not reached with current configs; kept to preserve the original structure
            block.append(("fused", ConvBNAct(c.in_ch, inter_channel, c.kernel, c.stride, 1, c.norm_layer(), c.act)))  # type: ignore[attr-defined]
        elif c.fused:
            block.append(("fused", ConvBNAct(c.in_ch, inter_channel, c.kernel, c.stride, 1, c.norm_layer, c.act)))  # type: ignore[attr-defined]
            block.append(("fused_point_wise", ConvBNAct(inter_channel, c.out_ch, 1, 1, 1, c.norm_layer, nn.Identity)))  # type: ignore[attr-defined]
        else:
            block.append(("linear_bottleneck", ConvBNAct(c.in_ch, inter_channel, 1, 1, 1, nn.BatchNorm2d(inter_channel), c.act)))
            block.append(("depth_wise", ConvBNAct(inter_channel, inter_channel, c.kernel, c.stride, inter_channel, nn.BatchNorm2d(inter_channel), c.act)))
            block.append(("dcab", DCAB(inter_channel)))
            block.append(("point_wise", ConvBNAct(inter_channel, c.out_ch, 1, 1, 1, nn.BatchNorm2d(c.out_ch), nn.Identity)))

        self.block = nn.Sequential(OrderedDict(block))
        self.use_skip_connection = c.stride == 1 and c.in_ch == c.out_ch
        self.stochastic_path = StochasticDepth(sd_prob, "row")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.block(x)
        if self.use_skip_connection:
            out = x + self.stochastic_path(out)
        return out


class RepeatedMBConv(nn.Module):
    """Stack MBConv blocks according to the provided MBConvConfig.num_layers."""

    def __init__(self, config: MBConvConfig):
        super().__init__()
        layers = []
        for _ in range(config.num_layers):
            layers.append(MBConv(config))
            config.in_ch = config.out_ch
            config.stride = 1  # Stride must be 1 for subsequent layers
        self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


# ---------------------------------------------------------------------
# Last Block
# ---------------------------------------------------------------------
class LastBlock(nn.Module):
    """Final heads for translation and rotation; keeps extra layers for compatibility."""

    def __init__(self, channels: int):
        super().__init__()
        self.fc_translation = nn.Linear(channels * 2, 3, bias=True)
        self.fc_rotation = nn.Linear(channels * 2, 3, bias=True)
        # Kept for backward-compatibility even if not used directly
        self.fc = nn.Linear(channels * 2, 6, bias=True)
        self.fc_norm = nn.Linear(channels * 2, 1, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = x.view(-1, x.shape[1])
        intermediate = x
        x1 = self.fc_translation(x)
        x2 = self.fc_rotation(x)
        return x1, x2, intermediate


# ---------------------------------------------------------------------
# Backbone (DEAM-based)
# ---------------------------------------------------------------------
class ViT_DEAM(nn.Module):
    """Two-stream DEAM-style model on top of a ViT features-only trunk with cross-attention fusion."""

    def __init__(self):
        super().__init__()

        # Backbone
        self.features = timm.create_model(
            "vit_base_patch32_384.augreg_in21k_ft_in1k",
            pretrained=True,
            features_only=True,
            img_size=image_size,
        )

        last_channel = 768

        # MBConv configs per stage (kept identical to the original code)
        efficientnet_config = [
            # e, k, s,        in,          out,      xN,  se,   fused,  act
            [3, 3, 1, last_channel, last_channel, 1, True, False, "GE"],  # stage 1
            [3, 3, 1, last_channel, last_channel, 1, True, False, "GE"],  # stage 2
            [3, 3, 1, last_channel, last_channel, 1, True, False, "GE"],  # stage 3
            [3, 3, 1, last_channel, last_channel, 2, True, False, "GE"],  # stage 4
            [3, 3, 1, last_channel, last_channel, 2, True, False, "GE"],  # stage 5
        ]

        self.conv1_x1 = RepeatedMBConv(MBConvConfig(*efficientnet_config[0]))
        self.conv1_x2 = RepeatedMBConv(MBConvConfig(*efficientnet_config[0]))

        self.conv2_x1 = RepeatedMBConv(MBConvConfig(*efficientnet_config[1]))
        self.conv2_x2 = RepeatedMBConv(MBConvConfig(*efficientnet_config[1]))

        self.conv3_x1 = RepeatedMBConv(MBConvConfig(*efficientnet_config[2]))
        self.conv3_x2 = RepeatedMBConv(MBConvConfig(*efficientnet_config[2]))

        self.conv4_x1 = RepeatedMBConv(MBConvConfig(*efficientnet_config[3]))
        self.conv4_x2 = RepeatedMBConv(MBConvConfig(*efficientnet_config[3]))

        self.conv5_x1 = RepeatedMBConv(MBConvConfig(*efficientnet_config[4]))
        self.conv5_x2 = RepeatedMBConv(MBConvConfig(*efficientnet_config[4]))

        # Cross attention blocks between the two streams
        dims = [128, 64, 64, 64, 64]
        L = [1, 2, 2, 2, 2]

        self.cross_attention1 = MobileCrossViTBlockv2(last_channel, dims[0], 2, L[0], (2, 2))
        self.cross_attention2 = MobileCrossViTBlockv2(last_channel, dims[1], 2, L[1], (2, 2))
        self.cross_attention3 = MobileCrossViTBlockv2(last_channel, dims[2], 2, L[2], (2, 2))
        self.cross_attention4 = MobileCrossViTBlockv2(last_channel, dims[3], 2, L[3], (2, 2))
        self.cross_attention5 = MobileCrossViTBlockv2(last_channel, dims[4], 2, L[4], (2, 2))

        # Final pooling + heads
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.last_block = LastBlock(last_channel)

    def forward(self, img1: torch.Tensor, img2: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x1 = self.features(img1)
        x2 = self.features(img2)

        x1 = x1[-1]
        x2 = x2[-1]

        res_x1 = x1
        res_x2 = x2
        x1 = self.conv1_x1(x1)
        x2 = self.conv1_x2(x2)
        x1, x2 = self.cross_attention1(x1, x2)
        x1 = x1 + res_x1
        x2 = x2 + res_x2

        x = torch.cat((x1, x2), dim=1)
        x = self.pool(x)

        x1, x2, intermediate = self.last_block(x)
        return x1, x2, intermediate


def define_model(device: torch.device | str = "cpu") -> ViT_DEAM:
    """Helper that constructs ViT_DEAM and moves it to the given device."""
    model = ViT_DEAM()
    return model.to(device)


__all__ = ["ViT_DEAM", "define_model", "image_size"]
