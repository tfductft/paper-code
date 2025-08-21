#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-License-Identifier: MIT

from __future__ import annotations

from typing import TypeVar
import torch
import torch.nn as nn

T = TypeVar("T", bound=nn.Module)


class Conv2dWrapper(nn.Conv2d):
    """
    Thin wrapper around torch.nn.Conv2d that tolerates extra kwargs (e.g., temperature)
    and simply ignores them. Useful when mixing modules that may forward additional args.
    """

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:  # noqa: D401
        return super().forward(x)


class TempModule(nn.Module):
    """
    Base class for modules that accept a `temperature` argument in forward.
    Default behavior is identity; override in subclasses as needed.
    """

    def forward(self, x: torch.Tensor, temperature: float | None = None) -> torch.Tensor:  # noqa: D401
        return x


class BaseModel(TempModule):
    """
    Base model that keeps a reference to a convolution layer class (e.g., Conv2dWrapper).
    """

    def __init__(self, ConvLayer: type[nn.Conv2d] = nn.Conv2d):
        super().__init__()
        self.ConvLayer = ConvLayer


class TemperatureScheduler:
    """
    Linear temperature scheduler.

    Args:
        initial_value: value at epoch 1
        final_value: value at `final_epoch` (defaults to initial_value)
        final_epoch: last epoch to reach `final_value` (defaults to 1)
    """

    def __init__(self, initial_value: float, final_value: float | None = None, final_epoch: int | None = None):
        self.initial_value = float(initial_value)
        self.final_value = float(final_value if final_value is not None else initial_value)
        self.final_epoch = int(final_epoch if final_epoch is not None else 1)

        if self.final_epoch <= 1:
            self.step = 0.0
        else:
            self.step = (self.final_value - self.initial_value) / (self.final_epoch - 1)

    def get(self, crt_epoch: int | None = None) -> float:
        """
        Get temperature for the given epoch (1-based). If not provided, returns the
        value at `final_epoch`.
        """
        e = int(crt_epoch if crt_epoch is not None else self.final_epoch)
        e = max(1, min(e, self.final_epoch))
        return self.initial_value + (e - 1) * self.step


class CustomSequential(TempModule):
    """
    Sequential container that forwards `temperature` only to TempModule children.

    Example:
        seq = CustomSequential(Conv2dWrapper(...), MyTempBlock(...))
        y = seq(x, temperature=0.5)  # passed to MyTempBlock, ignored by Conv2dWrapper
    """

    def __init__(self, *layers: nn.Module):
        super().__init__()
        self.layers = nn.ModuleList(layers)

    def forward(self, x: torch.Tensor, temperature: float | None = None) -> torch.Tensor:
        for layer in self.layers:
            if isinstance(layer, TempModule):
                x = layer(x, temperature)
            else:
                x = layer(x)
        return x

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            return CustomSequential(*list(self.layers)[idx])
        return self.layers[idx]


class SmoothNLLLoss(nn.Module):
    """
    Label-smoothed negative log-likelihood loss.

    Notes:
        - `prediction` is expected to be **log-probabilities** (i.e., output of log_softmax).
        - `target` contains class indices with shape (N,).
        - `dim` is the class dimension in `prediction` (default: last dim).
    """

    def __init__(self, smoothing: float = 0.0, dim: int = -1):
        super().__init__()
        self.smoothing = float(smoothing)
        self.dim = int(dim)

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:  # noqa: D401
        with torch.no_grad():
            n_class = prediction.size(self.dim)
            smooth_target = torch.full_like(prediction, self.smoothing / (n_class - 1))

            scatter_dim = self.dim if self.dim >= 0 else prediction.dim() + self.dim
            index = target.unsqueeze(scatter_dim)
            smooth_target.scatter_(scatter_dim, index, 1.0 - self.smoothing)

        loss = torch.sum(-smooth_target * prediction, dim=self.dim)
        return loss.mean()


__all__ = [
    "Conv2dWrapper",
    "TempModule",
    "BaseModel",
    "TemperatureScheduler",
    "CustomSequential",
    "SmoothNLLLoss",
]
