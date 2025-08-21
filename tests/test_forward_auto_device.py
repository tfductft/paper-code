# tests/test_forward_auto_device.py
# SPDX-License-Identifier: MIT
#
# Minimal smoke tests for the ViT_DEAM forward pass.
# - Uses CPU by default, switches to CUDA if available
# - Checks output shapes, device placement, and finiteness
# - Includes a tiny backward pass to ensure the graph is valid

import sys
from pathlib import Path
import torch

# Add project root to import path (useful when IDE sets CWD to scripts/)
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from deam_texture_servo import define_model, image_size


def test_forward_auto_device():
    """Forward pass runs on CPU/GPU, returns expected shapes and finite values."""
    # Use cuda:0 explicitly if CUDA is available to avoid 'cuda' vs 'cuda:0' mismatch
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = define_model(device).eval()

    H, W = image_size  # image_size is (H, W)
    x1 = torch.randn(1, 3, H, W, device=device)
    x2 = torch.randn(1, 3, H, W, device=device)

    with torch.no_grad():
        y1, y2, inter = model(x1, x2)

    # Shape checks
    assert y1.shape == (1, 3), "Translation head must output [B, 3]"
    assert y2.shape == (1, 3), "Rotation head must output [B, 3]"
    assert inter.ndim == 2 and inter.shape[0] == 1 and inter.shape[1] > 0, "Intermediate must be [B, C]"

    # Device placement checks: compare tensor devices directly (robust to cuda vs cuda:0)
    assert y1.device == x1.device == x2.device == y2.device == inter.device, \
        f"Device mismatch: x1={x1.device}, y1={y1.device}, y2={y2.device}, inter={inter.device}"

    # Finiteness checks (no NaN/Inf)
    assert torch.isfinite(y1).all()
    assert torch.isfinite(y2).all()
    assert torch.isfinite(inter).all()


def test_backward_graph_integrity():
    """A tiny backward pass to ensure the computation graph is properly connected."""
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = define_model(device).train()

    H, W = image_size
    x1 = torch.randn(1, 3, H, W, device=device, requires_grad=True)
    x2 = torch.randn(1, 3, H, W, device=device, requires_grad=True)

    y1, y2, inter = model(x1, x2)
    # Simple scalar loss (L1 on outputs)
    loss = y1.abs().sum() + y2.abs().sum() + inter.abs().mean()
    loss.backward()  # should not raise

    # At least one parameter should have received a gradient
    grads = [p.grad for p in model.parameters() if p.requires_grad]
    assert any(g is not None for g in grads), "Expected at least one non-None gradient"
