# SPDX-License-Identifier: MIT
import sys
from pathlib import Path
import torch
from thop import profile

# Add project root to import path (useful when IDE sets CWD to scripts/)
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from deam_texture_servo import define_model, image_size  # noqa: E402


def format_size(n: float) -> str:
    """
    Human-readable number using 1024 base (same style as the original code).
    e.g., 123.00000000 M, 3.00000000 G
    """
    units = [' ', 'K', 'M', 'G', 'T', 'P', 'E', 'Z', 'Y']
    power = 1024.0
    u = 0
    v = float(n)
    while v >= power and u < len(units) - 1:
        v /= power
        u += 1
    return f"{v:.8f} {units[u]}"


def tensor_shape_str(t: torch.Tensor) -> str:
    """Return tensor shape as a compact string, e.g. '1x3x224x224'."""
    return "x".join(str(s) for s in t.shape)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = define_model(device)

    # Dummy inputs: two tensors, shape (N, C, H, W)
    c = 3
    h, w = image_size
    x = torch.randn(1, c, h, w, device=device)

    model.eval()

    with torch.no_grad():
        flops, params = profile(model, inputs=(x, x), verbose=False)

    print("== Model Summary (FLOPs/Params) ==")
    print(f"Device: {device.type}")
    print(f"Input size  (HxW, C): ({h}, {w}), {c}")
    print(f"Input 1 shape: {tensor_shape_str(x)}")
    print(f"Input 2 shape: {tensor_shape_str(x)}")
    print(f"FLOPs:  {format_size(flops)}")
    print(f"Params: {format_size(params)}")


if __name__ == "__main__":
    main()