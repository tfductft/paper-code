DEAM Texture Servo (PyTorch)

Transformer-driven visual servoing with dual-arm impedance control for fabric texture matching.
This repository provides the reference PyTorch implementation of ViT_DEAM, a two-stream model that uses a ViT (features-only) backbone and our Difference Extraction Attention Module (DEAM) with DCAB to predict pose deltas from a pair of images (desired vs. current).

Paper (PDF in repo): Transformer_Driven_Visual_Servoing_and_Dual_Arm_Impedance_Control_for_Fabric_Texture_Matching.pdf

DOI: 10.36227/techrxiv.175493452.28291434/v1

⚠️ This code focuses on the model, forward pass, and profiling. Training/control-loop code is intentionally minimal to keep the package lightweight for reproducibility.

Repository layout
paper-code/
├─ README.md
├─ LICENSE
├─ CITATION.cff                      # (optional but recommended)
├─ requirements.txt
├─ Transformer_..._Matching.pdf      # paper
│
├─ deam_texture_servo/               # Python package
│  ├─ __init__.py                    # exports ViT_DEAM, define_model, image_size
│  ├─ model.py                       # ViT_DEAM model & FLOPs helpers
│  └─ utils_dynamicConv.py           # TempModule & dynamic conv utilities
│
├─ scripts/
│  ├─ print_summary.py               # FLOPs/Params (THOP), input/output shapes, timing
│  └─ demo_infer.py                  # tiny inference demo (optional)
│
└─ tests/
   └─ test_forward_auto_device.py    # smoke test (CPU/GPU auto), tiny backward

Installation

We recommend a fresh virtual environment.

cd paper-code

# (1) Create & activate an environment (example with conda)
conda create -n deam python=3.12 -y
conda activate deam

# (2) Install PyTorch for your platform (choose CUDA/CPU build accordingly)
# See https://pytorch.org/get-started/locally/
# Example: CUDA 12.4 wheels
pip install torch==2.4.0 torchvision==0.19.0 --index-url https://download.pytorch.org/whl/cu124
# Example: CPU-only wheels
# pip install torch==2.4.0 torchvision==0.19.0 --index-url https://download.pytorch.org/whl/cpu

# (3) Install project dependencies
pip install -r requirements.txt


MKL/OpenMP note (Anaconda/Linux):
If you see MKL_THREADING_LAYER=INTEL is incompatible with libgomp.so.1, run:

export MKL_THREADING_LAYER=GNU
unset MKL_SERVICE_FORCE_INTEL

Quick start: inference
import torch
from deam_texture_servo import define_model, image_size

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = define_model(device).eval()

H, W = image_size  # (height, width)
x_desired = torch.randn(1, 3, H, W, device=device)
x_current = torch.randn(1, 3, H, W, device=device)

with torch.no_grad():
    d_trans, d_rot, features = model(x_desired, x_current)

print(d_trans.shape, d_rot.shape, features.shape)
# -> torch.Size([1, 3]) torch.Size([1, 3]) torch.Size([1, C])

Model summary & FLOPs

We provide a script that reproduces the legacy FLOPs/params calculation you used previously (same thop.profile(model, inputs=(x, x)) call).

python scripts/print_summary.py


Example output:

== Model Summary (Legacy-compatible FLOPs) ==
Input size  (HxW, C): (540, 960), 3
FLOPs: 81.23456789 M
Params: 23.45678901 M
Forward time (1x): 0.1234 s on cuda:0

Tests

Minimal smoke tests (forward on CPU/GPU, device checks, finiteness, and a tiny backward):

pytest -q
# or target a single file:
pytest -q tests/test_forward_auto_device.py


Force CPU only:

CUDA_VISIBLE_DEVICES="" pytest -q

What’s inside ViT_DEAM (high level)

Backbone: timm ViT vit_base_patch32_384.augreg_in21k_ft_in1k with features_only=True.

Two streams: process desired/current features.

Cross fusion: MobileCrossViT-style linear cross attention over patchwise features.

DCAB block: lightweight channel attention (ECA) + dynamic depthwise convolution + LN + GELU with residual.

Heads: translation and rotation regressors (each 3-DoF) on pooled fused features.

The public API is kept small and stable:

from deam_texture_servo import ViT_DEAM, define_model, image_size

Weights

Pretrained weights are not included. If you plan to distribute large .pt/.pth files, consider Git LFS
:

git lfs install
git lfs track "*.pt" "*.pth"

Citation

If you find this useful, please cite the paper:

Plain text

Tokuda, F., et al. "Transformer-Driven Visual Servoing and Dual-Arm Impedance Control for Fabric Texture Matching."
TechRxiv (2025). DOI: 10.36227/techrxiv.175493452.28291434/v1


BibTeX

@misc{tokuda2025deam,
  title        = {Transformer-Driven Visual Servoing and Dual-Arm Impedance Control for Fabric Texture Matching},
  author       = {Tokuda, F. and collaborators},
  year         = {2025},
  howpublished = {TechRxiv},
  doi          = {10.36227/techrxiv.175493452.28291434/v1}
}


You may also cite this repository:

@software{deam_texture_servo_code,
  title  = {DEAM Texture Servo (PyTorch) — ViT\_DEAM},
  author = {Tokuda, F.},
  year   = {2025},
  url    = {https://github.com/<your_org_or_user>/paper-code}
}

License

Released under the MIT License. See LICENSE for details.
Please also comply with licenses of third-party dependencies (e.g., timm, thop).

Acknowledgements

This codebase builds upon the excellent open-source ecosystem, including:

timm
 (ViT features-only backbone)

thop
 (FLOPs/params profiling)

Troubleshooting

ModuleNotFoundError: deam_texture_servo
Run scripts from the repo root (paper-code/) so Python can resolve the package, or install in editable mode:

pip install -e .


CUDA vs cuda:0 device mismatch in tests
The tests choose cuda:0 explicitly to avoid cuda/cuda:0 comparison issues.

MKL / libgomp
See the environment note in Installation.
