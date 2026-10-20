from __future__ import annotations

import platform
import sys
from importlib.metadata import PackageNotFoundError, version

import torch


def package_version(name: str) -> str:
    try:
        return version(name)
    except PackageNotFoundError:
        return "NOT INSTALLED"


def main() -> None:
    print(f"Python executable: {sys.executable}")
    print(f"Python version   : {sys.version.split()[0]}")
    print(f"Platform         : {platform.platform()}")
    print(f"PyTorch          : {torch.__version__}")
    print(f"PyTorch CUDA     : {torch.version.cuda}")

    for name in ["torchvision", "numpy", "pandas", "matplotlib", "Pillow"]:
        print(f"{name:16s}: {package_version(name)}")
    available = torch.cuda.is_available()
    print(f"CUDA available   : {available}")

    if not available:
        print("GPU test         : SKIPPED")
        return

    properties = torch.cuda.get_device_properties(0)
    print(f"GPU              : {properties.name}")
    print(f"Compute capability: {properties.major}.{properties.minor}")
    print(f"GPU memory       : {properties.total_memory / 1024**3:.2f} GB")

    x = torch.randn(2048, 2048, device="cuda")
    result = (x @ x).abs().mean().item()
    print(f"GPU test         : OK ({result:.6f})")


if __name__ == "__main__":
    main()