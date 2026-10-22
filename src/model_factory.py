from __future__ import annotations

from pathlib import Path

import torch
from torch import nn
from torchvision.models import resnet18

from src.resnet import ResNet18
from src.simple_cnn import SimpleCNN


def build_pretrained_architecture() -> nn.Module:
    model = resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 10)
    return model


BUILDERS = {
    "SimpleCNN": lambda: SimpleCNN(num_classes=10),
    "ResNet18FromScratch": lambda: ResNet18(num_classes=10),
    "ResNet18TransferLearning": build_pretrained_architecture,
}


def load_model(
    checkpoint_path: Path,
    device: torch.device,
) -> tuple[nn.Module, dict]:
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=True,
    )

    model_name = checkpoint["model_name"]
    if model_name not in BUILDERS:
        raise ValueError(f"Unsupported model type: {model_name}")

    model = BUILDERS[model_name]().to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model, checkpoint