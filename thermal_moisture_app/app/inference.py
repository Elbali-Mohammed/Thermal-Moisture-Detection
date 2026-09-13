"""Frozen inference pipeline for the thermal-moisture ensemble."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import SegformerForSemanticSegmentation
import segmentation_models_pytorch as smp


def load_config(config_path: str | Path) -> dict[str, Any]:
    with open(config_path, encoding="utf-8") as file:
        return json.load(file)


def _state_dict(checkpoint_path: Path) -> dict[str, torch.Tensor]:
    """Accept checkpoints saved as a state dict or in common wrapper dictionaries."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if isinstance(checkpoint, dict):
        for key in ("model_state_dict", "state_dict", "model"):
            if key in checkpoint and isinstance(checkpoint[key], dict):
                checkpoint = checkpoint[key]
                break

    if not isinstance(checkpoint, dict):
        raise TypeError(f"Unsupported checkpoint format: {checkpoint_path}")

    # Checkpoints saved with DataParallel prefix every key with "module.".
    return {
        key.removeprefix("module."): value
        for key, value in checkpoint.items()
    }


def _load_weights(model: torch.nn.Module, checkpoint_path: Path) -> None:
    model.load_state_dict(_state_dict(checkpoint_path), strict=True)
    model.eval()


def build_models(config: dict[str, Any], device: torch.device) -> dict[str, torch.nn.Module]:
    """Recreate the three architectures used by the final ensemble."""
    checkpoint_paths = {
        name: Path(path)
        for name, path in config["checkpoints"].items()
    }

    # This must match the original training notebook: one-logit binary DeepLabV3+.
    deeplab_r50 = smp.DeepLabV3Plus(
        encoder_name="resnet50",
        encoder_weights=None,
        in_channels=3,
        classes=1,
        activation=None,
    )

    # This must match the original training notebook: two semantic classes,
    # class 1 = humidity.
    segformer_b0 = SegformerForSemanticSegmentation.from_pretrained(
        "nvidia/mit-b0", num_labels=2, ignore_mismatched_sizes=True
    )
    segformer_b1 = SegformerForSemanticSegmentation.from_pretrained(
        "nvidia/mit-b1", num_labels=2, ignore_mismatched_sizes=True
    )

    models = {
        "deeplab_r50": deeplab_r50,
        "segformer_b0": segformer_b0,
        "segformer_b1": segformer_b1,
    }
    for name, model in models.items():
        _load_weights(model, checkpoint_paths[name])
        model.to(device)

    return models


def preprocess(image: Image.Image, config: dict[str, Any]) -> torch.Tensor:
    """Resize and normalize an RGB thermal rendering exactly once for all models."""
    image = image.convert("RGB").resize(
        (config["image_width"], config["image_height"]), Image.Resampling.BILINEAR
    )
    array = np.asarray(image, dtype=np.float32) / 255.0
    mean = np.asarray(config["normalization"]["mean"], dtype=np.float32)
    std = np.asarray(config["normalization"]["std"], dtype=np.float32)
    array = (array - mean) / std
    return torch.from_numpy(array.transpose(2, 0, 1)).unsqueeze(0)


@torch.inference_mode()
def predict(
    image: Image.Image,
    models: dict[str, torch.nn.Module],
    config: dict[str, Any],
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the frozen weighted probability map and final binary mask."""
    tensor = preprocess(image, config).to(device)
    target_size = tensor.shape[-2:]

    r50_probability = torch.sigmoid(models["deeplab_r50"](tensor))

    b0_logits = models["segformer_b0"](tensor).logits
    b0_logits = F.interpolate(b0_logits, size=target_size, mode="bilinear", align_corners=False)
    b0_probability = torch.softmax(b0_logits, dim=1)[:, 1:2]

    b1_logits = models["segformer_b1"](tensor).logits
    b1_logits = F.interpolate(b1_logits, size=target_size, mode="bilinear", align_corners=False)
    b1_probability = torch.softmax(b1_logits, dim=1)[:, 1:2]

    weights = config["ensemble"]
    probability = (
        weights["r50_weight"] * r50_probability
        + weights["b0_weight"] * b0_probability
        + weights["b1_weight"] * b1_probability
    )
    mask = probability >= weights["threshold"]

    return probability[0, 0].cpu().numpy(), mask[0, 0].cpu().numpy()
