"""Local, optional perceptual metrics for GT-mask editing evaluation."""
from __future__ import annotations

from pathlib import Path
from typing import Protocol

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


class ImageSimilarityMetric(Protocol):
    def similarity(self, prediction: Image.Image, target: Image.Image) -> float: ...


def mask_bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    """Return an exclusive `(left, top, right, bottom)` bbox for a non-empty mask."""
    values = np.asarray(mask, dtype=bool)
    if values.ndim != 2:
        raise ValueError(f"ROI mask must be HxW, got shape {values.shape}")
    rows, columns = np.where(values)
    if not len(rows):
        raise ValueError("ROI perceptual metrics require a non-empty GT edit mask")
    return int(columns.min()), int(rows.min()), int(columns.max() + 1), int(rows.max() + 1)


def padded_mask_bbox(mask: np.ndarray, *, padding_ratio: float) -> tuple[int, int, int, int]:
    """Expand the GT-mask bbox by a fractional context margin and clamp to image bounds."""
    if padding_ratio < 0:
        raise ValueError("roi_padding_ratio must be non-negative")
    values = np.asarray(mask, dtype=bool)
    left, top, right, bottom = mask_bbox(values)
    height, width = values.shape
    pad_x = int(np.ceil((right - left) * padding_ratio))
    pad_y = int(np.ceil((bottom - top) * padding_ratio))
    return (
        max(0, left - pad_x),
        max(0, top - pad_y),
        min(width, right + pad_x),
        min(height, bottom + pad_y),
    )


def paired_roi_crops(
    prediction: Image.Image, target: Image.Image, mask: np.ndarray, *, padding_ratio: float,
) -> tuple[Image.Image, Image.Image, tuple[int, int, int, int]]:
    """Crop prediction and target with exactly the same padded GT-mask bbox."""
    if prediction.size != target.size:
        raise ValueError(f"prediction/target geometry differs: {prediction.size} != {target.size}")
    if np.asarray(mask).shape != (prediction.height, prediction.width):
        raise ValueError("ROI mask/image geometry mismatch")
    box = padded_mask_bbox(mask, padding_ratio=padding_ratio)
    return prediction.crop(box), target.crop(box), box


def cosine_similarity(prediction_features: torch.Tensor, target_features: torch.Tensor) -> float:
    """L2-normalized cosine similarity for one paired image embedding."""
    prediction_features = prediction_features.reshape(prediction_features.shape[0], -1)
    target_features = target_features.reshape(target_features.shape[0], -1)
    if prediction_features.shape != target_features.shape:
        raise ValueError("prediction/target feature shapes differ")
    values = F.cosine_similarity(prediction_features.float(), target_features.float(), dim=-1)
    if values.numel() != 1:
        raise ValueError("image similarity expects exactly one prediction/target pair")
    return float(values.item())


def spatial_lpips_masked_mean(spatial_map: torch.Tensor, mask: np.ndarray) -> float:
    """Mean a spatial LPIPS map only over an effective GT pixel mask."""
    values = torch.as_tensor(spatial_map).float()
    if values.ndim == 3:
        values = values[:, None]
    if values.ndim != 4 or values.shape[0] != 1:
        raise ValueError("spatial LPIPS must have shape [1, 1, H, W]")
    mask_values = torch.as_tensor(np.asarray(mask, dtype=np.float32))[None, None]
    resized = F.interpolate(mask_values, size=values.shape[-2:], mode="nearest")[0, 0].bool()
    if not bool(resized.any()):
        raise ValueError("ROI LPIPS requires a non-empty GT edit mask after resize")
    return float(values[0, 0][resized].mean().item())


def _image_tensor(image: Image.Image, device: torch.device) -> torch.Tensor:
    values = torch.from_numpy(np.asarray(image.convert("RGB"), dtype=np.float32).copy())
    return values.permute(2, 0, 1).unsqueeze(0).to(device) / 127.5 - 1.0


def _ensure_local_lpips_backbone(net: str) -> None:
    """Reject a missing LPIPS trunk instead of allowing a random fallback/download."""
    from torchvision.models import AlexNet_Weights, SqueezeNet1_1_Weights, VGG16_Weights

    weights = {
        "alex": AlexNet_Weights.IMAGENET1K_V1,
        "vgg": VGG16_Weights.IMAGENET1K_V1,
        "squeeze": SqueezeNet1_1_Weights.IMAGENET1K_V1,
    }.get(net)
    if weights is None:
        raise ValueError("lpips_net must be one of: alex, vgg, squeeze")
    checkpoint = Path(torch.hub.get_dir()) / "checkpoints" / Path(weights.url).name
    if not checkpoint.is_file():
        raise RuntimeError(
            "LPIPS pretrained trunk weights are unavailable in the local torch cache: "
            f"{checkpoint}. Pre-populate the official pretrained checkpoint; random LPIPS backbones are not allowed."
        )


class LPIPSMetric:
    """Lazy LPIPS wrapper; importing this class does not import the optional package."""

    def __init__(self, *, net: str, device: torch.device) -> None:
        _ensure_local_lpips_backbone(net)
        try:
            import lpips
        except ImportError as error:  # pragma: no cover - depends on optional extra
            raise RuntimeError("LPIPS was requested; install the eval extra with `uv sync --extra eval`.") from error
        self.device = device
        try:
            self.model = lpips.LPIPS(net=net, spatial=True, pretrained=True, pnet_rand=False).to(device).eval()
        except Exception as error:  # pragma: no cover - depends on local weight cache
            raise RuntimeError("failed to load pretrained LPIPS weights from the local cache") from error

    @torch.inference_mode()
    def scores(self, prediction: Image.Image, target: Image.Image, mask: np.ndarray) -> dict[str, float]:
        values = self.model(_image_tensor(prediction, self.device), _image_tensor(target, self.device))
        return {
            "full_lpips": float(values.mean().item()),
            "roi_lpips": spatial_lpips_masked_mean(values, mask),
        }


class DINOImageMetric:
    """DINO/DINOv2 CLS-image similarity loaded only from a local model directory."""

    def __init__(self, model_path: Path, *, device: torch.device) -> None:
        from transformers import AutoImageProcessor, AutoModel

        self.device = device
        try:
            self.processor = AutoImageProcessor.from_pretrained(model_path, local_files_only=True)
            self.model = AutoModel.from_pretrained(model_path, local_files_only=True).to(device).eval()
        except OSError as error:
            raise RuntimeError(f"cannot load local DINO model from {model_path}") from error

    @torch.inference_mode()
    def similarity(self, prediction: Image.Image, target: Image.Image) -> float:
        inputs = self.processor(images=[prediction, target], return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(self.device)
        features = self.model(pixel_values=pixel_values).last_hidden_state[:, 0]
        return cosine_similarity(features[:1], features[1:])


class CLIPImageMetric:
    """CLIP image-image similarity loaded only from a local model directory."""

    def __init__(self, model_path: Path, *, device: torch.device) -> None:
        from transformers import AutoProcessor, CLIPModel

        self.device = device
        try:
            self.processor = AutoProcessor.from_pretrained(model_path, local_files_only=True)
            self.model = CLIPModel.from_pretrained(model_path, local_files_only=True).to(device).eval()
        except OSError as error:
            raise RuntimeError(f"cannot load local CLIP model from {model_path}") from error

    @torch.inference_mode()
    def similarity(self, prediction: Image.Image, target: Image.Image) -> float:
        inputs = self.processor(images=[prediction, target], return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(self.device)
        features = self.model.get_image_features(pixel_values=pixel_values)
        return cosine_similarity(features[:1], features[1:])


def image_similarity_metrics(
    metric: ImageSimilarityMetric,
    prediction: Image.Image,
    target: Image.Image,
    mask: np.ndarray,
    *,
    metric_name: str,
    roi_padding_ratio: float,
) -> dict[str, float]:
    """Calculate full and same-crop ROI similarity for a DINO- or CLIP-like metric."""
    prediction_roi, target_roi, _ = paired_roi_crops(
        prediction, target, mask, padding_ratio=roi_padding_ratio
    )
    return {
        f"full_{metric_name}": metric.similarity(prediction, target),
        f"roi_{metric_name}": metric.similarity(prediction_roi, target_roi),
    }
