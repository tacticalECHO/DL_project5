from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from PIL import Image


def mse_metric(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(pred, target, reduction="mean")


def psnr_metric(pred: torch.Tensor, target: torch.Tensor, max_val: float = 1.0) -> torch.Tensor:
    mse = mse_metric(pred, target).clamp_min(1e-12)
    return 10.0 * torch.log10(torch.tensor(max_val * max_val, device=pred.device) / mse)


def ssim_metric(pred: torch.Tensor, target: torch.Tensor, data_range: float = 1.0) -> torch.Tensor:
    if pred.shape != target.shape:
        raise ValueError(f"Expected matching shapes, got {pred.shape} and {target.shape}")

    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2

    mu_x = F.avg_pool2d(pred, kernel_size=3, stride=1, padding=1)
    mu_y = F.avg_pool2d(target, kernel_size=3, stride=1, padding=1)

    sigma_x = F.avg_pool2d(pred * pred, 3, 1, 1) - mu_x * mu_x
    sigma_y = F.avg_pool2d(target * target, 3, 1, 1) - mu_y * mu_y
    sigma_xy = F.avg_pool2d(pred * target, 3, 1, 1) - mu_x * mu_y

    numerator = (2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)
    denominator = (mu_x.square() + mu_y.square() + c1) * (sigma_x + sigma_y + c2)
    score = numerator / denominator.clamp_min(1e-12)
    return score.mean()


def save_image_tensor(img: torch.Tensor, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    array = (img.detach().cpu().clamp(0.0, 1.0).permute(1, 2, 0).numpy() * 255.0).round().astype("uint8")
    Image.fromarray(array).save(path)


@dataclass
class PerceptualMetricBundle:
    fid: Any
    lpips: Any


def build_perceptual_metrics(device: torch.device) -> PerceptualMetricBundle:
    try:
        from torchmetrics.image.fid import FrechetInceptionDistance
        from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
    except ImportError as exc:
        raise ImportError(
            "FID/LPIPS evaluation requires torchmetrics. Install it before running evaluate_gan.py."
        ) from exc

    fid = FrechetInceptionDistance(feature=2048, normalize=True).to(device)
    lpips = LearnedPerceptualImagePatchSimilarity(net_type="alex").to(device)
    return PerceptualMetricBundle(fid=fid, lpips=lpips)
