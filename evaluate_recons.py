from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from models.inpaint import inpaint, load_model
from utils.dataset import FixedMaskInpaintingDataset
from utils.io import load_paths
from utils.metrics import build_perceptual_metrics, mse_metric, psnr_metric, save_image_tensor, ssim_metric


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate the reconstruction-based inpainting model.")
    parser.add_argument("--checkpoint", type=str, default="outputs/rcon/best_model.pt")
    parser.add_argument("--split-paths", type=str, default="data/test_paths.txt")
    parser.add_argument("--small-mask-path", type=str, default="data/fixed_masks_small.pt")
    parser.add_argument("--medium-mask-path", type=str, default="data/fixed_masks_medium.pt")
    parser.add_argument("--large-mask-path", type=str, default="data/fixed_masks_large.pt")
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--save-samples-dir", type=str, default="")
    parser.add_argument("--save-csv-dir", type=str, default="outputs/rcon/eval_csv")
    parser.add_argument("--max-save-samples", type=int, default=16)
    return parser.parse_args()


@torch.no_grad()
def evaluate_regime(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    sample_dir: Path | None,
    max_save_samples: int,
) -> tuple[dict[str, float], list[dict[str, float | str]]]:
    metric_bundle = build_perceptual_metrics(device)
    total_mse = 0.0
    total_ssim = 0.0
    total_psnr = 0.0
    total_lpips = 0.0
    total = 0
    saved = 0
    per_image_rows: list[dict[str, float | str]] = []

    for batch in loader:
        gt = batch["gt"].to(device)
        mask = batch["mask"].to(device)
        masked = batch["masked"].to(device)

        completed = inpaint(model, masked, mask)
        batch_size = gt.size(0)

        total_mse += mse_metric(completed, gt).item() * batch_size
        total_ssim += ssim_metric(completed, gt).item() * batch_size
        total_psnr += psnr_metric(completed, gt).item() * batch_size

        metric_bundle.fid.update(gt, real=True)
        metric_bundle.fid.update(completed, real=False)
        lpips_batch = metric_bundle.lpips(completed * 2.0 - 1.0, gt * 2.0 - 1.0)
        total_lpips += lpips_batch.item() * batch_size
        total += batch_size

        for i in range(batch_size):
            gt_i = gt[i : i + 1]
            mask_i = mask[i : i + 1]
            completed_i = completed[i : i + 1]
            lpips_i = metric_bundle.lpips(completed_i * 2.0 - 1.0, gt_i * 2.0 - 1.0)
            per_image_rows.append(
                {
                    "path": batch["path"][i],
                    "mask_ratio": float(mask_i.mean().item()),
                    "mse": float(mse_metric(completed_i, gt_i).item()),
                    "ssim": float(ssim_metric(completed_i, gt_i).item()),
                    "psnr": float(psnr_metric(completed_i, gt_i).item()),
                    "lpips": float(lpips_i.item()),
                }
            )

        if sample_dir is not None and saved < max_save_samples:
            for i in range(batch_size):
                if saved >= max_save_samples:
                    break
                save_image_tensor(masked[i], sample_dir / f"{saved:03d}_masked.png")
                save_image_tensor(completed[i], sample_dir / f"{saved:03d}_completed.png")
                save_image_tensor(gt[i], sample_dir / f"{saved:03d}_gt.png")
                saved += 1

    return (
        {
            "mse": total_mse / max(total, 1),
            "ssim": total_ssim / max(total, 1),
            "psnr": total_psnr / max(total, 1),
            "lpips": total_lpips / max(total, 1),
            "fid": float(metric_bundle.fid.compute().item()),
        },
        per_image_rows,
    )


def build_loader(split_paths: str, fixed_mask_path: str, image_size: int, batch_size: int, num_workers: int) -> DataLoader:
    dataset = FixedMaskInpaintingDataset(
        load_paths(split_paths),
        fixed_masks_path=fixed_mask_path,
        image_size=image_size,
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)


def save_per_image_csv(csv_path: Path, rows: list[dict[str, float | str]]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["path", "regime", "mask_ratio", "mse", "ssim", "psnr", "lpips"]
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)

    model = load_model(args.checkpoint, device=device)

    loaders = {
        "small": build_loader(args.split_paths, args.small_mask_path, args.image_size, args.batch_size, args.num_workers),
        "medium": build_loader(args.split_paths, args.medium_mask_path, args.image_size, args.batch_size, args.num_workers),
        "large": build_loader(args.split_paths, args.large_mask_path, args.image_size, args.batch_size, args.num_workers),
    }

    save_root = Path(args.save_samples_dir) if args.save_samples_dir else None
    csv_root = Path(args.save_csv_dir) if args.save_csv_dir else None
    results = {}
    all_rows: list[dict[str, float | str]] = []
    for regime, loader in loaders.items():
        sample_dir = save_root / regime if save_root is not None else None
        regime_metrics, regime_rows = evaluate_regime(model, loader, device, sample_dir, args.max_save_samples)
        results[regime] = regime_metrics
        for row in regime_rows:
            row["regime"] = regime
        all_rows.extend(regime_rows)
        if csv_root is not None:
            save_per_image_csv(csv_root / f"{regime}_per_image_metrics.csv", regime_rows)

    report = {
        "checkpoint": args.checkpoint,
        "method": "reconstruction_based_partial_conv",
        "conditioning_strategy": {
            "generator_input": "(masked_image, mask)",
            "reason": "Partial convolutions explicitly use the binary hole mask to distinguish missing"
            " pixels from observed context, which matches the training objective and evaluation protocol.",
        },
        "perception_metric_note": "LPIPS is reported as the extra perception metric because it better tracks"
        " perceptual similarity than pixel-wise errors.",
        "per_image_csv_note": "Per-image CSVs include MSE, SSIM, PSNR, and LPIPS. FID is only reported at the"
        " dataset level because it is a distribution-level metric rather than a per-image metric.",
        "results": results,
    }
    if csv_root is not None:
        save_per_image_csv(csv_root / "all_regimes_per_image_metrics.csv", all_rows)
        report["per_image_csv_dir"] = str(csv_root)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
