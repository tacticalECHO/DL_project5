import argparse
import csv
import json
import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchmetrics.image import StructuralSimilarityIndexMeasure, PeakSignalNoiseRatio
from torchmetrics.image.fid import FrechetInceptionDistance
import lpips
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.dataset import FixedMaskInpaintingDataset
from cvae.model import CVAE


def load_paths(txt_path):
    with open(txt_path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


@torch.no_grad()
def evaluate_regime(
    model, loader, device,
    lpips_fn, regime, method_name="cvae",
):
    model.eval()

    ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    psnr_metric = PeakSignalNoiseRatio(data_range=1.0).to(device)
    fid_metric = FrechetInceptionDistance(feature=2048, normalize=True).to(device)

    per_image_rows = []

    for batch in tqdm(loader, desc=f"[eval {regime}]"):
        gt = batch["gt"].to(device, non_blocking=True)
        masked = batch["masked"].to(device, non_blocking=True)
        mask = batch["mask"].to(device, non_blocking=True)
        paths = batch["path"]

        x_hat, _ = model(gt, masked, mask)
        x_hat = x_hat.clamp(0, 1)

        B = gt.shape[0]

        for i in range(B):
            gt_i = gt[i:i+1]
            xh_i = x_hat[i:i+1]
            mask_i = mask[i:i+1]

            # MSE
            mse = F.mse_loss(xh_i, gt_i).item()

            # SSIM 
            ssim_metric.reset()
            ssim = ssim_metric(xh_i, gt_i).item()

            # PSNR
            psnr_metric.reset()
            psnr = psnr_metric(xh_i, gt_i).item()

            # LPIPS
            lpips_val = lpips_fn(
                xh_i * 2 - 1, gt_i * 2 - 1
            ).item()

            per_image_rows.append({
                "path":        paths[i],
                "regime":      regime,
                "mask_ratio":  mask_i.mean().item(),
                "mse":         mse,
                "ssim":        ssim,
                "psnr":        psnr,
                "lpips":       lpips_val,
            })

        fid_metric.update(gt, real=True)
        fid_metric.update(x_hat, real=False)

    fid_value = fid_metric.compute().item()

    n = len(per_image_rows)
    summary = {
        "method": method_name,
        "regime": regime,
        "n_images": n,
        "mse": sum(r["mse"]   for r in per_image_rows) / n,
        "ssim": sum(r["ssim"]  for r in per_image_rows) / n,
        "psnr": sum(r["psnr"]  for r in per_image_rows) / n,
        "lpips": sum(r["lpips"] for r in per_image_rows) / n,
        "fid": fid_value,
    }

    return per_image_rows, summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--run-name", type=str, default="run_01")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--method-name", type=str, default="cvae")
    parser.add_argument("--output-dir", type=str, default="reports")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")


    print(f"Loading checkpoint: {args.ckpt}")
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    latent_dim = ckpt["args"].get("latent_dim", 256)
    model = CVAE(latent_dim=latent_dim).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"loaded epoch {ckpt['epoch']}, val_l1={ckpt['val_l1']:.4f}")

    print("Loading LPIPS...")
    lpips_fn = lpips.LPIPS(net="alex").to(device)
    lpips_fn.eval()

    all_per_image = []
    all_summary = []

    for regime in ["small", "medium", "large"]:
        dataset = FixedMaskInpaintingDataset(
            image_paths=load_paths("./data/test_paths.txt"),
            fixed_masks_path=f"./data/fixed_masks_{regime}.pt",
            image_size=256,
        )
        loader = DataLoader(
            dataset, batch_size=args.batch_size, shuffle=False,
            num_workers=args.num_workers, pin_memory=True,
        )
        print(f"Regime: {regime} ({len(dataset)} images)")

        per_img, summary = evaluate_regime(
            model, loader, device, lpips_fn, regime, method_name=args.method_name,
        )
        all_per_image.extend(per_img)
        all_summary.append(summary)

        print(
            f"MSE={summary['mse']:.6f}  "
            f"SSIM={summary['ssim']:.4f}  "
            f"PSNR={summary['psnr']:.2f}  "
            f"LPIPS={summary['lpips']:.4f}  "
            f"FID={summary['fid']:.2f}"
        )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = out_dir / f"{args.method_name}_per_image.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["path", "regime", "mask_ratio", "mse", "ssim", "psnr", "lpips"]
        )
        writer.writeheader()
        writer.writerows(all_per_image)
    print(f"\nPer-image CSV to {csv_path}  ({len(all_per_image)} rows)")

    json_path = out_dir / f"{args.method_name}_summary.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({
            "method": args.method_name,
            "ckpt": args.ckpt,
            "epoch": ckpt["epoch"],
            "results":  all_summary,
        }, f, indent=2)
    print(f"Summary JSON to {json_path}")

    print("\n")
    print("Summary:")
    print(f"{'regime':<8s} {'MSE':>10s} {'PSNR':>8s} {'SSIM':>8s} {'LPIPS':>8s} {'FID':>8s}")
    for s in all_summary:
        print(
            f"{s['regime']:<8s} "
            f"{s['mse']:>10.6f} "
            f"{s['psnr']:>8.2f} "
            f"{s['ssim']:>8.4f} "
            f"{s['lpips']:>8.4f} "
            f"{s['fid']:>8.2f}"
        )

if __name__ == "__main__":
    main()