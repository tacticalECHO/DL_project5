import argparse
import json
import os
import sys
from pathlib import Path

import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision.utils import make_grid, save_image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.dataset import FixedMaskInpaintingDataset
from cvae.model import CVAE


def load_paths(txt_path):
    with open(txt_path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]

@torch.no_grad()
def make_pluralistic_grid(
    model, device, save_path,
    n_images: int = 4, n_samples: int = 6,
    regime: str = "large",
):
    print(f"Generating pluralistic grid ({n_images} images × {n_samples} samples)...")

    dataset = FixedMaskInpaintingDataset(
        image_paths = load_paths("./data/val_paths.txt"),
        fixed_masks_path = f"./data/fixed_masks_val_{regime}.pt",
        image_size = 256,
    )

    indices = torch.linspace(0, len(dataset) - 1, n_images).long().tolist()

    rows = []
    for idx in indices:
        sample = dataset[idx]
        gt = sample["gt"].unsqueeze(0).to(device)
        masked = sample["masked"].unsqueeze(0).to(device)
        mask = sample["mask"].unsqueeze(0).to(device)

        samples = model.sample(masked, mask, n_samples=n_samples)
        samples = samples.squeeze(1)

        row = torch.cat([
            gt[0].unsqueeze(0), masked[0].unsqueeze(0), samples,
            ], dim=0)
        rows.append(row)

    all_imgs = torch.cat(rows, dim=0)
    grid = make_grid(
        all_imgs.clamp(0, 1),
        nrow=2 + n_samples,
        padding=2,
        pad_value=1.0,
    )

    save_path.parent.mkdir(parents=True, exist_ok=True)
    save_image(grid, save_path)
    print(f"Saved {save_path}")
    print(f"Columns: gt | masked | sample_1 ... sample_{n_samples}")
    print(f"Rows: {n_images} different val images")


def make_pd_plane(summary_json_path, save_path):
    print(f"Generating P-D plane from {summary_json_path}...")

    with open(summary_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    results = data["results"]

    regimes = [r["regime"] for r in results]
    mse = [r["mse"] for r in results]
    ssim = [r["ssim"] for r in results]
    lpips_v = [r["lpips"] for r in results]
    fid = [r["fid"] for r in results]

    one_minus_ssim = [1 - s for s in ssim]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax = axes[0]
    ax.plot(mse, fid, "o-", color="tab:blue", markersize=10, linewidth=2)
    for i, r in enumerate(regimes):
        ax.annotate(r, (mse[i], fid[i]),
                    textcoords="offset points", xytext=(8, 8), fontsize=11)
    ax.set_xlabel("MSE (distortion) →", fontsize=11)
    ax.set_ylabel("FID (perception) →", fontsize=11)
    ax.set_title("CVAE: MSE vs FID", fontsize=12)
    ax.grid(alpha=0.3)

    ax = axes[1]
    ax.plot(one_minus_ssim, lpips_v, "s-", color="tab:orange", markersize=10, linewidth=2)
    for i, r in enumerate(regimes):
        ax.annotate(r, (one_minus_ssim[i], lpips_v[i]),
                    textcoords="offset points", xytext=(8, 8), fontsize=11)
    ax.set_xlabel("1 − SSIM (distortion) →", fontsize=11)
    ax.set_ylabel("LPIPS (perception) →", fontsize=11)
    ax.set_title("CVAE: 1−SSIM vs LPIPS", fontsize=12)
    ax.grid(alpha=0.3)

    plt.suptitle("Perception-Distortion Plane — CVAE only (3 mask regimes)", fontsize=13)
    plt.tight_layout()

    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {save_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--summary-json", type=str, default="reports/cvae_summary.json")
    parser.add_argument("--output-dir", type=str, default="reports")
    parser.add_argument("--n-images", type=int, default=4)
    parser.add_argument("--n-samples", type=int, default=6)
    parser.add_argument("--regime", type=str, default="large")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Loading checkpoint: {args.ckpt}")
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    latent_dim = ckpt["args"].get("latent_dim", 256)
    model = CVAE(latent_dim=latent_dim).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    out_dir = Path(args.output_dir)

    # Pluralistic grid
    make_pluralistic_grid(
        model, device,
        save_path=out_dir / f"pluralistic_grid_{args.regime}.png",
        n_images=args.n_images, n_samples=args.n_samples,
        regime=args.regime,
    )

    # P-D plane
    make_pd_plane(
        summary_json_path=args.summary_json,
        save_path=out_dir / "pd_plane_cvae_only.png",
    )


if __name__ == "__main__":
    main()