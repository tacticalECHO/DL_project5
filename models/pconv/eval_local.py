"""
Local evaluation of the trained partial-conv model.

Computes MSE / SSIM / LPIPS on the test set under three mask regimes
(small, medium, large), using the fixed test masks from data/.

Run from the repo root:
  python models/pconv/eval_local.py                 # full test set (~10 min)
  python models/pconv/eval_local.py --n_eval 50     # quick sanity check
"""
import os
import sys
import argparse
import time

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import torch
from torch.utils.data import DataLoader, Subset
import numpy as np
from skimage.metrics import structural_similarity as ssim_fn
import lpips

from utils.dataset import FixedMaskInpaintingDataset
from models.pconv.inpaint import load_model, inpaint


def load_paths(txt_path):
    with open(txt_path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def to_numpy_img(t):
    """[3,H,W] in [0,1] -> [H,W,3] float32 numpy."""
    return t.detach().cpu().permute(1, 2, 0).numpy().astype(np.float32)


def compute_metrics(model, dataset, device, lpips_fn, batch_size=8):
    """
    Evaluate the model and return averaged MSE / SSIM / LPIPS.
    Computed on the composite (observed + inpainted) vs ground truth.
    """
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        num_workers=0, pin_memory=True)

    mse_sum = 0.0
    ssim_sum = 0.0
    lpips_sum = 0.0
    n = 0

    model.eval()
    with torch.no_grad():
        for batch in loader:
            gt = batch['gt'].to(device)
            hole_mask = batch['mask'].to(device)
            masked = batch['masked'].to(device)

            out = inpaint(model, masked, hole_mask)

            # MSE
            mse_per_img = ((out - gt) ** 2).mean(dim=[1, 2, 3])
            mse_sum += mse_per_img.sum().item()

            # SSIM (CPU, per-image)
            for i in range(gt.shape[0]):
                gt_np = to_numpy_img(gt[i])
                out_np = to_numpy_img(out[i])
                s = ssim_fn(gt_np, out_np, data_range=1.0, channel_axis=2)
                ssim_sum += s

            # LPIPS expects inputs in [-1, 1]
            gt_lp = gt * 2 - 1
            out_lp = out * 2 - 1
            lpips_vals = lpips_fn(out_lp, gt_lp)
            lpips_sum += lpips_vals.sum().item()

            n += gt.shape[0]

    return {
        'MSE': mse_sum / n,
        'SSIM': ssim_sum / n,
        'LPIPS': lpips_sum / n,
        'count': n,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str,
                        default='./checkpoints/pconv/best_model.pt')
    parser.add_argument('--test_paths', type=str,
                        default='./data/test_paths.txt')
    parser.add_argument('--image_size', type=int, default=256)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--n_eval', type=int, default=-1,
                        help='how many test images to use (-1 = all)')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"device: {device}")

    print(f"loading model from {args.checkpoint}")
    model = load_model(args.checkpoint, device=device)

    print("loading LPIPS (alex backbone)")
    lpips_fn = lpips.LPIPS(net='alex').to(device)

    test_paths = load_paths(args.test_paths)
    n_total = len(test_paths)
    n_use = n_total if args.n_eval <= 0 else min(args.n_eval, n_total)
    print(f"evaluating on {n_use} / {n_total} test images")

    mask_files = {
        'small':  './data/fixed_masks_small.pt',
        'medium': './data/fixed_masks_medium.pt',
        'large':  './data/fixed_masks_large.pt',
    }

    all_results = {}
    for regime, mask_path in mask_files.items():
        print(f"\n==== {regime} masks ====")
        t0 = time.time()

        # Build with the full set of paths so the count matches the mask file.
        dataset = FixedMaskInpaintingDataset(
            test_paths,
            fixed_masks_path=mask_path,
            image_size=args.image_size,
        )
        # Then take the prefix we actually want to evaluate on.
        if n_use < n_total:
            dataset = Subset(dataset, range(n_use))

        results = compute_metrics(model, dataset, device, lpips_fn,
                                   batch_size=args.batch_size)
        dt = time.time() - t0

        print(f"  images: {results['count']}")
        print(f"  MSE:    {results['MSE']:.6f}")
        print(f"  SSIM:   {results['SSIM']:.4f}")
        print(f"  LPIPS:  {results['LPIPS']:.4f}")
        print(f"  time:   {dt:.1f}s")

        all_results[regime] = results

    print("\n==== summary ====")
    print(f"{'regime':<10} {'MSE':>10} {'SSIM':>8} {'LPIPS':>8}")
    for regime, r in all_results.items():
        print(f"{regime:<10} {r['MSE']:>10.6f} {r['SSIM']:>8.4f} {r['LPIPS']:>8.4f}")


if __name__ == "__main__":
    main()
