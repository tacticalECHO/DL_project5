from __future__ import annotations
import sys as _sys; from pathlib import Path as _Path; _sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader

from models import InpaintingGenerator, compose_inpainting
from utils.dataset import FixedMaskInpaintingDataset
from utils.io import load_paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize GAN inpainting results.")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--split-paths", type=str, default="data/test_paths.txt")
    parser.add_argument("--mask-path", type=str, required=True)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--num-samples", type=int, default=8)
    parser.add_argument("--output", type=str, default="")
    parser.add_argument("--title", type=str, default="")
    return parser.parse_args()


def tensor_to_image(x: torch.Tensor) -> torch.Tensor:
    return x.detach().cpu().clamp(0.0, 1.0).permute(1, 2, 0)


@torch.no_grad()
def collect_samples(
    generator: InpaintingGenerator,
    loader: DataLoader,
    device: torch.device,
    num_samples: int,
) -> list[dict[str, torch.Tensor | str]]:
    samples: list[dict[str, torch.Tensor | str]] = []
    for batch in loader:
        gt = batch["gt"].to(device)
        mask = batch["mask"].to(device)
        masked = batch["masked"].to(device)
        pred = generator(torch.cat([masked, mask], dim=1))
        completed = compose_inpainting(pred, masked, mask)

        for i in range(gt.size(0)):
            if len(samples) >= num_samples:
                return samples
            samples.append(
                {
                    "gt": gt[i].cpu(),
                    "mask": mask[i].cpu(),
                    "masked": masked[i].cpu(),
                    "completed": completed[i].cpu(),
                    "path": batch["path"][i],
                }
            )
    return samples


def render_grid(samples: list[dict[str, torch.Tensor | str]], title: str = "") -> plt.Figure:
    num_rows = len(samples)
    fig, axes = plt.subplots(num_rows, 4, figsize=(12, 3 * num_rows))
    if num_rows == 1:
        axes = [axes]

    column_titles = ["GT", "Mask", "Masked", "Completed"]
    for col, col_title in enumerate(column_titles):
        axes[0][col].set_title(col_title)

    for row, sample in enumerate(samples):
        gt = sample["gt"]
        mask = sample["mask"]
        masked = sample["masked"]
        completed = sample["completed"]
        path = Path(str(sample["path"])).name
        mask_ratio = float(torch.as_tensor(mask).mean().item())

        axes[row][0].imshow(tensor_to_image(torch.as_tensor(gt)))
        axes[row][1].imshow(torch.as_tensor(mask).squeeze(0), cmap="gray", vmin=0.0, vmax=1.0)
        axes[row][2].imshow(tensor_to_image(torch.as_tensor(masked)))
        axes[row][3].imshow(tensor_to_image(torch.as_tensor(completed)))

        axes[row][0].set_ylabel(f"{path}\nmask={mask_ratio:.3f}", rotation=0, labelpad=55, va="center")
        for col in range(4):
            axes[row][col].axis("off")

    if title:
        fig.suptitle(title, fontsize=14)
        fig.tight_layout(rect=(0, 0, 1, 0.97))
    else:
        fig.tight_layout()
    return fig


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)

    checkpoint = torch.load(args.checkpoint, map_location=device)
    generator = InpaintingGenerator().to(device)
    generator.load_state_dict(checkpoint["generator"])
    generator.eval()

    dataset = FixedMaskInpaintingDataset(
        load_paths(args.split_paths),
        fixed_masks_path=args.mask_path,
        image_size=args.image_size,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    samples = collect_samples(generator, loader, device, args.num_samples)
    title = args.title or f"GAN Inpainting Visualization | {Path(args.mask_path).name}"
    fig = render_grid(samples, title=title)

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=200, bbox_inches="tight")
        print(f"Saved visualization to {output_path}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
