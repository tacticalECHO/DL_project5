from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader

from models import InpaintingDiscriminator, InpaintingGenerator, compose_inpainting
from utils.dataset import FixedMaskInpaintingDataset, InpaintingTrainDataset
from utils.io import load_paths
from utils.metrics import mse_metric, ssim_metric


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a conditional GAN for face inpainting.")
    parser.add_argument("--train-paths", type=str, default="data/train_paths.txt")
    parser.add_argument("--val-paths", type=str, default="data/val_paths.txt")
    parser.add_argument("--val-mask-path", type=str, default="data/fixed_masks_val_medium.pt")
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--lambda-adv", type=float, default=0.1)
    parser.add_argument("--lambda-rec", type=float, default=10.0)
    parser.add_argument("--lambda-hole", type=float, default=20.0)
    parser.add_argument("--lambda-valid", type=float, default=5.0)
    parser.add_argument("--mixed-probs", type=float, nargs=3, default=(1 / 3, 1 / 3, 1 / 3))
    parser.add_argument("--output-dir", type=str, default="outputs/gan")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-every", type=int, default=5)
    parser.add_argument("--log-every", type=int, default=100)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_loaders(args: argparse.Namespace) -> tuple[DataLoader, DataLoader]:
    train_dataset = InpaintingTrainDataset(
        load_paths(args.train_paths),
        image_size=args.image_size,
        regime="mixed",
        mixed_probs=tuple(args.mixed_probs),
    )
    val_dataset = FixedMaskInpaintingDataset(
        load_paths(args.val_paths),
        fixed_masks_path=args.val_mask_path,
        image_size=args.image_size,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=args.device.startswith("cuda"),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.device.startswith("cuda"),
    )
    return train_loader, val_loader


def discriminator_hinge_loss(real_logits: torch.Tensor, fake_logits: torch.Tensor) -> torch.Tensor:
    return F.relu(1.0 - real_logits).mean() + F.relu(1.0 + fake_logits).mean()


def generator_hinge_loss(fake_logits: torch.Tensor) -> torch.Tensor:
    return -fake_logits.mean()


@torch.no_grad()
def evaluate_generator(
    generator: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> dict[str, float]:
    generator.eval()
    total_mse = 0.0
    total_ssim = 0.0
    total_samples = 0
    for batch in loader:
        gt = batch["gt"].to(device)
        mask = batch["mask"].to(device)
        masked = batch["masked"].to(device)
        pred = generator(torch.cat([masked, mask], dim=1))
        completed = compose_inpainting(pred, masked, mask)
        batch_size = gt.size(0)
        total_mse += mse_metric(completed, gt).item() * batch_size
        total_ssim += ssim_metric(completed, gt).item() * batch_size
        total_samples += batch_size
    return {
        "mse": total_mse / max(total_samples, 1),
        "ssim": total_ssim / max(total_samples, 1),
    }


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    train_loader, val_loader = build_loaders(args)

    generator = InpaintingGenerator().to(device)
    discriminator = InpaintingDiscriminator().to(device)

    opt_g = torch.optim.Adam(generator.parameters(), lr=args.lr, betas=(0.5, 0.999))
    opt_d = torch.optim.Adam(discriminator.parameters(), lr=args.lr, betas=(0.5, 0.999))

    history: list[dict[str, float]] = []
    best_val = float("inf")

    for epoch in range(1, args.epochs + 1):
        generator.train()
        discriminator.train()
        epoch_start = time.time()

        running_g = 0.0
        running_d = 0.0
        num_batches = 0
        total_batches = len(train_loader)

        print(
            f"[Epoch {epoch}/{args.epochs}] start | "
            f"train_batches={total_batches} | val_batches={len(val_loader)} | device={device}",
            flush=True,
        )

        for batch_idx, batch in enumerate(train_loader, start=1):
            gt = batch["gt"].to(device)
            mask = batch["mask"].to(device)
            masked = batch["masked"].to(device)

            cond = torch.cat([masked, mask], dim=1)
            pred = generator(cond)
            completed = compose_inpainting(pred, masked, mask)

            opt_d.zero_grad(set_to_none=True)
            real_logits = discriminator(gt, masked, mask)
            fake_logits = discriminator(completed.detach(), masked, mask)
            d_loss = discriminator_hinge_loss(real_logits, fake_logits)
            d_loss.backward()
            opt_d.step()

            opt_g.zero_grad(set_to_none=True)
            fake_logits_for_g = discriminator(completed, masked, mask)
            g_adv = generator_hinge_loss(fake_logits_for_g)
            g_rec = F.l1_loss(pred, gt)
            g_hole = F.l1_loss(pred * mask, gt * mask)
            g_valid = F.l1_loss(pred * (1.0 - mask), gt * (1.0 - mask))
            g_loss = (
                args.lambda_adv * g_adv
                + args.lambda_rec * g_rec
                + args.lambda_hole * g_hole
                + args.lambda_valid * g_valid
            )
            g_loss.backward()
            opt_g.step()

            running_g += g_loss.item()
            running_d += d_loss.item()
            num_batches += 1

            if batch_idx == 1 or batch_idx % args.log_every == 0 or batch_idx == total_batches:
                elapsed = time.time() - epoch_start
                avg_g = running_g / num_batches
                avg_d = running_d / num_batches
                print(
                    f"[Epoch {epoch}/{args.epochs}] "
                    f"batch {batch_idx}/{total_batches} | "
                    f"g_loss={avg_g:.4f} | d_loss={avg_d:.4f} | "
                    f"elapsed={elapsed / 60.0:.1f}m",
                    flush=True,
                )

        val_metrics = evaluate_generator(generator, val_loader, device)
        epoch_time = time.time() - epoch_start
        epoch_log = {
            "epoch": epoch,
            "g_loss": running_g / max(num_batches, 1),
            "d_loss": running_d / max(num_batches, 1),
            "val_mse": val_metrics["mse"],
            "val_ssim": val_metrics["ssim"],
            "epoch_minutes": epoch_time / 60.0,
        }
        history.append(epoch_log)
        print(json.dumps(epoch_log), flush=True)

        checkpoint = {
            "epoch": epoch,
            "args": vars(args),
            "generator": generator.state_dict(),
            "discriminator": discriminator.state_dict(),
            "opt_g": opt_g.state_dict(),
            "opt_d": opt_d.state_dict(),
            "history": history,
            "conditioning_strategy": {
                "generator_input": "(masked_image, mask)",
                "discriminator_conditioning": "(candidate_image, masked_image, mask)",
                "reason": "Passing both y and M lets the model separate visible context from the hole location,"
                " which is more stable than conditioning on y alone when mask size varies.",
            },
        }

        torch.save(checkpoint, output_dir / "last.pt")
        if epoch % args.save_every == 0:
            torch.save(checkpoint, output_dir / f"epoch_{epoch:03d}.pt")
        if val_metrics["mse"] < best_val:
            best_val = val_metrics["mse"]
            torch.save(checkpoint, output_dir / "best.pt")


if __name__ == "__main__":
    main()
