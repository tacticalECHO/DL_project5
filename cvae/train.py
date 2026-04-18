import argparse
import os
import random
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
from torchvision.utils import make_grid, save_image
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.dataset import InpaintingTrainDataset, FixedMaskInpaintingDataset
from cvae.model import CVAE
from cvae.loss import cvae_loss, beta_schedule


def load_paths(txt_path: str) -> list[str]:
    with open(txt_path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def set_seed(seed: int = 42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def horizontal_flip_batch(batch: dict) -> dict:
    if random.random() < 0.5:
        batch["gt"] = torch.flip(batch["gt"], dims=[-1])
        batch["masked"] = torch.flip(batch["masked"], dims=[-1])
        batch["mask"] = torch.flip(batch["mask"], dims=[-1])
    return batch


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    device: torch.device,
    epoch: int,
    beta: float,
    log_every: int = 100,
) -> dict:
    model.train()
    running = {"l1": 0.0, "kl": 0.0, "total": 0.0}
    n_batches = 0

    pbar = tqdm(loader, desc=f"[train e{epoch}]", leave=False,
                dynamic_ncols=True, mininterval=1.0)
    for step, batch in enumerate(pbar):
        batch = horizontal_flip_batch(batch)
        gt = batch["gt"].to(device, non_blocking=True)
        masked = batch["masked"].to(device, non_blocking=True)
        mask = batch["mask"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with autocast("cuda"):
            x_hat, (mu_q, logvar_q, mu_p, logvar_p) = model(gt, masked, mask)

        x_hat    = x_hat.float()
        mu_q     = mu_q.float()
        logvar_q = logvar_q.float()
        mu_p     = mu_p.float()
        logvar_p = logvar_p.float()

        total, log = cvae_loss(
            x_hat, gt,
            mu_q, logvar_q, mu_p, logvar_p,
            beta=beta,
        )

        scaler.scale(total).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        running["l1"] += log["l1"]
        running["kl"] += log["kl"]
        running["total"] += log["total"]
        n_batches += 1

        if (step + 1) % log_every == 0:
            pbar.set_postfix(l1=f"{log['l1']:.4f}", kl=f"{log['kl']:.1f}", beta=f"{beta:.2f}")

    return {k: v / max(n_batches, 1) for k, v in running.items()}


@torch.no_grad()
def validate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> dict:
    model.eval()
    total_l1 = 0.0
    n = 0
    for batch in loader:
        gt = batch["gt"].to(device, non_blocking=True)
        masked = batch["masked"].to(device, non_blocking=True)
        mask = batch["mask"].to(device, non_blocking=True)
        x_hat, _ = model(gt, masked, mask)
        total_l1 += (x_hat - gt).abs().mean().item()
        n += 1
    return {"val_l1": total_l1 / max(n, 1)}


@torch.no_grad()
def save_visualization(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    save_path: Path,
    n_samples: int = 4,
):
    model.eval()
    batch = next(iter(loader))
    gt = batch["gt"][:n_samples].to(device)
    masked = batch["masked"][:n_samples].to(device)
    mask = batch["mask"][:n_samples].to(device)
    x_hat, _ = model(gt, masked, mask)

    mask_vis = mask.repeat(1, 3, 1, 1)

    rows = []
    for i in range(n_samples):
        rows.extend([gt[i], masked[i], x_hat[i], mask_vis[i]])
    grid = make_grid(torch.stack(rows), nrow=4, padding=2, pad_value=1.0)

    save_path.parent.mkdir(parents=True, exist_ok=True)
    save_image(grid.clamp(0, 1), save_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name",     type=str, default="run_01")
    parser.add_argument("--epochs",       type=int, default=50)
    parser.add_argument("--batch-size",   type=int, default=12)
    parser.add_argument("--lr",           type=float, default=1e-4)
    parser.add_argument("--latent-dim",   type=int, default=256)
    parser.add_argument("--warmup-epochs", type=int, default=10)
    parser.add_argument("--subset",       type=int, default=-1,
                        help="Use only first N training images (-1 = all)")
    parser.add_argument("--num-workers",  type=int, default=4)
    parser.add_argument("--seed",         type=int, default=42)
    parser.add_argument("--log-every",    type=int, default=50)
    parser.add_argument("--max-beta",     type=float, default=1.0,
                        help="Max beta after warmup (v1.3 default 1.0)")
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Out dir
    ckpt_dir = Path("checkpoints") / args.run_name
    vis_dir = ckpt_dir / "vis"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    vis_dir.mkdir(parents=True, exist_ok=True)

    # Dataset
    train_paths = load_paths("./data/train_paths.txt")
    if args.subset > 0:
        train_paths = train_paths[:args.subset]
    print(f"Train set: {len(train_paths)} images (mixed regime)")

    train_set = InpaintingTrainDataset(
        image_paths=train_paths,
        image_size=256,
        regime="mixed",
    )
    val_set = FixedMaskInpaintingDataset(
        image_paths=load_paths("./data/val_paths.txt"),
        fixed_masks_path="./data/fixed_masks_val_medium.pt",
        image_size=256,
    )
    print(f"Val set: {len(val_set)} images (medium regime, fixed masks)")

    train_loader = DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_set, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
    )

    model = CVAE(latent_dim=args.latent_dim).to(device)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Model: CVAE latent_dim={args.latent_dim}, {n_params:.2f}M params")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999))
    scaler = GradScaler("cuda")

    # Training loop
    best_val_l1 = float("inf")
    log_file = ckpt_dir / "train.log"
    log_file.write_text(
        f"epoch,beta,train_l1,train_kl,train_total,val_l1,time_s\n"
    )

    for epoch in range(args.epochs):
        t0 = time.time()
        beta = beta_schedule(epoch, warmup_epochs=args.warmup_epochs, max_beta=args.max_beta)

        train_metrics = train_one_epoch(
            model, train_loader, optimizer, scaler, device,
            epoch=epoch, beta=beta,
            log_every=args.log_every,
        )
        val_metrics = validate(model, val_loader, device)
        dt = time.time() - t0

        print(
            f"Epoch {epoch:3d} | beta={beta:.2f} | "
            f"train l1={train_metrics['l1']:.4f} kl={train_metrics['kl']:7.2f} "
            f"total={train_metrics['total']:7.2f} | "
            f"val l1={val_metrics['val_l1']:.4f} | {dt:.1f}s"
        )

        with open(log_file, "a") as f:
            f.write(
                f"{epoch},{beta:.4f},{train_metrics['l1']:.6f},"
                f"{train_metrics['kl']:.4f},{train_metrics['total']:.4f},"
                f"{val_metrics['val_l1']:.6f},{dt:.2f}\n"
            )

        save_visualization(
            model, val_loader, device,
            save_path=vis_dir / f"epoch_{epoch:03d}.png",
        )

        ckpt = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "args": vars(args),
            "val_l1": val_metrics["val_l1"],
        }
        torch.save(ckpt, ckpt_dir / "last.pt")

        if val_metrics["val_l1"] < best_val_l1:
            best_val_l1 = val_metrics["val_l1"]
            torch.save(ckpt, ckpt_dir / "best.pt")
            print(f"  → new best val_l1={best_val_l1:.4f}, saved best.pt")

    print(f"\nTraining done. Best val_l1: {best_val_l1:.4f}")

if __name__ == "__main__":
    main()