import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from utils.dataset import FixedMaskInpaintingDataset
from cvae.model import CVAE

device = torch.device("cuda")

ckpt = torch.load("checkpoints/run_01/best.pt", map_location=device, weights_only=False)
model = CVAE(latent_dim=256).to(device)
model.load_state_dict(ckpt["model_state_dict"])
model.eval()

def load_paths(p):
    return [l.strip() for l in open(p, "r", encoding="utf-8") if l.strip()]

val_set = FixedMaskInpaintingDataset(
    image_paths=load_paths("./data/val_paths.txt"),
    fixed_masks_path="./data/fixed_masks_val_medium.pt",
    image_size=256,
)
sample = val_set[0]
masked = sample["masked"].unsqueeze(0).to(device)
mask = sample["mask"].unsqueeze(0).to(device)

with torch.no_grad():
    samples = model.sample(masked, mask, n_samples=5)  # [5, 1, 3, 256, 256]

print("Pairwise differences between 5 pluralistic samples:")
for i in range(5):
    for j in range(i+1, 5):
        diff = (samples[i] - samples[j]).abs().mean().item()
        print(f"  sample[{i}] vs sample[{j}]: {diff:.6f}")

with torch.no_grad():
    prior_input = torch.cat([masked, mask], dim=1)
    mu_p, logvar_p = model.prior(prior_input)
    sigma_p = (0.5 * logvar_p).exp()
    print(f"\nPrior sigma stats:")
    print(f"  mean = {sigma_p.mean().item():.4f}")
    print(f"  std  = {sigma_p.std().item():.4f}")
    print(f"  min  = {sigma_p.min().item():.4f}")
    print(f"  max  = {sigma_p.max().item():.4f}")