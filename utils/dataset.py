import os
from typing import cast
import random
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from utils.mask_generator import sample_rect_mask

class InpaintingTrainDataset(Dataset):
    def __init__(
        self,
        image_paths,
        image_size=256,
        regime="small",
        mixed_probs=(1/3, 1/3, 1/3),
    ):
        if len(mixed_probs) != 3:
            raise ValueError("mixed_probs must have length 3 for small/medium/large.")
        if sum(mixed_probs) <= 0:
            raise ValueError("mixed_probs must have positive total weight.")
        self.image_paths = self._expand_image_paths(image_paths)
        self.regime = regime
        self.mixed_probs = mixed_probs
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ])

    @staticmethod
    def _expand_image_paths(image_paths):
        if isinstance(image_paths, (str, os.PathLike)):
            image_paths = [image_paths]

        valid_ext = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        expanded = []

        for p in image_paths:
            p = os.fspath(p)
            if os.path.isdir(p):
                for name in sorted(os.listdir(p), key=lambda x: (len(x), x)):
                    full = os.path.join(p, name)
                    if os.path.isfile(full) and os.path.splitext(name.lower())[1] in valid_ext:
                        expanded.append(full)
            else:
                expanded.append(p)

        if len(expanded) == 0:
            raise ValueError("No images found. Please provide valid image file paths or folders.")

        return expanded

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        if self.regime == "mixed":
            regime = random.choices(
                ["small", "medium", "large"],
                weights=self.mixed_probs,
                k=1
            )[0]
        else:
            regime = self.regime

        path = self.image_paths[idx]
        img = Image.open(path).convert("RGB")
        x = cast(torch.Tensor, self.transform(img))

        _, H, W = x.shape
        M = sample_rect_mask(H, W, regime)
        y = x * (1.0 - M)

        return {
            "gt": x,
            "mask": M,
            "masked": y,
            "path": path,
        }

class FixedMaskInpaintingDataset(Dataset):
    def __init__(self, image_paths, fixed_masks_path, image_size=256):
        self.image_paths = InpaintingTrainDataset._expand_image_paths(image_paths)
        self.fixed_masks = torch.load(fixed_masks_path)   # [N,1,H,W]
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ])

        if len(self.image_paths) != self.fixed_masks.shape[0]:
            raise ValueError(
                f"Number of images ({len(self.image_paths)}) does not match "
                f"number of fixed masks ({self.fixed_masks.shape[0]})."
            )

        if self.fixed_masks.ndim != 4 or self.fixed_masks.shape[1] != 1:
            raise ValueError(
                f"Expected fixed masks of shape [N,1,H,W], got {self.fixed_masks.shape}"
            )

        mask_h, mask_w = self.fixed_masks.shape[-2:]
        if mask_h != image_size or mask_w != image_size:
            raise ValueError(
                f"Mask size ({mask_h}, {mask_w}) does not match image_size ({image_size}, {image_size})."
            )

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        img = Image.open(path).convert("RGB")
        x = cast(torch.Tensor, self.transform(img))   # [3,H,W]

        M = self.fixed_masks[idx]   # [1,H,W]
        y = x * (1.0 - M)

        return {
            "gt": x,
            "mask": M,
            "masked": y,
            "path": path,
        }