if __package__ is None or __package__ == "":
    import sys
    import os
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
from utils.dataset import InpaintingTrainDataset
import torch
from utils.mask_generator import sample_rect_mask

def make_fixed_masks(image_paths, image_size=256, regime="small", save_path="fixed_masks.pt"):
    image_list = InpaintingTrainDataset._expand_image_paths(image_paths)

    fixed_masks = []
    for _ in range(len(image_list)):
        mask = sample_rect_mask(image_size, image_size, regime)  # [1,H,W]
        fixed_masks.append(mask)

    fixed_masks = torch.stack(fixed_masks, dim=0)  # [N,1,H,W]
    torch.save(fixed_masks, save_path)
    print(f"Saved {len(image_list)} fixed masks to {save_path}")
    print(f"Mask tensor shape: {fixed_masks.shape}")
    print(f"Mean area ratio (first 5): {fixed_masks.mean(dim=(1,2,3))[:5]}")
def load_paths(txt_path):
    with open(txt_path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]
    
if __name__ == "__main__":
    image_paths = ["./CelebAMask-HQ/CelebA-HQ-img"]
if __name__ == "__main__":
    val_paths = load_paths("./data/val_paths.txt")
    test_paths = load_paths("./data/test_paths.txt")

    make_fixed_masks(val_paths, image_size=256, regime="small",  save_path="./data/fixed_masks_val_small.pt")
    make_fixed_masks(val_paths, image_size=256, regime="medium", save_path="./data/fixed_masks_val_medium.pt")
    make_fixed_masks(val_paths, image_size=256, regime="large",  save_path="./data/fixed_masks_val_large.pt")

    make_fixed_masks(test_paths, image_size=256, regime="small",  save_path="./data/fixed_masks_small.pt")
    make_fixed_masks(test_paths, image_size=256, regime="medium", save_path="./data/fixed_masks_medium.pt")
    make_fixed_masks(test_paths, image_size=256, regime="large",  save_path="./data/fixed_masks_large.pt")
