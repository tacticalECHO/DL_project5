import matplotlib.pyplot as plt
from utils.dataset import InpaintingDataset
small_dataset = InpaintingDataset(image_paths=["./CelebAMask-HQ/CelebA-HQ-img"], image_size=128, regime="small")
medium_dataset = InpaintingDataset(image_paths=["./CelebAMask-HQ/CelebA-HQ-img"], image_size=128, regime="medium")
large_dataset = InpaintingDataset(image_paths=["./CelebAMask-HQ/CelebA-HQ-img"], image_size=128, regime="large")
small_sample = small_dataset[0]
medium_sample = medium_dataset[0]
large_sample = large_dataset[0]
test_num = 10
fig, axes = plt.subplots(test_num, 3, figsize=(9, 3 * test_num))
for i in range(test_num): 
    sample = large_dataset[i]
    mask = sample["mask"]
    ratio = mask.mean().item()
    print(ratio)
    axes[i, 0].imshow(sample["gt"].permute(1, 2, 0))
    axes[i, 0].set_title("GT")
    axes[i, 0].axis("off")

    axes[i, 1].imshow(sample["mask"].squeeze(0), cmap="gray")
    axes[i, 1].set_title("Mask")
    axes[i, 1].axis("off")

    axes[i, 2].imshow(sample["masked"].permute(1, 2, 0))
    axes[i, 2].set_title("Masked")
    axes[i, 2].axis("off")
plt.tight_layout()
plt.show()
