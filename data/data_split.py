if __package__ is None or __package__ == "":
    import sys
    import os
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
import random
from utils.dataset import InpaintingTrainDataset

def split_paths(image_roots, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, seed=42):
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-8

    all_paths = InpaintingTrainDataset._expand_image_paths(image_roots)
    rng = random.Random(seed)
    rng.shuffle(all_paths)

    n = len(all_paths)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    n_test = n - n_train - n_val

    train_paths = all_paths[:n_train]
    val_paths = all_paths[n_train:n_train + n_val]
    test_paths = all_paths[n_train + n_val:]

    print(f"Total: {n}")
    print(f"Train: {len(train_paths)}")
    print(f"Val:   {len(val_paths)}")
    print(f"Test:  {len(test_paths)}")

    return train_paths, val_paths, test_paths

def save_paths(paths, save_path):
    with open(save_path, "w", encoding="utf-8") as f:
        for p in paths:
            f.write(p + "\n")



if __name__ == "__main__":
    image_roots = ["./CelebAMask-HQ/CelebA-HQ-img"]
    train_paths, val_paths, test_paths = split_paths(image_roots, seed=42)
    save_paths(train_paths, "./data/train_paths.txt")
    save_paths(val_paths, "./data/val_paths.txt")
    save_paths(test_paths, "./data/test_paths.txt")