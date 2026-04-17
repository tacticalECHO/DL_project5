# utils/

This folder contains reusable utilities for the image inpainting project.

These files define:

- how masks are generated
- how datasets are loaded
- how train/test behavior differs

---

## Files

### 1. `mask_generator.py`
This file defines the rectangular mask generator.

Main function:

- `sample_rect_mask(H, W, regime)`

Supported regimes:

- `small`
- `medium`
- `large`

Mask convention:

- `1` = masked region
- `0` = visible region

Masked image is constructed as:

`y = x * (1 - M)`

where:

- `x` = ground-truth image
- `M` = binary mask
- `y` = masked input image

---

### 2. `dataset.py`
This file defines dataset classes used by the project.

Currently the folder contains two important dataset types:

- `InpaintingTrainDataset`
- `FixedMaskInpaintingDataset`

---

## Dataset classes

### `InpaintingTrainDataset`
Used for training.

Behavior:

- reads image
- resizes image
- generates mask on the fly
- returns:
  - `gt`
  - `mask`
  - `masked`
  - `path`

Recommended usage:

- `regime="mixed"`
- `mixed_probs=(1/3, 1/3, 1/3)`

This means small / medium / large masks are sampled with equal probability during training.

#### Example: import and use training dataset

```python
from utils.dataset import InpaintingTrainDataset

def load_paths(txt_path):
    with open(txt_path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]

train_paths = load_paths("./data/train_paths.txt")

train_dataset = InpaintingTrainDataset(
    train_paths,
    image_size=256,
    regime="mixed",
    mixed_probs=(1/3, 1/3, 1/3),
)
```

#### Example: inspect one sample

```python
sample = train_dataset[0]

print(sample["gt"].shape)      # [3, H, W]
print(sample["mask"].shape)    # [1, H, W]
print(sample["masked"].shape)  # [3, H, W]
print(sample["path"])
```

#### Example: use with DataLoader

```python
from torch.utils.data import DataLoader

train_loader = DataLoader(
  train_dataset,
  batch_size=16,
  shuffle=True,
  num_workers=4,
)
```

#### Typical model input

```python
import torch

for batch in train_loader:
  inp = torch.cat([batch["masked"], batch["mask"]], dim=1)  # [B, 4, H, W]
  gt = batch["gt"]                                            # [B, 3, H, W]
  break
```

---

### `FixedMaskInpaintingDataset`
Used for validation and test.

Behavior:

- reads image
- loads pre-generated fixed mask from `.pt`
- applies the corresponding mask by index
- returns:
  - `gt`
  - `mask`
  - `masked`
  - `path`

#### Example: validation dataset

```python
from utils.dataset import FixedMaskInpaintingDataset

def load_paths(txt_path):
  with open(txt_path, "r", encoding="utf-8") as f:
    return [line.strip() for line in f if line.strip()]

val_paths = load_paths("./data/val_paths.txt")

val_dataset = FixedMaskInpaintingDataset(
  val_paths,
  fixed_masks_path="./data/fixed_masks_val_medium.pt",
  image_size=256,
)
```

#### Example: test datasets

```python
from utils.dataset import FixedMaskInpaintingDataset

def load_paths(txt_path):
  with open(txt_path, "r", encoding="utf-8") as f:
    return [line.strip() for line in f if line.strip()]

test_paths = load_paths("./data/test_paths.txt")

test_small_dataset = FixedMaskInpaintingDataset(
  test_paths,
  fixed_masks_path="./data/fixed_masks_small.pt",
  image_size=256,
)

test_medium_dataset = FixedMaskInpaintingDataset(
  test_paths,
  fixed_masks_path="./data/fixed_masks_medium.pt",
  image_size=256,
)

test_large_dataset = FixedMaskInpaintingDataset(
  test_paths,
  fixed_masks_path="./data/fixed_masks_large.pt",
  image_size=256,
)
```

---

## Returned sample format

All dataset classes return a dictionary like:

```python
{
  "gt": x,
  "mask": M,
  "masked": y,
  "path": path,
}
```

where:

- `gt`: ground-truth image tensor, shape `[3, H, W]`
- `mask`: binary mask tensor, shape `[1, H, W]`
- `masked`: masked image tensor, shape `[3, H, W]`
- `path`: original image path

---

## Train vs Validation/Test

### Training

- mask is sampled randomly
- supports `mixed`
- used for learning

### Validation/Test

- mask is fixed
- loaded from saved `.pt` files
- used for fair and reproducible evaluation

---

## Important conventions

### Input convention

The model input is usually:

- masked image
- mask

Often concatenated as:

```python
inp = torch.cat([masked, mask], dim=1)
```

So the model input has 4 channels:

- 3 image channels
- 1 mask channel

### Do not change casually

Please do not change the following without team agreement:

- mask convention (`1 = masked`)
- image resize size
- dataset split
- fixed mask files
- mixed mask ratio

These directly affect the fairness of experimental comparison.

---

## Summary

- `mask_generator.py` defines how rectangular masks are sampled.
- `dataset.py` defines training and evaluation dataset behavior.
- `InpaintingTrainDataset` uses random mask generation for training.
- `FixedMaskInpaintingDataset` uses fixed mask loading for validation/test.