# data/

This folder stores the **data split files** and **fixed mask files** used in the image inpainting project.

All team members should use the same files in this folder to ensure fair comparison across methods.

---

## Files

### 1. `data_split.py`
This script splits the dataset into:

- `train_paths.txt`
- `val_paths.txt`
- `test_paths.txt`

Default split ratio:

- train = 80%
- val = 10%
- test = 10%

These `.txt` files store image paths.

---

### 2. `make_fix_data.py`
This script generates **fixed masks** for validation and test.

Recommended usage:

- validation:
  - `fixed_masks_val_medium.pt`
- test:
  - `fixed_masks_small.pt`
  - `fixed_masks_medium.pt`
  - `fixed_masks_large.pt`

The `.pt` files store pre-generated mask tensors and are used for reproducible evaluation.

---

## Existing files

### Path list files
- `train_paths.txt`
- `val_paths.txt`
- `test_paths.txt`

These files define which images belong to each split.

### Fixed mask files

#### Validation
- `fixed_masks_val_medium.pt`
- optionally:
  - `fixed_masks_val_small.pt`
  - `fixed_masks_val_large.pt`

#### Test
- `fixed_masks_small.pt`
- `fixed_masks_medium.pt`
- `fixed_masks_large.pt`

Each mask file is expected to have shape:

```python
[N, 1, H, W]
```

where:

- `N` = number of images in that split
- `H, W` = image size

---

## How to use

### 1. Split the dataset

```bash
python data/data_split.py
```

This generates:

- `data/train_paths.txt`
- `data/val_paths.txt`
- `data/test_paths.txt`

### 2. Generate fixed masks

```bash
python data/make_fix_data.py
```

This generates fixed mask files for validation and test.

Recommended setup:

- validation: medium only
- test: small / medium / large

### 3. Load path lists in code

```python
def load_paths(txt_path):
    with open(txt_path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]

train_paths = load_paths("./data/train_paths.txt")
val_paths = load_paths("./data/val_paths.txt")
test_paths = load_paths("./data/test_paths.txt")
```

### 4. Typical usage in training / evaluation

#### Training

Training uses:

- `train_paths.txt`

Training masks are generated on the fly, so fixed mask `.pt` files are not used for training.

#### Validation

Validation uses:

- `val_paths.txt`
- corresponding fixed mask file

Example:

```python
val_paths = load_paths("./data/val_paths.txt")
```

Paired with:

- `./data/fixed_masks_val_medium.pt`

#### Test

Test uses:

- `test_paths.txt`
- `fixed_masks_small.pt`
- `fixed_masks_medium.pt`
- `fixed_masks_large.pt`

This means final evaluation should be reported separately under:

- small masks
- medium masks
- large masks

---

## Why fixed masks are needed

The project compares multiple methods, such as:

- reconstruction-based
- GAN-based
- third generative method

If each method is tested with different random holes, the comparison will not be fair.

Therefore:

- training masks are random
- validation/test masks are fixed

---

## Important notes

### 1. Do not regenerate split files casually

Please do not rerun `data_split.py` and overwrite existing split files unless the team agrees to update the protocol.

### 2. Do not generate separate mask files for different methods

All methods should use the same fixed mask files.

### 3. Fixed masks are for validation/test only

Training should use random masks generated on the fly.

---

## Summary

- `data_split.py` creates train / val / test split.
- `make_fix_data.py` creates fixed validation/test masks.
- `*.txt` are path list files.
- `*.pt` are fixed mask files for evaluation.

This folder defines the shared data protocol for the whole team.