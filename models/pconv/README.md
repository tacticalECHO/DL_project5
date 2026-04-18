# models/

Reconstruction-based inpainting using Partial Convolutions (Liu et al.,
ECCV 2018).

## Files

- `partial_conv.py` - the PartialConv2d layer
- `model.py` - 7-level PConv-UNet
- `losses.py` - L1_hole + L1_valid + perceptual + style + TV
- `train.py` - training script
- `inpaint.py` - inference interface for evaluation

## Mask convention

Everything at the interface level uses the same convention as the rest of
the repo: `hole_mask` is 1 on the masked region and 0 on the visible
region, so `masked = gt * (1 - hole_mask)`.

The internal partial conv layer uses the opposite convention (1 = valid)
because that's how the paper writes the math. The flip happens inside
`model.py`, so callers don't need to care.

## How to run

All commands are meant to be run from the repo root. Before training, the
data preparation scripts under `data/` need to have been run so that the
path lists and the fixed validation masks exist.

Quick sanity check (50 images, 2 epochs, ~1 minute on a 5080):

```
python models/train.py --overfit_test
```

Full training:

```
python models/train.py --epochs 20
```

This produces:

- `./checkpoints/pconv/best_model.pt` and `latest_model.pt`
- `./logs/pconv/train_log.txt`
- `./samples/pconv/epoch_XXX.png`

## Inference

```python
from models.inpaint import load_model, inpaint

model = load_model('./checkpoints/pconv/best_model.pt', device='cuda')

# Single image
result = inpaint(model, masked_image, hole_mask)

# Or over a dataloader
for batch in test_loader:
    out = inpaint(model, batch['masked'].cuda(), batch['mask'].cuda())
```

The returned tensor is the composite: visible pixels from the input,
masked pixels from the model output.

## Model size and training cost

Around 25.8M parameters. With batch size 4 and bf16 mixed precision on
a 5080 Laptop (16 GB), one epoch over the 24k training images takes
roughly 10-15 minutes, so 20 epochs finishes overnight.
