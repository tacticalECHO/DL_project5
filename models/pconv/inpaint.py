"""
Inference interface for the trained partial-conv model.

Usage:
    from models.inpaint import load_model, inpaint

    model = load_model('./checkpoints/pconv/best_model.pt', device='cuda')
    result = inpaint(model, masked_image, hole_mask)

Both the inputs and outputs follow the repo's mask convention:
  hole_mask: 1 = masked, 0 = visible.
"""
import os
import sys
import torch

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

try:
    from models.pconv.model import PConvUNet
except ImportError:
    from .model import PConvUNet


def load_model(checkpoint_path, device='cuda'):
    """Load a checkpoint produced by train.py and put the model in eval mode."""
    model = PConvUNet().to(device)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model'])
    model.eval()
    return model


@torch.no_grad()
def inpaint(model, masked_image, hole_mask):
    """
    Run inpainting on a batch or a single image.

    masked_image: [3, H, W] or [B, 3, H, W], values in [0, 1]. Masked pixels
                  should be zero but we enforce it below just in case.
    hole_mask:    [1, H, W] or [B, 1, H, W], 1 = masked, 0 = visible.

    Returns the composite: visible pixels from the input, masked pixels from
    the model output. This is the standard reporting convention for inpainting
    and it helps the MSE/SSIM numbers look right.
    """
    device = next(model.parameters()).device

    # Handle single-image inputs by adding a batch dim, then remove it later
    single = (masked_image.dim() == 3)
    if single:
        masked_image = masked_image.unsqueeze(0)
        hole_mask = hole_mask.unsqueeze(0)

    masked_image = masked_image.to(device)
    hole_mask = hole_mask.to(device)

    # Zero out the masked region just to be safe
    masked_image = masked_image * (1 - hole_mask).expand_as(masked_image)

    output = model(masked_image, hole_mask)

    # Standard inpainting composite
    hole_3ch = hole_mask.expand_as(output)
    composite = (1 - hole_3ch) * masked_image + hole_3ch * output
    composite = composite.clamp(0, 1)

    if single:
        composite = composite.squeeze(0)

    return composite


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: python models/inpaint.py <checkpoint_path>")
        print("example: python models/inpaint.py ./checkpoints/pconv/best_model.pt")
        sys.exit(1)

    ckpt_path = sys.argv[1]
    print(f"loading {ckpt_path}")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = load_model(ckpt_path, device=device)
    print(f"loaded on {device}")

    x = torch.rand(1, 3, 256, 256, device=device)
    hole = torch.zeros(1, 1, 256, 256, device=device)
    hole[:, :, 80:176, 80:176] = 1
    masked = x * (1 - hole)

    result = inpaint(model, masked, hole)
    print(f"in: {masked.shape}, mask: {hole.shape} -> out: {result.shape}")
    print(f"output range: [{result.min().item():.3f}, {result.max().item():.3f}]")
    print("passed")
