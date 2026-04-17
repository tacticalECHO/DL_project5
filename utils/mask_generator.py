import random
import torch

def sample_rect_mask(H, W, regime="small"):
    if regime == "small":
        h_min, h_max = int(0.1 * H), int(0.2 * H)
        w_min, w_max = int(0.1 * W), int(0.2 * W)
    elif regime == "medium":
        h_min, h_max = int(0.3 * H), int(0.4 * H)
        w_min, w_max = int(0.3 * W), int(0.4 * W)
    elif regime == "large":
        h_min, h_max = int(0.5 * H), int(0.6 * H)
        w_min, w_max = int(0.5 * W), int(0.6 * W)
    else:
        raise ValueError(f"Unknown regime: {regime}")

    h = random.randint(h_min, max(h_min, h_max))
    w = random.randint(w_min, max(w_min, w_max))

    cy = random.randint(h // 2, H - (h - h // 2))
    cx = random.randint(w // 2, W - (w - w // 2))

    y1 = cy - h // 2
    y2 = y1 + h
    x1 = cx - w // 2
    x2 = x1 + w

    mask = torch.zeros((1, H, W), dtype=torch.float32)
    mask[:, y1:y2, x1:x2] = 1.0
    return mask