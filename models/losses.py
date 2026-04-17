"""
Loss terms used for training. Five components as in the original paper:
  - L1 on hole region (main supervision signal)
  - L1 on valid region (keep observed pixels unchanged)
  - Perceptual: L1 on VGG-16 features of output and composite
  - Style: L1 on Gram matrices of the same VGG features
  - Total variation on the composite, for smoother hole edges

Mask convention here follows the repo: hole_mask is 1 on the masked region
and 0 on the visible region.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class VGG16Features(nn.Module):
    """Pretrained VGG16, frozen. Returns features after relu1_2, relu2_2, relu3_3."""
    def __init__(self):
        super().__init__()
        vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1).features
        self.slice1 = nn.Sequential(*[vgg[i] for i in range(4)])     # relu1_2
        self.slice2 = nn.Sequential(*[vgg[i] for i in range(4, 9)])  # relu2_2
        self.slice3 = nn.Sequential(*[vgg[i] for i in range(9, 16)]) # relu3_3
        for p in self.parameters():
            p.requires_grad = False
        # ImageNet normalization
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std',  torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, x):
        x = (x - self.mean) / self.std
        h1 = self.slice1(x)
        h2 = self.slice2(h1)
        h3 = self.slice3(h2)
        return [h1, h2, h3]


def gram_matrix(x):
    b, c, h, w = x.shape
    feat = x.view(b, c, h * w)
    return torch.bmm(feat, feat.transpose(1, 2)) / (c * h * w)


class InpaintingLoss(nn.Module):
    """
    Weighted sum of all five loss terms. Default weights from the paper.
    Set use_style / use_tv to False if you just want the lightweight version.
    """
    def __init__(self, w_hole=6.0, w_valid=1.0, w_perceptual=0.05,
                 w_style=120.0, w_tv=0.1,
                 use_style=True, use_tv=True):
        super().__init__()
        self.w_hole = w_hole
        self.w_valid = w_valid
        self.w_perceptual = w_perceptual
        self.w_style = w_style
        self.w_tv = w_tv
        self.use_style = use_style
        self.use_tv = use_tv
        self.vgg = VGG16Features()
        self.l1 = nn.L1Loss()

    def forward(self, output, target, hole_mask):
        loss_dict = {}
        hole_3ch = hole_mask.expand_as(output)

        # Composite: use ground truth where visible, model output where masked.
        # Matches the reporting convention used at inference time.
        composite = (1 - hole_3ch) * target + hole_3ch * output

        # L1 on the two regions. Separating them lets us weight the hole
        # much more than the visible part.
        loss_valid = self.l1((1 - hole_3ch) * output, (1 - hole_3ch) * target)
        loss_hole = self.l1(hole_3ch * output, hole_3ch * target)

        loss_dict['valid'] = loss_valid.item()
        loss_dict['hole'] = loss_hole.item()
        total = self.w_valid * loss_valid + self.w_hole * loss_hole

        # Perceptual: L1 in VGG feature space. Compute for both the raw
        # output and the composite, since both versions matter.
        feat_out = self.vgg(output)
        feat_comp = self.vgg(composite)
        feat_target = self.vgg(target)
        loss_perc = 0
        for fo, fc, ft in zip(feat_out, feat_comp, feat_target):
            loss_perc = loss_perc + self.l1(fo, ft) + self.l1(fc, ft)
        loss_dict['perceptual'] = loss_perc.item()
        total = total + self.w_perceptual * loss_perc

        # Style: L1 on Gram matrices, captures texture statistics.
        if self.use_style:
            loss_style = 0
            for fo, fc, ft in zip(feat_out, feat_comp, feat_target):
                loss_style = loss_style + self.l1(gram_matrix(fo), gram_matrix(ft))
                loss_style = loss_style + self.l1(gram_matrix(fc), gram_matrix(ft))
            loss_dict['style'] = loss_style.item()
            total = total + self.w_style * loss_style

        # Total variation on the composite to reduce visible seams at hole edges.
        if self.use_tv:
            loss_tv = (
                torch.mean(torch.abs(composite[:, :, :, :-1] - composite[:, :, :, 1:])) +
                torch.mean(torch.abs(composite[:, :, :-1, :] - composite[:, :, 1:, :]))
            )
            loss_dict['tv'] = loss_tv.item()
            total = total + self.w_tv * loss_tv

        loss_dict['total'] = total.item()
        return total, loss_dict


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    loss_fn = InpaintingLoss().to(device)
    output = torch.rand(2, 3, 256, 256, device=device, requires_grad=True)
    target = torch.rand(2, 3, 256, 256, device=device)
    hole_mask = torch.zeros(2, 1, 256, 256, device=device)
    hole_mask[:, :, 80:176, 80:176] = 1
    total, losses = loss_fn(output, target, hole_mask)
    print(f"total: {total.item():.4f}")
    for k, v in losses.items():
        print(f"  {k}: {v:.4f}")
    total.backward()
    print("passed")
