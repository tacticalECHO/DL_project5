"""
PConv-UNet: 7-layer encoder-decoder with partial convolutions, following
Liu et al. 2018. Skip connections pass both features and masks between
matching encoder/decoder levels.

Mask convention at the interface follows the shared repo (1 = hole,
0 = visible). We flip to valid_mask inside forward() because that's what
the partial conv math is written for.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from models.pconv.partial_conv import PartialConv2d
except ImportError:
    from .partial_conv import PartialConv2d


class PCBActiv(nn.Module):
    """Partial conv + BN + activation. Just for readability."""
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=1,
                 bn=True, activ='relu'):
        super().__init__()
        self.conv = PartialConv2d(in_ch, out_ch, kernel_size,
                                  stride=stride, padding=padding, bias=(not bn))
        self.bn = nn.BatchNorm2d(out_ch) if bn else None
        if activ == 'relu':
            self.activ = nn.ReLU(inplace=True)
        elif activ == 'leaky':
            self.activ = nn.LeakyReLU(0.2, inplace=True)
        else:
            self.activ = None

    def forward(self, x, mask):
        x, mask = self.conv(x, mask)
        if self.bn is not None:
            x = self.bn(x)
        if self.activ is not None:
            x = self.activ(x)
        return x, mask


class PConvUNet(nn.Module):
    def __init__(self):
        super().__init__()

        # Encoder: 256 -> 128 -> 64 -> 32 -> 16 -> 8 -> 4 -> 2
        # First layer has no BN per the original paper
        self.enc_1 = PCBActiv(3,   64,  kernel_size=7, stride=2, padding=3, bn=False)
        self.enc_2 = PCBActiv(64,  128, kernel_size=5, stride=2, padding=2)
        self.enc_3 = PCBActiv(128, 256, kernel_size=5, stride=2, padding=2)
        self.enc_4 = PCBActiv(256, 512, kernel_size=3, stride=2, padding=1)
        self.enc_5 = PCBActiv(512, 512, kernel_size=3, stride=2, padding=1)
        self.enc_6 = PCBActiv(512, 512, kernel_size=3, stride=2, padding=1)
        self.enc_7 = PCBActiv(512, 512, kernel_size=3, stride=2, padding=1)

        # Decoder: upsample, concat with skip, partial conv
        # Channels = (upsampled + skip) -> out
        self.dec_7 = PCBActiv(512 + 512, 512, kernel_size=3, padding=1, activ='leaky')
        self.dec_6 = PCBActiv(512 + 512, 512, kernel_size=3, padding=1, activ='leaky')
        self.dec_5 = PCBActiv(512 + 512, 512, kernel_size=3, padding=1, activ='leaky')
        self.dec_4 = PCBActiv(512 + 256, 256, kernel_size=3, padding=1, activ='leaky')
        self.dec_3 = PCBActiv(256 + 128, 128, kernel_size=3, padding=1, activ='leaky')
        self.dec_2 = PCBActiv(128 + 64,   64, kernel_size=3, padding=1, activ='leaky')
        # Last layer maps back to 3 channels, no BN / activation
        self.dec_1 = PCBActiv(64 + 3,     3,  kernel_size=3, padding=1, bn=False, activ=None)

    def forward(self, image, hole_mask):
        # Flip from repo convention (1=hole) to internal (1=valid)
        valid_mask = 1.0 - hole_mask

        # Encoder path. Keep every intermediate for the skip connections.
        e0_img, e0_mask = image, valid_mask
        e1_img, e1_mask = self.enc_1(image, valid_mask)
        e2_img, e2_mask = self.enc_2(e1_img, e1_mask)
        e3_img, e3_mask = self.enc_3(e2_img, e2_mask)
        e4_img, e4_mask = self.enc_4(e3_img, e3_mask)
        e5_img, e5_mask = self.enc_5(e4_img, e4_mask)
        e6_img, e6_mask = self.enc_6(e5_img, e5_mask)
        e7_img, e7_mask = self.enc_7(e6_img, e6_mask)

        # Decoder path with skip connections
        x, m = self._up_concat_conv(e7_img, e7_mask, e6_img, e6_mask, self.dec_7)
        x, m = self._up_concat_conv(x, m, e5_img, e5_mask, self.dec_6)
        x, m = self._up_concat_conv(x, m, e4_img, e4_mask, self.dec_5)
        x, m = self._up_concat_conv(x, m, e3_img, e3_mask, self.dec_4)
        x, m = self._up_concat_conv(x, m, e2_img, e2_mask, self.dec_3)
        x, m = self._up_concat_conv(x, m, e1_img, e1_mask, self.dec_2)
        x, m = self._up_concat_conv(x, m, e0_img, e0_mask, self.dec_1)

        return x

    @staticmethod
    def _up_concat_conv(x, mask, skip_x, skip_mask, conv_layer):
        # Nearest-neighbor upsample keeps mask edges sharp
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        mask = F.interpolate(mask, scale_factor=2, mode='nearest')
        x = torch.cat([x, skip_x], dim=1)
        mask = torch.cat([mask, skip_mask], dim=1)
        # After concat take max across channels: valid if any channel is valid
        mask = mask.max(dim=1, keepdim=True)[0]
        x, mask = conv_layer(x, mask)
        return x, mask


if __name__ == "__main__":
    print("Testing PConvUNet...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = PConvUNet().to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"params: {n_params / 1e6:.2f} M")

    image = torch.randn(2, 3, 256, 256, device=device)
    hole_mask = torch.zeros(2, 1, 256, 256, device=device)
    hole_mask[:, :, 80:176, 80:176] = 1
    masked_image = image * (1 - hole_mask)

    with torch.no_grad():
        out = model(masked_image, hole_mask)
    print(f"in: {masked_image.shape}, mask: {hole_mask.shape} -> out: {out.shape}")
    print("passed")
