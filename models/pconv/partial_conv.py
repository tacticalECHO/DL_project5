"""
Partial Convolution layer from Liu et al. ECCV 2018.

Note on mask convention: this file uses valid_mask internally, where 1 means
a pixel is valid and 0 means it's a hole. The shared repo uses the opposite
(1 = masked). The flip happens at the model entry in model.py, so callers
outside don't need to worry about this.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class PartialConv2d(nn.Conv2d):
    """
    Mask-aware 2D conv. Subclasses nn.Conv2d so we get all the usual init
    and param handling for free.

    forward(x, mask) returns (output, updated_mask). The mask shrinks as we
    go deeper since any valid pixel in the window makes the output valid.
    """

    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1, groups=1, bias=True):
        super().__init__(in_channels, out_channels, kernel_size,
                         stride=stride, padding=padding,
                         dilation=dilation, groups=groups, bias=bias)

        # All-ones kernel for doing the mask convolution
        self.register_buffer(
            'mask_kernel',
            torch.ones(self.out_channels, self.in_channels // self.groups,
                       self.kernel_size[0], self.kernel_size[1])
        )
        self.window_size = self.in_channels * self.kernel_size[0] * self.kernel_size[1]

    def forward(self, x, mask):
        # Broadcast mask to match input channels
        if mask.shape[1] != x.shape[1]:
            mask = mask.expand(-1, x.shape[1], -1, -1)

        with torch.no_grad():
            # Convolve mask with all-ones kernel -> count of valid pixels per window
            mask_out = F.conv2d(
                mask, self.mask_kernel,
                bias=None, stride=self.stride,
                padding=self.padding, dilation=self.dilation,
                groups=1
            )
            # Rescale factor compensates for windows that have fewer valid pixels
            mask_ratio = self.window_size / (mask_out + 1e-8)
            mask_out = torch.clamp(mask_out, 0, 1)
            mask_ratio = mask_ratio * mask_out

        # Standard conv on masked input - the zeros won't pollute the result
        output = super().forward(x * mask)

        # Apply rescale
        if self.bias is not None:
            bias_view = self.bias.view(1, self.out_channels, 1, 1)
            output = (output - bias_view) * mask_ratio + bias_view
            output = output * mask_out[:, :1] if mask_out.shape[1] > 1 else output * mask_out
        else:
            output = output * mask_ratio

        mask_out_single = mask_out[:, :1] if mask_out.shape[1] > 1 else mask_out
        return output, mask_out_single


if __name__ == "__main__":
    print("Testing PartialConv2d...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    x = torch.randn(2, 3, 256, 256, device=device)
    mask = torch.ones(2, 1, 256, 256, device=device)
    mask[:, :, 64:192, 64:192] = 0
    layer = PartialConv2d(3, 64, kernel_size=3, padding=1).to(device)
    out, new_mask = layer(x, mask)
    print(f"input x: {x.shape}, mask: {mask.shape}")
    print(f"output: {out.shape}, new_mask: {new_mask.shape}")
    print("passed")
