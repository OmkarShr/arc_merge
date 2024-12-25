# model.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    """(Conv2d -> ReLU) x 2"""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.conv(x)

class UNetSmall(nn.Module):
    """
    A smaller 2-level U-Net for binary segmentation (1 output channel).
    Encoder path: inc(3->64) -> down1(64->128) -> bottleneck(128->256)
    Decoder path: up1(256->128) -> cat skip(128) => 256 -> DoubleConv => 128 -> outc => 1
    """
    def __init__(self, in_channels=3, out_channels=1):
        super(UNetSmall, self).__init__()

        # Encoder
        self.inc = DoubleConv(in_channels, 64)      # -> 64
        self.down1 = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(64, 128)                    # -> 128
        )

        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(128, 256)                   # -> 256
        )

        # Decoder: We only have one up block
        self.up1 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv1 = DoubleConv(256, 128)  # cat(128 + 128) = 256 in channels

        # Output
        self.outc = nn.Conv2d(128, out_channels, kernel_size=1)

    def forward(self, x):
        # Encoder
        x1 = self.inc(x)      # (B,64,H,W)
        x2 = self.down1(x1)   # (B,128,H/2,W/2)

        # Bottleneck
        x3 = self.bottleneck(x2)  # (B,256,H/4,W/4)

        # Decoder
        x = self.up1(x3)          # (B,128,H/2,W/2)
        # cat with x2 skip
        # ensure x has the same H,W as x2 (check input size is multiple of 4)
        x = torch.cat([x, x2], dim=1)  # => (B,128+128=256,H/2,W/2)
        x = self.conv1(x)             # => (B,128,H/2,W/2)

        # Out
        x = self.outc(x)             # => (B,1,H/2,W/2)
        # Optionally upsample back to original H,W if needed
        return x
