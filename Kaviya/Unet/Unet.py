import torch
import torch.nn as nn
from Unet_parts import ResidualDoubleConv, Down, Up, OutConv


class UNet_AttRes(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        super().__init__()
        self.inc   = ResidualDoubleConv(in_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 1024)

        self.up1 = Up(in_ch=1024, out_ch=512, skip_ch=512)
        self.up2 = Up(in_ch=512,  out_ch=256, skip_ch=256)
        self.up3 = Up(in_ch=256,  out_ch=128, skip_ch=128)
        self.up4 = Up(in_ch=128,  out_ch=64,  skip_ch=64)

        self.outc = OutConv(64, out_channels)

    def forward(self, x):
        x1 = self.inc(x)      # 64
        x2 = self.down1(x1)   # 128
        x3 = self.down2(x2)   # 256
        x4 = self.down3(x3)   # 512
        x5 = self.down4(x4)   # 1024 (bottleneck)

        x = self.up1(x5, x4)  # 512
        x = self.up2(x,  x3)  # 256
        x = self.up3(x,  x2)  # 128
        x = self.up4(x,  x1)  # 64

        return self.outc(x)
