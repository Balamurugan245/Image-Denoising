import torch
import torch.nn as nn
from Unet_parts import DoubleConv

class Down(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv = DoubleConv(in_c, out_c)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        c = self.conv(x)
        return c, self.pool(c)


class Up(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_c, in_c // 2, 2, stride=2)
        self.conv = DoubleConv(in_c, out_c)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        return self.conv(torch.cat([x2, x1], dim=1))


class UNetCBAM(nn.Module):
    def __init__(self):
        super().__init__()
        self.d1 = Down(3, 64)
        self.d2 = Down(64, 128)
        self.d3 = Down(128, 256)
        self.d4 = Down(256, 512)
        self.bottleneck = DoubleConv(512, 1024)
        self.u1 = Up(1024, 512)
        self.u2 = Up(512, 256)
        self.u3 = Up(256, 128)
        self.u4 = Up(128, 64)
        self.out = nn.Conv2d(64, 3, 1)

    def forward(self, x):
        c1, p1 = self.d1(x)
        c2, p2 = self.d2(p1)
        c3, p3 = self.d3(p2)
        c4, p4 = self.d4(p3)
        b = self.bottleneck(p4)
        x = self.u1(b, c4)
        x = self.u2(x, c3)
        x = self.u3(x, c2)
        x = self.u4(x, c1)
        return torch.sigmoid(self.out(x))
