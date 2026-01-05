# model.py
import torch
import torch.nn as nn

class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg = nn.AdaptiveAvgPool2d(1)
        self.max = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return self.sigmoid(self.fc(self.avg(x)) + self.fc(self.max(x))) * x


class SpatialAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, 7, padding=3, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg = torch.mean(x, dim=1, keepdim=True)
        max,_ = torch.max(x, dim=1, keepdim=True)
        return self.sigmoid(self.conv(torch.cat([avg, max], 1))) * x


class CBAM(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.ca = ChannelAttention(channels)
        self.sa = SpatialAttention()

    def forward(self, x):
        return self.sa(self.ca(x))


class DoubleConv(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, 3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            CBAM(out_c)
        )

    def forward(self, x):
        return self.net(x)


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
        self.up = nn.ConvTranspose2d(in_c, in_c//2, 2, stride=2)
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
