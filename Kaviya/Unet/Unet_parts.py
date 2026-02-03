import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualDoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)

        self.shortcut = None
        if in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_ch)
            )

    def forward(self, x):
        identity = x

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        if self.shortcut is not None:
            identity = self.shortcut(identity)

        out = out + identity
        out = self.relu(out)
        return out


class AttentionGate(nn.Module):
    def __init__(self, x_ch, g_ch, inter_ch):
        super().__init__()
        self.Wx = nn.Sequential(
            nn.Conv2d(x_ch, inter_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(inter_ch)
        )
        self.Wg = nn.Sequential(
            nn.Conv2d(g_ch, inter_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(inter_ch)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(inter_ch, 1, kernel_size=1, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, g):
        if x.size(2) != g.size(2) or x.size(3) != g.size(3):
            g = F.interpolate(g, size=x.shape[2:], mode="bilinear", align_corners=False)

        x1 = self.Wx(x)
        g1 = self.Wg(g)
        a = self.relu(x1 + g1)
        alpha = self.psi(a)
        return x * alpha


class Down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.conv = ResidualDoubleConv(in_ch, out_ch)

    def forward(self, x):
        x = self.pool(x)
        x = self.conv(x)
        return x


class Up(nn.Module):
    def __init__(self, in_ch, out_ch, skip_ch):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, in_ch // 2, kernel_size=2, stride=2)

        inter_ch = max(out_ch // 2, 16)  # safe small intermediate
        self.att = AttentionGate(x_ch=skip_ch, g_ch=in_ch // 2, inter_ch=inter_ch)
        self.conv = ResidualDoubleConv(skip_ch + (in_ch // 2), out_ch)

    def forward(self, x_low, x_skip):
        x_up = self.up(x_low)
        diffY = x_skip.size(2) - x_up.size(2)
        diffX = x_skip.size(3) - x_up.size(3)
        if diffY != 0 or diffX != 0:
            x_up = F.pad(
                x_up,
                [diffX // 2, diffX - diffX // 2,
                 diffY // 2, diffY - diffY // 2]
            )

        x_skip_att = self.att(x_skip, x_up)

        x = torch.cat([x_skip_att, x_up], dim=1)
        x = self.conv(x)
        return x


class OutConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=1)

    def forward(self, x):
        return self.conv(x)
