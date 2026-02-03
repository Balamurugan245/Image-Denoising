import torch
import torch.nn as nn
import torch.nn.functional as F

class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction=16, Conv2d=False, Bias=False):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        if(Conv2d==True):
              self.fc = nn.Sequential(
                   nn.Conv2d(in_channels, in_channels // reduction, 1, bias=Bias),
                   nn.ReLU(inplace=True),
                   nn.Conv2d(in_channels // reduction, in_channels, 1, bias=Bias)
                   )
        else:
              self.fc = nn.Sequential(
                   nn.linear(in_channels, in_channels // reduction, 1, bias=Bias),
                   nn.ReLU(inplace=True),
                   nn.linear(in_channels // reduction, in_channels, 1, bias=Bias)
                   )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out= self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out) * x

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out,_ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avg_out, max_out], dim=1)
        out = self.conv(out)
        return self.sigmoid(out) * x


class ResidualCBAMBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.ca = ChannelAttention(out_channels,True,True)
        self.sa = SpatialAttention()


    def forward(self, x):
        identity = x

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        out = self.ca(out)
        out = self.sa(out)

        out += identity
        return self.relu(out)


if __name__ == "__main__":
    x = torch.randn(1, 1, 32, 32)
    block = ResidualCBAMBlock(1, 64)
    y = block(x)
    print(x.shape)
    print(y.shape)
