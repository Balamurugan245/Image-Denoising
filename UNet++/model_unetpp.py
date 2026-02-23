import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, padding=1),
            nn.InstanceNorm2d(out_c, affine=True),  
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, 3, padding=1),
            nn.InstanceNorm2d(out_c, affine=True),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class UNetPP(nn.Module):
    def __init__(self, in_ch=1, out_ch=1):
        super().__init__()

        f = [32, 64, 128, 256]  

        self.pool = nn.MaxPool2d(2)

        # Encoder
        self.x0_0 = ConvBlock(in_ch, f[0])
        self.x1_0 = ConvBlock(f[0], f[1])
        self.x2_0 = ConvBlock(f[1], f[2])
        self.x3_0 = ConvBlock(f[2], f[3])

        # Nested Decoder
        self.x0_1 = ConvBlock(f[0] + f[1], f[0])
        self.x1_1 = ConvBlock(f[1] + f[2], f[1])
        self.x2_1 = ConvBlock(f[2] + f[3], f[2])

        self.x0_2 = ConvBlock(f[0]*2 + f[1], f[0])
        self.x1_2 = ConvBlock(f[1]*2 + f[2], f[1])

        self.x0_3 = ConvBlock(f[0]*3 + f[1], f[0])

        self.final = nn.Conv2d(f[0], out_ch, 1)

    def forward(self, x):
        x0_0 = self.x0_0(x)
        x1_0 = self.x1_0(self.pool(x0_0))
        x2_0 = self.x2_0(self.pool(x1_0))
        x3_0 = self.x3_0(self.pool(x2_0))

        x0_1 = self.x0_1(torch.cat([x0_0, F.interpolate(x1_0, scale_factor=2, mode="bilinear", align_corners=False)], 1))
        x1_1 = self.x1_1(torch.cat([x1_0, F.interpolate(x2_0, scale_factor=2, mode="bilinear", align_corners=False)], 1))
        x2_1 = self.x2_1(torch.cat([x2_0, F.interpolate(x3_0, scale_factor=2, mode="bilinear", align_corners=False)], 1))

        x0_2 = self.x0_2(torch.cat([x0_0, x0_1,
                                    F.interpolate(x1_1, scale_factor=2, mode="bilinear", align_corners=False)], 1))

        x1_2 = self.x1_2(torch.cat([x1_0, x1_1,
                                    F.interpolate(x2_1, scale_factor=2, mode="bilinear", align_corners=False)], 1))

        x0_3 = self.x0_3(torch.cat([x0_0, x0_1, x0_2,
                                    F.interpolate(x1_2, scale_factor=2, mode="bilinear", align_corners=False)], 1))

        return self.final(x0_3)