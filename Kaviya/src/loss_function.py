import torch
import torch.nn as nn
import torch.nn.functional as F

class SSIMLoss(nn.Module):
    def __init__(self, window_size=11):
        super().__init__()
        self.window_size = window_size

    def forward(self, x, y):
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2

        mu_x = F.avg_pool2d(x, self.window_size, 1, self.window_size // 2)
        mu_y = F.avg_pool2d(y, self.window_size, 1, self.window_size // 2)

        sigma_x = F.avg_pool2d(x * x, self.window_size, 1, self.window_size // 2) - mu_x ** 2
        sigma_y = F.avg_pool2d(y * y, self.window_size, 1, self.window_size // 2) - mu_y ** 2
        sigma_xy = F.avg_pool2d(x * y, self.window_size, 1, self.window_size // 2) - mu_x * mu_y

        ssim = ((2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)) / \
               ((mu_x ** 2 + mu_y ** 2 + C1) * (sigma_x + sigma_y + C2))

        return 1 - ssim.mean()
