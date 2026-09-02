"""
Loss functions for LISN training.
Paper Section 3.1, Eq. 5:  L = ||I_SR - I_HR||_1 + α₁ * ||S(I_SR) - S(I_HR)||_1
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SobelEdgeLoss(nn.Module):
    """Sobel edge loss — encourages sharp edges in SR output.
    
    Computes Sobel gradients in x and y, combines into edge magnitude,
    then takes L1 distance between SR and HR edge maps.
    """
    def __init__(self):
        super().__init__()
        sobel_x = torch.tensor(
            [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32
        ).view(1, 1, 3, 3)
        sobel_y = torch.tensor(
            [[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32
        ).view(1, 1, 3, 3)
        self.register_buffer("sobel_x", sobel_x)
        self.register_buffer("sobel_y", sobel_y)

    def _edges(self, img):
        ex = F.conv2d(img, self.sobel_x, padding=1)
        ey = F.conv2d(img, self.sobel_y, padding=1)
        return torch.sqrt(ex ** 2 + ey ** 2 + 1e-6)

    def forward(self, sr, hr):
        return F.l1_loss(self._edges(sr), self._edges(hr))


class LISNLoss(nn.Module):
    """Combined loss: L1 + α * Sobel edge loss (Eq. 5).
    
    Default α=0.1 as stated in Section 3.1.
    """
    def __init__(self, edge_weight=0.1):
        super().__init__()
        self.l1 = nn.L1Loss()
        self.edge = SobelEdgeLoss()
        self.edge_weight = edge_weight

    def forward(self, sr, hr):
        return self.l1(sr, hr) + self.edge_weight * self.edge(sr, hr)
