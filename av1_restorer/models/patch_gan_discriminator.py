# models/discriminator.py

import torch
import torch.nn as nn
from typing import List

class PatchGANDiscriminator(nn.Module):
    """
    PatchGAN discriminator for adversarial training.

    This discriminator classifies whether 70x70 overlapping patches of an image
    are real or fake. This patch-level feedback encourages the generator to
    produce sharp, high-frequency details across the entire image.

    Args:
        in_channels (int): Number of input channels (e.g., 3 for RGB).
        base_channels (int): Number of channels in the first convolutional layer.
        n_layers (int): Number of downsampling layers in the network.
    """
    def __init__(self, in_channels: int = 3, base_channels: int = 64, n_layers: int = 3):
        super().__init__()
        
        layers: List[nn.Module] = [
            nn.Conv2d(in_channels, base_channels, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True)
        ]
        
        mult = 1
        for i in range(1, n_layers):
            mult_prev = mult
            mult = min(2**i, 8)
            layers += [
                nn.Conv2d(base_channels * mult_prev, base_channels * mult, kernel_size=4, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(base_channels * mult),
                nn.LeakyReLU(0.2, inplace=True)
            ]
            
        mult_prev = mult
        mult = min(2**n_layers, 8)
        layers += [
            nn.Conv2d(base_channels * mult_prev, base_channels * mult, kernel_size=4, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(base_channels * mult),
            nn.LeakyReLU(0.2, inplace=True)
        ]
        
        layers += [nn.Conv2d(base_channels * mult, 1, kernel_size=4, stride=1, padding=1)]
        
        self.model = nn.Sequential(*layers)

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        """
        Takes an image and returns a 2D patch of realism scores.
        """
        return self.model(img)
        