# models/discriminator.py
"""
Robust PatchGAN discriminator for conditional GAN training.

Features
- Spectral Normalization on every Conv2d
- No BatchNorm (recommended with SN)
- Conditional concatenation (e.g., torch.cat([cond, img], dim=1))
- Supports 'hinge' and 'lsgan' adversarial losses
- Utility functions for R1 regularization and gradient penalty
"""

from typing import Optional
import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm


class PatchGANDiscriminator(nn.Module):
    """
    PatchGAN discriminator with Spectral Normalization.

    Args:
        in_channels: number of channels of the *target* image (e.g., 3 for RGB).
        cond_channels: number of channels of the conditioning input (e.g., LQ image). 
                       If conditional=False, set to 0.
        base_channels: number of channels in the first conv layer.
        n_layers: number of downsampling layers (3 -> typical 70x70 receptive patch at 256x256).
        conditional: whether to concat conditioning input along channel dim.
    """
    def __init__(
        self,
        in_channels: int = 3,
        cond_channels: int = 3,
        base_channels: int = 64,
        n_layers: int = 3,
        conditional: bool = True,
    ):
        super().__init__()
        self.conditional = conditional
        # input channels to the discriminator: target + optional cond
        in_ch = in_channels + (cond_channels if conditional else 0)

        layers = []
        # Layer 0: Input (no bias=False here, allowing the bias term to work with SN)
        layers += [
            spectral_norm(
                nn.Conv2d(in_ch, base_channels, kernel_size=4, stride=2, padding=1)
            ),
            nn.LeakyReLU(0.2, inplace=True),
        ]

        # Layers 1 to n_layers-1: Downsampling
        mult = 1
        for i in range(1, n_layers):
            mult_prev = mult
            mult = min(2**i, 8)
            layers += [
                # Removed BatchNorm2d and kept bias=False for memory/stability
                spectral_norm(
                    nn.Conv2d(base_channels * mult_prev, base_channels * mult,
                              kernel_size=4, stride=2, padding=1, bias=False)
                ),
                nn.LeakyReLU(0.2, inplace=True),
            ]

        # Intermediate conv (stride=1)
        mult_prev = mult
        mult = min(2**n_layers, 8)
        layers += [
            # Removed BatchNorm2d and kept bias=False
            spectral_norm(
                nn.Conv2d(base_channels * mult_prev, base_channels * mult,
                          kernel_size=4, stride=1, padding=1, bias=False)
            ),
            nn.LeakyReLU(0.2, inplace=True),
        ]

        # Final output conv -> single-channel score map (linear output, no activation)
        layers += [
            spectral_norm(
                nn.Conv2d(base_channels * mult, 1, kernel_size=4, stride=1, padding=1)
            )
        ]

        self.model = nn.Sequential(*layers)

    def forward(self, target: torch.Tensor, cond: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            target: (B, C, H, W) - real or generated image
            cond: (B, C_cond, H, W) - conditional input (e.g., low-quality image)
        Returns:
            (B, 1, H', W') patch realism scores (no activation)
        """
        if self.conditional:
            if cond is None:
                raise ValueError("Discriminator configured as conditional but `cond` is None.")
            x = torch.cat([cond, target], dim=1)
        else:
            x = target
        return self.model(x)


# ---------------------------
# Loss helpers (hinge, lsgan)
# ---------------------------

def d_loss_fn(d_real_scores: torch.Tensor, d_fake_scores: torch.Tensor, loss_type: str = "hinge"):
    """
    Discriminator loss.

    loss_type: 'hinge' (recommended with SN), or 'lsgan' (MSE)
    """
    if loss_type == "hinge":
        # hinge loss for D: max(0, 1 - D(x)) + max(0, 1 + D(G(z)))
        loss_real = torch.mean(torch.relu(1.0 - d_real_scores))
        loss_fake = torch.mean(torch.relu(1.0 + d_fake_scores))
        return 0.5 * (loss_real + loss_fake)
    elif loss_type == "lsgan":
        # MSE to target 1 for real and 0 for fake
        return 0.5 * (torch.mean((d_real_scores - 1.0) ** 2) + torch.mean((d_fake_scores - 0.0) ** 2))
    else:
        raise ValueError(f"Unsupported loss_type: {loss_type}")


def g_loss_fn(d_fake_scores: torch.Tensor, loss_type: str = "hinge"):
    """
    Generator adversarial loss.

    hinge: -D(G(x)) averaged (we want D(G) to be large positive)
    lsgan: MSE to 1.
    """
    if loss_type == "hinge":
        # Generator: maximize D(G) -> minimize -D(G)
        return -torch.mean(d_fake_scores)
    elif loss_type == "lsgan":
        return 0.5 * torch.mean((d_fake_scores - 1.0) ** 2)
    else:
        raise ValueError(f"Unsupported loss_type: {loss_type}")


# ---------------------------
# Regularizers / penalties
# ---------------------------

def r1_regularization(real_scores: torch.Tensor, real_images: torch.Tensor):
    """
    R1 gradient penalty: gradient of logits wrt real images.
    R1 = gamma/2 * ||grad(D(x))||^2
    Caller should multiply returned value by gamma/2 or whatever factor they use.
    This function returns ||grad||^2 (averaged over batch).
    """
    # real_scores: (B,1,H',W')  -> sum to scalar per batch element
    grad = torch.autograd.grad(
        outputs=real_scores.sum(), inputs=real_images, create_graph=True, retain_graph=True, only_inputs=True
    )[0]
    return torch.mean(grad.view(grad.size(0), -1).pow(2).sum(dim=1))


def gradient_penalty(discriminator: nn.Module, real: torch.Tensor, fake: torch.Tensor, cond: Optional[torch.Tensor] = None, eps: float = 1e-6):
    """
    WGAN-GP style gradient penalty between real and fake. Useful if you choose GP-based training.
    Note: this is not typically used with SN+hinge (R1 is better with SN).
    """
    alpha = torch.rand(real.size(0), 1, 1, 1, device=real.device, dtype=real.dtype)
    interpolates = (alpha * real + (1 - alpha) * fake).requires_grad_(True)
    if getattr(discriminator, "conditional", False):
        d_interpolates = discriminator(interpolates, cond)
    else:
        d_interpolates = discriminator(interpolates, None)
    grads = torch.autograd.grad(
        outputs=d_interpolates.sum(), inputs=interpolates, create_graph=True, retain_graph=True
    )[0]
    slopes = torch.sqrt((grads.view(grads.size(0), -1).pow(2).sum(1)) + eps)
    gp = ((slopes - 1.0) ** 2).mean()
    return gp


# ---------------------------
# Utility: init & optimizer helpers
# ---------------------------

def init_weights(module: nn.Module, init_type: str = "kaiming"):
    """
    He/Kaiming init for conv layers (fan_in), zeros for biases.
    Spectral norm wraps the conv: we still initialize the underlying conv weights.
    """
    for m in module.modules():
        if isinstance(m, nn.Conv2d):
            if init_type == "kaiming":
                nn.init.kaiming_normal_(m.weight, a=0.2, mode="fan_in", nonlinearity="leaky_relu")
            elif init_type == "xavier":
                nn.init.xavier_uniform_(m.weight)
            else:
                pass
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
