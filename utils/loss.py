# core/loss.py

"""
Modular Loss Functions for AV1 Image Restoration.

This module provides a config-driven CombinedLoss that can dynamically construct
a weighted combination of various loss functions suitable for image-to-image tasks.
"""
import logging
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# Optional dependency checks
try:
    from pytorch_msssim import MS_SSIM
    MSSSIM_AVAILABLE = True
except ImportError:
    MSSSIM_AVAILABLE = False

try:
    import lpips
    LPIPS_AVAILABLE = True
except ImportError:
    LPIPS_AVAILABLE = False

from torchvision.models import vgg19, VGG19_Weights

logger = logging.getLogger(__name__)


# ==============================================================================
# SECTION 1: Standard Image Restoration Losses (av1_restorer)
# ==============================================================================

# --- Individual Loss Components ---

class CharbonnierLoss(nn.Module):
    """Robust L1 Loss (Smooth L1) - less sensitive to outliers than MSE."""
    def __init__(self, eps: float = 1e-3):
        super().__init__()
        self.eps = eps

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Calculates the mean Charbonnier loss."""
        return torch.sqrt((pred - target) ** 2 + self.eps ** 2).mean()


class PerceptualLoss(nn.Module):
    """
    VGG19 or LPIPS-based perceptual loss with proper input normalization.
    """
    def __init__(self, network: str = 'vgg'):
        super().__init__()
        self.network_type = network
        
        if self.network_type == 'vgg':
            # vgg = torch.hub.load('pytorch/vision:v0.10.0', 'vgg19', pretrained=True).features
            vgg = vgg19(weights=VGG19_Weights.IMAGENET1K_V1).features
            self.model = nn.Sequential(vgg[:35]).eval()  # To relu4_4
            
            # VGG normalization (ImageNet stats)
            self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
            self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
            
        elif self.network_type == 'lpips':
            if not LPIPS_AVAILABLE:
                raise ImportError("LPIPS is not available. Install with 'pip install lpips'")
            self.model = lpips.LPIPS(net='alex', spatial=False)
        else:
            raise ValueError(f"Unknown perceptual network: {network}. Choose 'vgg' or 'lpips'.")

        for param in self.parameters():
            param.requires_grad = False

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: [B, 3, H, W] in [0, 1] range (already denormalized by CombinedLoss)
            target: [B, 3, H, W] in [0, 1] range
        """
        if self.network_type == 'vgg':
            # Apply ImageNet normalization
            pred_norm = (pred - self.mean) / self.std
            target_norm = (target - self.mean) / self.std
            
            pred_features = self.model(pred_norm)
            target_features = self.model(target_norm)
            return F.l1_loss(pred_features, target_features)
            
        else:  # LPIPS
            # LPIPS expects input in [-1, 1] range, so convert from [0, 1]
            pred_lpips = pred * 2.0 - 1.0
            target_lpips = target * 2.0 - 1.0
            return self.model(pred_lpips, target_lpips).mean()


class FrequencyLoss(nn.Module):
    """
    Frequency domain loss (FFT Loss). Penalizes differences in the frequency spectrum,
    which is excellent for preserving high-frequency details (textures, edges).
    """
    def __init__(self, loss_func: nn.Module = nn.L1Loss(), alpha: float = 1.0):
        super().__init__()
        self.loss_func = loss_func
        self.alpha = alpha  # Weight for the magnitude component vs. phase

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Calculates the FFT loss on both magnitude and phase."""
        pred_fft = torch.fft.rfft2(pred, norm='ortho')
        target_fft = torch.fft.rfft2(target, norm='ortho')
        
        magnitude_loss = self.loss_func(torch.abs(pred_fft), torch.abs(target_fft))
        phase_loss = self.loss_func(torch.angle(pred_fft), torch.angle(target_fft))
        
        return self.alpha * magnitude_loss + (1 - self.alpha) * phase_loss


# --- Main Combined Loss Manager ---
class CombinedLoss(nn.Module):
    """A modular loss manager driven by a configuration dict for standard restoration."""
    
    def __init__(self, loss_config: Dict[str, Dict], norm_range: Tuple[float, float] = (-1, 1)):
        """
        Args:
            loss_config: Loss configuration from YAML
            norm_range: Normalization range from dataset config (e.g., (-1, 1) or (0, 1))
        """
        super().__init__()
        self.losses = nn.ModuleDict()
        self.weights = {}
        self.loss_config = loss_config
        self.norm_range = norm_range  # Store normalization range
        
        # Determine if we need to denormalize for certain losses
        self.needs_denorm = (norm_range == (-1, 1))

        logger.info("Initializing CombinedLoss:")
        logger.info(f"  Normalization range: {norm_range}")
        logger.info(f"  Denormalization required: {self.needs_denorm}")
        
        for name, config in loss_config.items():
            if not config.get('enabled', False):
                continue

            weight = config.get('weight', 1.0)
            params = config.get('params', {})
            self.weights[name] = weight

            if name == 'charbonnier':
                self.losses['charbonnier'] = CharbonnierLoss(**params)
            elif name == 'l1': 
                self.losses['l1'] = nn.L1Loss(**params)
            elif name == 'l2' or name == 'mse':
                self.losses[name] = nn.MSELoss(**params)
            elif name == 'perceptual':
                self.losses['perceptual'] = PerceptualLoss(**params)
            elif name == 'ms_ssim':
                if not MSSSIM_AVAILABLE:
                    logger.warning("pytorch-msssim not installed. Skipping MS-SSIM loss.")
                    continue
                self.losses['ms_ssim'] = MS_SSIM(data_range=1.0, size_average=True, channel=3, **params)
            elif name == 'frequency':
                self.losses['frequency'] = FrequencyLoss(**params)
            else:
                raise ValueError(f"Unknown loss type: {name}")
            logger.info(f"  - Enabled '{name}' loss with weight {weight} and params {params}")
            
        if not self.losses:
            raise ValueError("No losses were enabled in the configuration.")

    def _denormalize(self, x: torch.Tensor) -> torch.Tensor:
        """Convert from [-1, 1] to [0, 1] if needed."""
        if self.needs_denorm:
            return (x + 1.0) / 2.0
        return x

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Calculates the total weighted loss with proper normalization handling.
        
        Args:
            pred: Predicted image in range specified by norm_range
            target: Target image in range specified by norm_range
        
        Returns:
            (total_loss, loss_dict)
        """
        total_loss = torch.tensor(0.0, device=pred.device)
        loss_dict = {}

        # # ======================= DEBUGGING START =======================
        # # 1. Check if the inputs to the loss function are already corrupt.
        # if torch.isnan(pred).any() or torch.isinf(pred).any():
        #     print("!!! DEBUG: `pred` tensor is NaN or Inf BEFORE loss calculation!")
        # if torch.isnan(target).any() or torch.isinf(target).any():
        #     print("!!! DEBUG: `target` tensor is NaN or Inf BEFORE loss calculation!")
        # # ======================== DEBUGGING END ========================
        
        # Denormalize for losses that require [0, 1] range
        pred_01 = None
        target_01 = None

        for name, loss_fn in self.losses.items():
            # Determine if this loss needs [0, 1] range
            needs_01_range = name in ['perceptual', 'ms_ssim']
            
            if needs_01_range:
                # Lazy denormalization (only when needed)
                if pred_01 is None:
                    pred_01 = self._denormalize(pred)
                    target_01 = self._denormalize(target)
                pred_for_loss = pred_01
                target_for_loss = target_01
            else:
                # Use original range
                pred_for_loss = pred
                target_for_loss = target
            
            # Compute loss
            if name == 'ms_ssim':
                # MS-SSIM returns similarity (higher is better), so we use 1 - SSIM as loss
                val = loss_fn(
                    torch.clamp(pred_for_loss, 0, 1), 
                    torch.clamp(target_for_loss, 0, 1)
                )
                loss = 1.0 - val
                loss_dict['metric_ms_ssim'] = val  # Log the actual SSIM value
            else:
                loss = loss_fn(pred_for_loss, target_for_loss)

            # # ======================= DEBUGGING START =======================
            # # 2. Print each individual loss component's value.
            # # The one that prints 'nan' first is the source of the problem.
            # print(f"--- DEBUG: Loss component '{name}' = {loss.item()}")
            # if torch.isnan(loss):
            #     print(f"!!! DEBUG: '{name}' loss is NaN. Stopping execution to inspect.")
            #     # This will crash the program, allowing you to see the last valid prints.
            #     # You can also use a breakpoint here if you are using a debugger.
            #     import sys; sys.exit(1)
            # # ======================== DEBUGGING END ========================

            loss_dict[f'loss_{name}'] = loss
            total_loss += self.weights[name] * loss
            
        loss_dict['total_loss'] = total_loss
        return total_loss, loss_dict


# ==============================================================================
# SECTION 2: Dual Learning Losses (Reconstruction + Parameter Prediction) (av1_prior_embedder)
# ==============================================================================

class DualLearning(nn.Module):
    """Abstract base class for dual learning losses to reduce code duplication."""
    def __init__(self, crf_min: float, crf_max: float, use_perceptual: bool, perceptual_weight: float):
        super().__init__()
        self.crf_min = crf_min
        self.crf_max = crf_max
        self.crf_range = crf_max - crf_min
        self.use_perceptual = use_perceptual
        self.perceptual_weight = perceptual_weight
        self.perceptual_loss = None

        if self.use_perceptual:
            if not LPIPS_AVAILABLE:
                logger.warning("LPIPS requested but not installed. Disabling perceptual loss.")
                self.use_perceptual = False
            else:
                # Lazy initialization is not needed if we instantiate here.
                # Device will be handled by the main training script's `.to(device)` call.
                self.perceptual_loss = lpips.LPIPS(net='alex').eval()
                for param in self.perceptual_loss.parameters():
                    param.requires_grad = False

    def _normalize_crf(self, crf: torch.Tensor) -> torch.Tensor:
        return (crf - self.crf_min) / self.crf_range

    def _denormalize_crf(self, crf_norm: torch.Tensor) -> torch.Tensor:
        return crf_norm * self.crf_range + self.crf_min

    def _charbonnier_loss(self, pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-3) -> torch.Tensor:
        return torch.sqrt((pred - target) ** 2 + eps ** 2).mean()

    def _get_reconstruction_loss(self, recon_img: torch.Tensor, target_img: torch.Tensor, eps: float) -> Tuple[torch.Tensor, torch.Tensor]:
        charbonnier = self._charbonnier_loss(recon_img, target_img, eps)
        perceptual = torch.tensor(0.0, device=recon_img.device)
        if self.use_perceptual and self.perceptual_loss is not None:
            perceptual = self.perceptual_loss(recon_img, target_img).mean()
        
        total_recon = charbonnier + self.perceptual_weight * perceptual
        return total_recon, charbonnier, perceptual

    def forward(self, *args, **kwargs) -> Dict[str, torch.Tensor]:
        raise NotImplementedError


class DualLearningLoss(DualLearning):
    """Combines a reconstruction loss and a CRF prediction loss."""
    def __init__(self, recon_weight: float = 1.0, crf_weight: float = 0.1, **kwargs):
        super().__init__(**kwargs)
        self.recon_weight = recon_weight
        self.crf_weight = crf_weight
        logger.info(f"DualLearningLoss: λ_rec={recon_weight}, λ_crf={crf_weight}")

    def forward(self, recon_img: torch.Tensor, target_img: torch.Tensor, pred_crf: torch.Tensor, target_crf: torch.Tensor, **kwargs) -> Dict[str, torch.Tensor]:
        # 1. Reconstruction Loss
        total_recon_loss, charbonnier, perceptual = self._get_reconstruction_loss(recon_img, target_img, eps=1e-3)
        
        # 2. CRF Prediction Loss
        pred_crf_norm = self._normalize_crf(pred_crf.squeeze())
        target_crf_norm = self._normalize_crf(target_crf.squeeze())
        crf_loss = F.smooth_l1_loss(pred_crf_norm, target_crf_norm, beta=0.1)

        # 3. Total Weighted Loss
        total_loss = self.recon_weight * total_recon_loss + self.crf_weight * crf_loss

        return {
            'total_loss': total_loss,
            'loss_recon_total': total_recon_loss,
            'loss_charbonnier': charbonnier,
            'loss_perceptual': perceptual,
            'loss_crf': crf_loss,
        }


class AdaptiveDualLearningLoss(DualLearningLoss):
    """Adaptive version of DualLearningLoss using learnable uncertainty to balance tasks."""
    def __init__(self, initial_log_var_rec: float = 0.0, initial_log_var_crf: float = -2.0, **kwargs):
        super().__init__(**kwargs)
        self.log_var_rec = nn.Parameter(torch.tensor(initial_log_var_rec))
        self.log_var_crf = nn.Parameter(torch.tensor(initial_log_var_crf))
        logger.info("Using AdaptiveDualLearningLoss with learnable task weighting.")

    def forward(self, **kwargs) -> Dict[str, torch.Tensor]:
        base_losses = super().forward(**kwargs)
        
        rec_loss = base_losses['loss_recon_total']
        crf_loss = base_losses['loss_crf']
        
        precision_rec = torch.exp(-self.log_var_rec)
        precision_crf = torch.exp(-self.log_var_crf)
        
        weighted_rec = precision_rec * rec_loss + self.log_var_rec
        weighted_crf = precision_crf * crf_loss + self.log_var_crf
        
        base_losses['total_loss'] = weighted_rec + weighted_crf
        base_losses['weight_recon'] = precision_rec
        base_losses['weight_crf'] = precision_crf
        return base_losses


class BaselineRelativeLoss(DualLearning):
    """Dual learning where the model is only rewarded for improving upon a simple baseline."""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.register_buffer('crf_mean', torch.tensor(43.0)) # Initialize with a reasonable guess
        logger.info("Using BaselineRelativeLoss.")

    def forward(self, recon_img: torch.Tensor, target_img: torch.Tensor, pred_crf: torch.Tensor, target_crf: torch.Tensor, lq_img: torch.Tensor, **kwargs) -> Dict[str, torch.Tensor]:
        # 1. Reconstruction: Must beat the "do nothing" baseline (i.e., the LQ input)
        model_recon_loss, _, _ = self._get_reconstruction_loss(recon_img, target_img, eps=1e-3)
        with torch.no_grad():
            baseline_recon_loss, _, _ = self._get_reconstruction_loss(lq_img, target_img, eps=1e-3)
        
        recon_improvement = baseline_recon_loss - model_recon_loss
        relative_recon_loss = -torch.tanh(recon_improvement * 5) # Smooth reward for improvement
        
        # 2. CRF Prediction: Must beat the "always guess the mean" baseline
        if self.training: # Update running mean of CRF
            self.crf_mean = 0.99 * self.crf_mean + 0.01 * target_crf.mean()

        pred_crf_norm = self._normalize_crf(pred_crf.squeeze())
        target_crf_norm = self._normalize_crf(target_crf.squeeze())
        baseline_crf_norm = self._normalize_crf(self.crf_mean.expand_as(target_crf))

        model_crf_loss = F.smooth_l1_loss(pred_crf_norm, target_crf_norm, beta=0.1)
        with torch.no_grad():
            baseline_crf_loss = F.smooth_l1_loss(baseline_crf_norm, target_crf_norm, beta=0.1)

        crf_improvement = baseline_crf_loss - model_crf_loss
        relative_crf_loss = -torch.tanh(crf_improvement * 10)

        total_loss = relative_recon_loss + 0.1 * relative_crf_loss

        return {
            'total_loss': total_loss,
            'loss_recon_relative': relative_recon_loss,
            'loss_crf_relative': relative_crf_loss,
            'improvement_recon': recon_improvement,
            'improvement_crf': crf_improvement,
        }

# test CombinedLoss
if __name__ == "__main__":
    import torch
    
    # Test configuration
    loss_config = {
        'charbonnier': {'enabled': True, 'weight': 1.0, 'params': {'eps': 0.001}},
        'perceptual': {'enabled': True, 'weight': 0.2, 'params': {'network': 'vgg'}},
        'ms_ssim': {'enabled': True, 'weight': 0.15, 'params': {}},
    }
    
    # Test with [-1, 1] range
    print("Testing with norm_range=(-1, 1):")
    loss_fn = CombinedLoss(loss_config, norm_range=(-1, 1))
    
    i_size = 256

    pred = torch.randn(2, 3, i_size, i_size) * 0.5  # Simulate [-1, 1] range
    target = torch.randn(2, 3, i_size, i_size) * 0.5
    
    loss, loss_dict = loss_fn(pred, target)
    print(f"  Total loss: {loss.item():.6f}")
    for k, v in loss_dict.items():
        print(f"    {k}: {v.item():.6f}")
    
    # Test with [0, 1] range
    print("\nTesting with norm_range=(0, 1):")
    loss_fn = CombinedLoss(loss_config, norm_range=(0, 1))
    
    pred = torch.rand(2, 3, i_size, i_size)  # [0, 1] range
    target = torch.rand(2, 3, i_size, i_size)
    
    loss, loss_dict = loss_fn(pred, target)
    print(f"  Total loss: {loss.item():.6f}")
    for k, v in loss_dict.items():
        print(f"    {k}: {v.item():.6f}")
    
    print("\n✅ All tests passed!")
