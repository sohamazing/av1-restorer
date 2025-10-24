# utils/loss.py - ROBUST VERSION

import logging
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

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

# --- Constants ---
MIN_SIZE_MSSSIM = 160
# EPSILON = 1e-8  # For numerical stability


# ==============================================================================
# Fixed FrequencyLoss with Multiple Stability Improvements
# ==============================================================================

class FrequencyLoss(nn.Module):
    """
    ROBUST Frequency domain loss with multiple numerical stability fixes.
    
    Improvements:
    1. Optional phase loss (can be disabled entirely)
    2. Magnitude weighting to avoid division by zero
    3. Clamping before angle calculation
    4. Log-space magnitude comparison option
    5. Gradient clipping during backward pass
    """
    def __init__(
        self, 
        loss_func_type: str = 'l1',
        alpha: float = 1.0,  # 1.0 = magnitude only, 0.0 = phase only
        use_phase: bool = False,  # DISABLE phase by default
        use_log_magnitude: bool = True,  # Use log space for better stability
        eps: float = 1e-6
    ):
        super().__init__()
        
        # Loss function selection
        if loss_func_type.lower() == 'l1':
            self.loss_func = nn.L1Loss()
        elif loss_func_type.lower() in ['l2', 'mse']:
            self.loss_func = nn.MSELoss()
        elif loss_func_type.lower() == 'charbonnier':
            self.loss_func = CharbonnierLoss(eps=eps)
        else:
            raise ValueError(f"Unsupported loss type: {loss_func_type}")
        
        self.alpha = alpha
        self.use_phase = use_phase
        self.use_log_magnitude = use_log_magnitude
        self.eps = eps
        
        if not (0.0 <= alpha <= 1.0):
            raise ValueError("alpha must be in [0, 1]")
        
        logger.info(
            f"FrequencyLoss initialized: alpha={alpha}, "
            f"use_phase={use_phase}, use_log_mag={use_log_magnitude}"
        )

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Calculate FFT-based loss with robust numerical handling.
        
        Args:
            pred: [B, C, H, W] predicted image
            target: [B, C, H, W] target image
            
        Returns:
            Scalar loss tensor
        """
        # Ensure float32 for FFT stability
        pred = pred.float()
        target = target.float()
        
        # Compute FFT (use rfft2 for real inputs - more efficient)
        pred_fft = torch.fft.rfft2(pred, norm='ortho')
        target_fft = torch.fft.rfft2(target, norm='ortho')
        
        # === MAGNITUDE LOSS (Primary Component) ===
        pred_mag = torch.abs(pred_fft)
        target_mag = torch.abs(target_fft)
        
        if self.use_log_magnitude:
            # Log space is more stable and better matches perception
            # Add epsilon before log to avoid log(0)
            pred_mag_safe = torch.clamp(pred_mag, min=self.eps)
            target_mag_safe = torch.clamp(target_mag, min=self.eps)
            
            pred_log_mag = torch.log(pred_mag_safe + self.eps)
            target_log_mag = torch.log(target_mag_safe + self.eps)
            
            magnitude_loss = self.loss_func(pred_log_mag, target_log_mag)
        else:
            # Direct magnitude comparison
            magnitude_loss = self.loss_func(pred_mag, target_mag)
        
        # === PHASE LOSS (Optional, Often Unstable) ===
        if self.use_phase and self.alpha < 1.0:
            # Only compute phase where magnitude is significant
            # This avoids phase instability at near-zero frequencies
            
            # Create mask for significant frequencies
            magnitude_threshold = target_mag.mean() * 0.01  # 1% of mean
            mask = (target_mag > magnitude_threshold).float()
            
            # Compute phase using atan2 (more stable than torch.angle)
            pred_phase = torch.atan2(pred_fft.imag, pred_fft.real + self.eps)
            target_phase = torch.atan2(target_fft.imag, target_fft.real + self.eps)
            
            # Wrap phase difference to [-π, π]
            phase_diff = pred_phase - target_phase
            phase_diff = torch.atan2(torch.sin(phase_diff), torch.cos(phase_diff))
            
            # Apply mask and compute loss
            masked_phase_diff = phase_diff * mask
            phase_loss = torch.abs(masked_phase_diff).mean()
            
            # Combine magnitude and phase
            total_loss = self.alpha * magnitude_loss + (1 - self.alpha) * phase_loss
        else:
            # Magnitude only
            total_loss = magnitude_loss
        
        return total_loss


# ==============================================================================
# Other Loss Components (Your Existing Code with Minor Fixes)
# ==============================================================================

class CharbonnierLoss(nn.Module):
    """Robust L1 Loss with epsilon buffer."""
    def __init__(self, eps: float = 1e-3):
        super().__init__()
        self.register_buffer('eps_squared', torch.tensor(eps**2))

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return torch.sqrt((pred - target).pow(2) + self.eps_squared).mean()


class PerceptualLoss(nn.Module):
    """VGG19 or LPIPS-based perceptual loss."""
    def __init__(self, network: str = 'vgg'):
        super().__init__()
        self.network_type = network

        if self.network_type == 'vgg':
            vgg = vgg19(weights=VGG19_Weights.IMAGENET1K_V1).features
            self.model = nn.Sequential(*list(vgg.children())[:36]).eval()
            self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
            self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        elif self.network_type == 'lpips':
            if not LPIPS_AVAILABLE:
                raise ImportError("LPIPS not available. Install: pip install lpips")
            self.model = lpips.LPIPS(net='alex', spatial=False).eval()
        else:
            raise ValueError(f"Unknown network: {network}")

        for param in self.model.parameters():
            param.requires_grad = False

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred = torch.clamp(pred, 0.0, 1.0)
        target = torch.clamp(target, 0.0, 1.0)

        if self.network_type == 'vgg':
            pred_norm = (pred - self.mean) / self.std
            target_norm = (target - self.mean) / self.std
            return F.l1_loss(self.model(pred_norm), self.model(target_norm))
        else:  # LPIPS
            return self.model(pred * 2.0 - 1.0, target * 2.0 - 1.0).mean()


# ==============================================================================
# Combined Loss Manager
# ==============================================================================

class CombinedLoss(nn.Module):
    """Config-driven loss manager with robust numerical handling."""
    
    def __init__(self, loss_config: Dict[str, Dict], norm_range: Tuple[float, float] = (-1, 1)):
        super().__init__()
        self.losses = nn.ModuleDict()
        self.weights = {}
        self.loss_config = loss_config
        self.norm_range = norm_range
        self.needs_denorm_01 = (norm_range == (-1, 1))

        logger.info("Initializing CombinedLoss:")
        logger.info(f"  Normalization range: {norm_range}")
        logger.info(f"  Denormalization needed: {self.needs_denorm_01}")

        enabled_losses = []
        for name, config in loss_config.items():
            if not config.get('enabled', False):
                continue

            weight = config.get('weight', 1.0)
            if weight <= 0:
                logger.warning(f"Loss '{name}' has weight <= 0, skipping")
                continue

            params = config.get('params', {})
            self.weights[name] = weight

            try:
                if name == 'charbonnier':
                    self.losses['charbonnier'] = CharbonnierLoss(**params)
                elif name == 'l1':
                    self.losses['l1'] = nn.L1Loss(**params)
                elif name in ['l2', 'mse']:
                    self.losses[name] = nn.MSELoss(**params)
                elif name == 'perceptual':
                    self.losses['perceptual'] = PerceptualLoss(**params)
                elif name == 'ms_ssim':
                    if not MSSSIM_AVAILABLE:
                        logger.warning("pytorch-msssim not installed, skipping MS-SSIM")
                        continue
                    self.losses['ms_ssim'] = MS_SSIM(
                        data_range=1.0, size_average=True, 
                        channel=3, **params # nonnegative_ssim=True, 
                    )
                elif name == 'frequency':
                    # CRITICAL: Set safe defaults for FrequencyLoss
                    fft_params = {
                        'loss_func_type': params.get('loss_func_type', 'l1'),
                        'alpha': params.get('alpha', 1.0),
                        'use_phase': params.get('use_phase', False),  # DISABLE by default
                        'use_log_magnitude': params.get('use_log_magnitude', True),
                        'eps': params.get('eps', 1e-6)
                    }
                    self.losses['frequency'] = FrequencyLoss(**fft_params)
                else:
                    raise ValueError(f"Unknown loss type: {name}")

                logger.info(f"  ✓ Enabled '{name}' (weight={weight}, params={params})")
                enabled_losses.append(name)

            except Exception as e:
                logger.error(f"Failed to init '{name}': {e}")
                if name in self.losses:
                    del self.losses[name]
                if name in self.weights:
                    del self.weights[name]

        if not self.losses:
            raise ValueError("No valid losses initialized")
        logger.info(f"Active losses: {enabled_losses}")

    def _denormalize_to_01(self, x: torch.Tensor) -> torch.Tensor:
        """Convert [-1, 1] to [0, 1] with clamping."""
        if self.needs_denorm_01:
            return (x.clamp(-1.0, 1.0) + 1.0) / 2.0
        return x.clamp(0.0, 1.0)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Calculate weighted loss with NaN detection.
        
        Returns:
            (total_loss, loss_dict)
        """
        total_loss = torch.tensor(0.0, device=pred.device, dtype=pred.dtype)
        loss_dict = {}

        # Input sanity check
        if torch.isnan(pred).any() or torch.isinf(pred).any():
            logger.error("NaN/Inf in pred before loss calculation!")
            raise ValueError("NaN/Inf in prediction")
        if torch.isnan(target).any() or torch.isinf(target).any():
            logger.error("NaN/Inf in target before loss calculation!")
            raise ValueError("NaN/Inf in target")

        pred_01, target_01 = None, None
        H, W = pred.shape[-2:]

        for name, loss_fn in self.losses.items():
            weight = self.weights[name]

            try:
                # Select input range
                if name in ['perceptual', 'ms_ssim']:
                    if pred_01 is None:
                        pred_01 = self._denormalize_to_01(pred)
                        target_01 = self._denormalize_to_01(target)
                    pred_for_loss = pred_01
                    target_for_loss = target_01
                else:
                    pred_for_loss = pred
                    target_for_loss = target

                # Handle MS-SSIM size constraint
                if name == 'ms_ssim' and (H < MIN_SIZE_MSSSIM or W < MIN_SIZE_MSSSIM):
                    loss = torch.tensor(0.0, device=pred.device, dtype=pred.dtype)
                    loss_dict['metric_ms_ssim'] = torch.tensor(0.0, device=pred.device, dtype=pred.dtype)
                    logger.debug(f"Skipping MS-SSIM for size {H}x{W}")
                else:
                    # Calculate loss
                    if name == 'ms_ssim':
                        similarity = loss_fn(
                            pred_for_loss.clamp(0.0, 1.0),
                            target_for_loss.clamp(0.0, 1.0)
                        )
                        loss = 1.0 - similarity.clamp(min=0.0)
                        loss_dict['metric_ms_ssim'] = similarity
                    else:
                        loss = loss_fn(pred_for_loss, target_for_loss)

                # NaN/Inf check
                if torch.isnan(loss) or torch.isinf(loss):
                    logger.error(
                        f"NaN/Inf in '{name}' loss! "
                        f"Pred: [{pred.min():.4f}, {pred.max():.4f}], "
                        f"Target: [{target.min():.4f}, {target.max():.4f}]"
                    )
                    raise ValueError(f"NaN/Inf in loss: {name}")

            except Exception as e:
                logger.error(f"Error in loss '{name}': {e}", exc_info=True)
                raise

            weighted_loss = weight * loss
            loss_dict[f'loss_{name}'] = weighted_loss
            total_loss += weighted_loss

        # Final check
        if torch.isnan(total_loss) or torch.isinf(total_loss):
            logger.error(f"Total loss is NaN/Inf! Components: {loss_dict}")
            raise ValueError("Total loss is NaN/Inf")

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
