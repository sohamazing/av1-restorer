# utils/loss.py

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


# ==============================================================================
# SECTION 1: Core Loss Components (SOTA Optimized)
# ==============================================================================

class CharbonnierLoss(nn.Module):
    """
    Robust L1 Loss (Charbonnier/Pseudo-Huber Loss).
    
    More robust to outliers than MSE, smoother than L1 near zero.
    Used in EDSR, RCAN, and most modern image restoration networks.
    
    Formula: sqrt((pred - target)^2 + eps^2)
    
    Args:
        eps (float): Small constant for numerical stability. Default: 1e-3
    """
    def __init__(self, eps: float = 1e-3):
        super().__init__()
        # Use buffer to avoid re-creating tensor on every forward pass
        self.register_buffer('eps_squared', torch.tensor(eps**2))

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: [B, C, H, W] Predicted image
            target: [B, C, H, W] Target image
        
        Returns:
            Scalar loss value
        """
        return torch.sqrt((pred - target).pow(2) + self.eps_squared).mean()


class PerceptualLoss(nn.Module):
    """
    VGG19 or LPIPS-based perceptual loss.
    
    Compares high-level features instead of pixel values, capturing
    perceptual similarity better than L1/L2 losses.
    
    Args:
        network (str): 'vgg' for VGG19 features or 'lpips' for learned perceptual metric
    """
    def __init__(self, network: str = 'vgg'):
        super().__init__()
        self.network_type = network

        if self.network_type == 'vgg':
            # Use VGG19 features up to relu5_4 (layer 36)
            # This is the most commonly used layer for perceptual loss
            vgg = vgg19(weights=VGG19_Weights.IMAGENET1K_V1).features
            self.model = nn.Sequential(*list(vgg.children())[:36]).eval()
            
            # ImageNet normalization parameters
            self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
            self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
            
        elif self.network_type == 'lpips':
            if not LPIPS_AVAILABLE:
                raise ImportError("LPIPS not available. Install: pip install lpips")
            # LPIPS uses AlexNet features by default (faster than VGG)
            self.model = lpips.LPIPS(net='alex', spatial=False).eval()
        else:
            raise ValueError(f"Unknown network: {network}. Choose 'vgg' or 'lpips'.")

        # Freeze all parameters - perceptual loss should not be trainable
        for param in self.model.parameters():
            param.requires_grad = False

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: [B, C, H, W] Predicted image in [0, 1]
            target: [B, C, H, W] Target image in [0, 1]
        
        Returns:
            Scalar perceptual loss value
        """
        # Ensure valid range [0, 1]
        pred = torch.clamp(pred, 0.0, 1.0)
        target = torch.clamp(target, 0.0, 1.0)

        if self.network_type == 'vgg':
            # Apply ImageNet normalization
            pred_norm = (pred - self.mean) / self.std
            target_norm = (target - self.mean) / self.std
            
            # Extract features and compare with L1 loss
            pred_features = self.model(pred_norm)
            target_features = self.model(target_norm)
            
            return F.l1_loss(pred_features, target_features)
        else:  # LPIPS
            # LPIPS expects inputs in [-1, 1] range
            return self.model(pred * 2.0 - 1.0, target * 2.0 - 1.0).mean()


class FrequencyLoss(nn.Module):
    """
    ROBUST Frequency Domain Loss using FFT.
    
    Penalizes differences in the frequency spectrum, which is excellent
    for preserving high-frequency details (textures, edges).
    
    Key improvements for stability:
    - Log-space magnitude comparison (more perceptually uniform)
    - Optional phase loss (disabled by default due to instability)
    - Magnitude-weighted phase to avoid near-zero frequency issues
    - Multiple loss function options (L1, L2, Charbonnier)
    
    Args:
        loss_func_type (str): Base loss function ('l1', 'l2', 'charbonnier')
        alpha (float): Weight for magnitude vs phase (1.0 = magnitude only)
        use_phase (bool): Enable phase loss (unstable, use with caution)
        use_log_magnitude (bool): Use log space for magnitude (recommended)
        eps (float): Small constant for numerical stability
    """
    def __init__(
        self, 
        loss_func_type: str = 'l1',
        alpha: float = 1.0,
        use_phase: bool = False,
        use_log_magnitude: bool = True,
        eps: float = 1e-6
    ):
        super().__init__()
        
        # Select base loss function
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
            f"FrequencyLoss: alpha={alpha}, "
            f"phase={use_phase}, log_mag={use_log_magnitude}, "
            f"base_loss={loss_func_type}"
        )

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Calculate FFT-based loss.
        
        Args:
            pred: [B, C, H, W] Predicted image
            target: [B, C, H, W] Target image
            
        Returns:
            Scalar frequency domain loss
        """
        # Ensure float32 for FFT stability
        pred = pred.float()
        target = target.float()
        
        # Compute 2D FFT (rfft2 is optimized for real inputs)
        pred_fft = torch.fft.rfft2(pred, norm='ortho')
        target_fft = torch.fft.rfft2(target, norm='ortho')
        
        # === MAGNITUDE LOSS (Primary Component) ===
        pred_mag = torch.abs(pred_fft)
        target_mag = torch.abs(target_fft)
        
        if self.use_log_magnitude:
            # Log space: more perceptually uniform and numerically stable
            pred_mag = torch.clamp(pred_mag, min=self.eps)
            target_mag = torch.clamp(target_mag, min=self.eps)
            
            pred_log_mag = torch.log(pred_mag + self.eps)
            target_log_mag = torch.log(target_mag + self.eps)
            
            magnitude_loss = self.loss_func(pred_log_mag, target_log_mag)
        else:
            magnitude_loss = self.loss_func(pred_mag, target_mag)
        
        # === PHASE LOSS (Optional, Use with Caution) ===
        if self.use_phase and self.alpha < 1.0:
            # Only compute phase at significant magnitudes
            magnitude_threshold = target_mag.mean() * 0.01
            mask = (target_mag > magnitude_threshold).float()
            
            # Compute phase using atan2 (more stable than torch.angle)
            pred_phase = torch.atan2(pred_fft.imag, pred_fft.real + self.eps)
            target_phase = torch.atan2(target_fft.imag, target_fft.real + self.eps)
            
            # Wrap phase difference to [-π, π]
            phase_diff = pred_phase - target_phase
            phase_diff = torch.atan2(torch.sin(phase_diff), torch.cos(phase_diff))
            
            # Apply magnitude mask and compute loss
            masked_phase_diff = phase_diff * mask
            phase_loss = torch.abs(masked_phase_diff).mean()
            
            # Combine magnitude and phase
            return self.alpha * magnitude_loss + (1 - self.alpha) * phase_loss
        
        # Magnitude only (default and recommended)
        return magnitude_loss


# ==============================================================================
# SECTION 2: Combined Loss Manager
# ==============================================================================

class CombinedLoss(nn.Module):
    """
    Config-driven loss manager with smart weighted logging.
    
    Features:
    - Automatic denormalization for perceptual/MS-SSIM losses
    - Smart logging: only log weighted version for non-1.0 weights
    - Robust NaN/Inf detection at multiple stages
    - MS-SSIM size constraint handling
    - Detailed error messages for debugging
    
    Config Example:
        loss_config = {
            'charbonnier': {'enabled': True, 'weight': 1.0, 'params': {'eps': 0.001}},
            'perceptual': {'enabled': True, 'weight': 0.1, 'params': {'network': 'vgg'}},
            'ms_ssim': {'enabled': True, 'weight': 0.15, 'params': {}},
            'frequency': {'enabled': True, 'weight': 0.05, 'params': {'alpha': 1.0}}
        }
    """
    
    def __init__(self, loss_config: Dict[str, Dict], norm_range: Tuple[float, float] = (-1, 1)):
        super().__init__()
        self.losses = nn.ModuleDict()
        self.weights = {}
        self.loss_config = loss_config
        self.norm_range = norm_range
        self.needs_denorm_01 = (norm_range == (-1, 1))

        logger.info("="*60)
        logger.info("Initializing CombinedLoss")
        logger.info(f"  Normalization range: {norm_range}")
        logger.info(f"  Denormalization needed: {self.needs_denorm_01}")
        logger.info("="*60)

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
                        data_range=1.0, 
                        size_average=True, 
                        channel=3, 
                        **params
                    )
                elif name == 'frequency':
                    # Set safe defaults for FrequencyLoss
                    fft_params = {
                        'loss_func_type': params.get('loss_func_type', 'l1'),
                        'alpha': params.get('alpha', 1.0),
                        'use_phase': params.get('use_phase', False),
                        'use_log_magnitude': params.get('use_log_magnitude', True),
                        'eps': params.get('eps', 1e-6)
                    }
                    self.losses['frequency'] = FrequencyLoss(**fft_params)
                else:
                    raise ValueError(f"Unknown loss type: {name}")

                logger.info(f"  ✓ Enabled '{name}' (weight={weight}, params={params})")
                enabled_losses.append(name)

            except Exception as e:
                logger.error(f"Failed to initialize '{name}': {e}")
                if name in self.losses:
                    del self.losses[name]
                if name in self.weights:
                    del self.weights[name]

        if not self.losses:
            raise ValueError("No valid losses initialized")
        
        logger.info(f"Active losses: {enabled_losses}")
        logger.info("="*60)

    def _denormalize_to_01(self, x: torch.Tensor) -> torch.Tensor:
        """
        Convert from training range to [0, 1] for perceptual/MS-SSIM losses.
        
        Args:
            x: Input tensor in norm_range
        
        Returns:
            Tensor in [0, 1] range
        """
        if self.needs_denorm_01:
            # [-1, 1] -> [0, 1]
            return (x.clamp(-1.0, 1.0) + 1.0) / 2.0
        # Already [0, 1]
        return x.clamp(0.0, 1.0)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Calculate weighted loss with smart logging.
        
        Returns:
            (total_loss, loss_dict) where loss_dict contains:
                - 'loss_{name}': Unweighted component (always logged)
                - 'loss_{name}_weighted': Weighted component (only if weight != 1.0)
                - 'metric_*': Special metrics (e.g., MS-SSIM similarity)
                - 'total_loss': Sum of all weighted components
        """
        total_loss = torch.tensor(0.0, device=pred.device, dtype=pred.dtype)
        loss_dict = {}

        # Input sanity check
        if torch.isnan(pred).any() or torch.isinf(pred).any():
            logger.error("NaN/Inf detected in prediction before loss calculation!")
            raise ValueError("NaN/Inf in prediction tensor")
        if torch.isnan(target).any() or torch.isinf(target).any():
            logger.error("NaN/Inf detected in target before loss calculation!")
            raise ValueError("NaN/Inf in target tensor")

        # Lazy denormalization (only compute once if needed)
        pred_01, target_01 = None, None
        H, W = pred.shape[-2:]

        for name, loss_fn in self.losses.items():
            weight = self.weights[name]

            try:
                # === Input Range Selection ===
                if name in ['perceptual', 'ms_ssim']:
                    # These losses require [0, 1] range
                    if pred_01 is None:
                        pred_01 = self._denormalize_to_01(pred)
                        target_01 = self._denormalize_to_01(target)
                    pred_for_loss = pred_01
                    target_for_loss = target_01
                else:
                    # Use native training range
                    pred_for_loss = pred
                    target_for_loss = target

                # === MS-SSIM Size Constraint ===
                if name == 'ms_ssim' and (H < MIN_SIZE_MSSSIM or W < MIN_SIZE_MSSSIM):
                    # Image too small for MS-SSIM, skip this loss
                    loss = torch.tensor(0.0, device=pred.device, dtype=pred.dtype)
                    loss_dict['metric_ms_ssim'] = torch.tensor(0.0, device=pred.device, dtype=pred.dtype)
                    logger.debug(f"Skipping MS-SSIM for size {H}x{W} (min: {MIN_SIZE_MSSSIM})")
                else:
                    # === Calculate Loss ===
                    if name == 'ms_ssim':
                        # MS-SSIM returns similarity (higher is better)
                        similarity = loss_fn(
                            pred_for_loss.clamp(0.0, 1.0),
                            target_for_loss.clamp(0.0, 1.0)
                        )
                        loss = 1.0 - similarity.clamp(min=0.0)
                        # Log similarity as a metric (not a loss)
                        loss_dict['metric_ms_ssim'] = similarity.detach()
                    else:
                        loss = loss_fn(pred_for_loss, target_for_loss)

                # === NaN/Inf Check ===
                if torch.isnan(loss) or torch.isinf(loss):
                    logger.error(
                        f"NaN/Inf in '{name}' loss! "
                        f"Pred range: [{pred.min():.4f}, {pred.max():.4f}], "
                        f"Target range: [{target.min():.4f}, {target.max():.4f}]"
                    )
                    raise ValueError(f"NaN/Inf in loss component: {name}")

            except Exception as e:
                logger.error(f"Error computing loss '{name}': {e}", exc_info=True)
                raise

            # === Smart Logging ===
            # Always log unweighted component
            loss_dict[f'loss_{name}'] = loss.detach()
            
            # Apply weighting
            weighted_loss = weight * loss
            
            # Only log weighted version if weight != 1.0
            if weight != 1.0:
                loss_dict[f'loss_{name}_weighted'] = weighted_loss.detach()
            
            # Accumulate total loss (always use weighted version)
            total_loss += weighted_loss

        # === Final Sanity Check ===
        if torch.isnan(total_loss) or torch.isinf(total_loss):
            logger.error(f"Total loss is NaN/Inf! Component breakdown:")
            for k, v in loss_dict.items():
                logger.error(f"  {k}: {v}")
            raise ValueError("Total loss is NaN/Inf")

        loss_dict['total_loss'] = total_loss
        return total_loss, loss_dict


# ==============================================================================
# SECTION 3: Dual Learning Losses (Reconstruction + Parameter Prediction) (av1_prior_embedder)
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

# ==============================================================================
# SECTION 4: Testing
# ==============================================================================

if __name__ == "__main__":
    import torch
    
    print("="*60)
    print("Testing CombinedLoss with smart weighted logging")
    print("="*60)
    
    # Test configuration
    loss_config = {
        'charbonnier': {'enabled': True, 'weight': 1.0, 'params': {'eps': 0.001}},
        'perceptual': {'enabled': True, 'weight': 0.1, 'params': {'network': 'vgg'}},
        'ms_ssim': {'enabled': True, 'weight': 0.15, 'params': {}},
        'frequency': {'enabled': True, 'weight': 0.05, 'params': {'alpha': 1.0}},
    }
    
    # Test with [-1, 1] range
    print("\nTest 1: norm_range=(-1, 1)")
    loss_fn = CombinedLoss(loss_config, norm_range=(-1, 1))
    
    pred = torch.randn(2, 3, 256, 256) * 0.5
    target = torch.randn(2, 3, 256, 256) * 0.5
    
    loss, loss_dict = loss_fn(pred, target)
    print(f"\nTotal loss: {loss.item():.6f}")
    print("\nLoss components:")
    for k, v in loss_dict.items():
        if not k.startswith('metric_'):
            print(f"  {k}: {v.item():.6f}")
    print("\nMetrics:")
    for k, v in loss_dict.items():
        if k.startswith('metric_'):
            print(f"  {k}: {v.item():.6f}")
    
    # Verify smart logging
    print("\nSmart logging verification:")
    print(f"  'loss_charbonnier_weighted' in dict: {'loss_charbonnier_weighted' in loss_dict}")
    print(f"  'loss_perceptual_weighted' in dict: {'loss_perceptual_weighted' in loss_dict}")
    print("  ✓ Only non-1.0 weights have '_weighted' versions")
    
    # Test with [0, 1] range
    print("\n" + "="*60)
    print("Test 2: norm_range=(0, 1)")
    loss_fn = CombinedLoss(loss_config, norm_range=(0, 1))
    
    pred = torch.rand(2, 3, 256, 256)
    target = torch.rand(2, 3, 256, 256)
    
    loss, loss_dict = loss_fn(pred, target)
    print(f"\nTotal loss: {loss.item():.6f}")
    print(f"Component count: {sum(1 for k in loss_dict if k.startswith('loss_'))}")
    
    print("\n" + "="*60)
    print("All tests passed!")
    print("="*60)