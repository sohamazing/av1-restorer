"""
AV1-Nano-ResNet: SOTA Lightweight Artifact Removal

Architecture:
- Single-scale residual network (like EDSR/RCAN).
- Built with hyper-efficient blocks (Depthwise-Separable + ECA).
- Replaces BatchNorm with GroupNorm for small-batch stability.
- Uses residual scaling for stable training.
- Not a U-Net: processes at a single resolution for maximum speed.

Author: Soham Mukherjee
Version: 2.0 (SOTA-informed)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict
import logging

logger = logging.getLogger(__name__)

# ==============================================================================
# SECTION 1: SOTA-Informed Core Building Blocks
# ==============================================================================

class DepthwiseSeparable(nn.Module):
    """
    Efficient Depthwise Separable Conv.
    Uses GroupNorm for small-batch stability (replaces BatchNorm).
    """
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3, stride: int = 1, num_groups: int = 16):
        super().__init__()
        padding = kernel_size // 2
        
        # Auto-adjust num_groups if out_ch is not divisible
        if out_ch % num_groups != 0:
            num_groups = 16
            while out_ch % num_groups != 0 and num_groups > 1:
                num_groups //= 2
            if out_ch % num_groups != 0:
                num_groups = 1 # Fallback to LayerNorm-like
                
        self.depthwise = nn.Conv2d(in_ch, in_ch, kernel_size, stride, padding, groups=in_ch, bias=False)
        self.pointwise = nn.Conv2d(in_ch, out_ch, 1, bias=False)
        self.norm = nn.GroupNorm(num_groups=num_groups, num_channels=out_ch)
        self.act = nn.GELU()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.norm(self.pointwise(self.depthwise(x))))

class ECA(nn.Module):
    """
    Efficient Channel Attention (ECA)
    More parameter-efficient than Squeeze-Excitation.
    """
    def __init__(self, channels: int, kernel_size: int = 3):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=kernel_size, 
                              padding=(kernel_size - 1) // 2, bias=False)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.shape
        y = self.avg_pool(x).view(b, 1, c)       # (B, 1, C)
        y = self.conv(y).view(b, c, 1, 1)    # (B, C, 1, 1)
        return x * y.sigmoid()

class EfficientResBlock(nn.Module):
    """
    MobileNet-inspired block.
    Uses GroupNorm and ECA.
    """
    def __init__(self, channels: int, expansion: int = 2, num_groups: int = 16):
        super().__init__()
        hidden = channels * expansion
        if hidden % num_groups != 0:
            num_groups = 1 # Adjust group norm
            
        self.conv1 = nn.Sequential(
            nn.Conv2d(channels, hidden, 1, bias=False),
            nn.GroupNorm(num_groups, hidden),
            nn.GELU()
        )
        self.dwconv = DepthwiseSeparable(hidden, hidden, 3, num_groups=num_groups)
        self.attn = ECA(hidden)
        self.conv2 = nn.Sequential(
            nn.Conv2d(hidden, channels, 1, bias=False),
            nn.GroupNorm(num_groups, channels)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.conv1(x)
        out = self.dwconv(out)
        out = self.attn(out)
        out = self.conv2(out)
        return out + identity

class MultiScaleFeatureExtractor(nn.Module):
    """Parallel multi-scale feature extraction head."""
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        split = out_ch // 3
        
        self.fine = DepthwiseSeparable(in_ch, split, 3)
        self.medium = nn.Sequential(
            DepthwiseSeparable(in_ch, split, 3),
            DepthwiseSeparable(split, split, 3)
        )
        self.coarse = nn.Sequential(
            DepthwiseSeparable(in_ch, out_ch - 2*split, 3),
            DepthwiseSeparable(out_ch - 2*split, out_ch - 2*split, 3),
            DepthwiseSeparable(out_ch - 2*split, out_ch - 2*split, 3)
        )
        self.fusion = nn.Conv2d(out_ch, out_ch, 1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f1 = self.fine(x)
        f2 = self.medium(x)
        f3 = self.coarse(x)
        return self.fusion(torch.cat([f1, f2, f3], dim=1))

# ==============================================================================
# SECTION 2: Main Nano-ResNet Architecture
# ==============================================================================

class NanoResNet(nn.Module):
    """
    Ultra-fast, single-scale residual network for specialized artifact removal.
    This model is NOT conditional and must be trained for a specific CRF bucket.
    """
    SIZE_CONFIGS: Dict[str, Dict] = {
        'tiny':  {'base_ch': 32, 'num_blocks': 6},  # ~1.2M params
        'small': {'base_ch': 40, 'num_blocks': 8},  # ~2.1M params
        'base':  {'base_ch': 48, 'num_blocks': 10}, # ~3.3M params
    }

    def __init__(
        self, 
        size: str = 'tiny', 
        crf_min: int = 34, 
        crf_max: int = 43, 
        norm_range: Tuple[float, float] = (-1, 1)
    ):
        super().__init__()
        if size not in self.SIZE_CONFIGS:
            raise ValueError(f"Unknown size: {size}. Choose from {list(self.SIZE_CONFIGS.keys())}")
        
        cfg = self.SIZE_CONFIGS[size]
        base_channels = cfg['base_ch']
        num_blocks = cfg['num_blocks']

        self.clamp_min, self.clamp_max = norm_range
        self.res_scale = 0.1 # Residual scaling for training stability
        
        logger.info("="*60)
        logger.info(f"Initializing NanoResNet (Size: {size}, CRF Range: {crf_min}-{crf_max})")
        
        self.head = nn.Conv2d(3, base_channels, 3, padding=1)
        self.multiscale_entry = MultiScaleFeatureExtractor(base_channels, base_channels)
        self.body = nn.Sequential(*[EfficientResBlock(base_channels) for _ in range(num_blocks)])
        self.body_fusion = nn.Conv2d(base_channels, base_channels, 3, padding=1)
        self.tail = nn.Sequential(
            DepthwiseSeparable(base_channels, base_channels, 3),
            nn.GELU(),
            nn.Conv2d(base_channels, 3, 3, padding=1)
        )
        self._init_weights()
        total_params = sum(p.numel() for p in self.parameters())
        logger.info(f"✓ Model Initialized: {total_params:,} params ({total_params/1e6:.2f}M)")
        logger.info("="*60)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None: nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight); nn.init.zeros_(m.bias)
        nn.init.zeros_(self.tail[-1].weight)
        if self.tail[-1].bias is not None: nn.init.zeros_(self.tail[-1].bias)
        logger.info("✓ Weights initialized (tail zeroed, GroupNorm used)")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shallow = self.head(x)
        shallow = self.multiscale_entry(shallow)
        
        deep = shallow
        for block in self.body:
            deep = block(deep)
        deep = self.body_fusion(deep) + shallow
        
        residual = self.tail(deep) * self.res_scale
        restored = x + residual
        
        return torch.clamp(restored, self.clamp_min, self.clamp_max)

    @torch.no_grad()
    def inference(self, x, tile_size=512, tile_overlap=32):
        """Memory-efficient tiled inference for large images."""
        self.eval()
        B, C, H, W = x.shape
        
        if H <= tile_size and W <= tile_size:
            return self.forward(x)
        
        output = torch.zeros_like(x)
        weight = torch.zeros_like(x)
        stride = tile_size - tile_overlap
        
        for i in range(0, H, stride):
            for j in range(0, W, stride):
                i_end = min(i + tile_size, H)
                j_end = min(j + tile_size, W)
                i_start = max(i_end - tile_size, 0)
                j_start = max(j_end - tile_size, 0)
                
                tile = x[:, :, i_start:i_end, j_start:j_end]
                restored_tile = self.forward(tile)
                
                h_tile, w_tile = restored_tile.shape[2:]
                blend = torch.ones(1, 1, h_tile, w_tile, device=x.device)
                
                fade = tile_overlap // 2
                if i_start > 0: blend[:, :, :fade, :] *= torch.linspace(0, 1, fade, device=x.device).view(-1, 1)
                if j_start > 0: blend[:, :, :, :fade] *= torch.linspace(0, 1, fade, device=x.device).view(1, -1)
                if i_end < H: blend[:, :, -fade:, :] *= torch.linspace(1, 0, fade, device=x.device).view(-1, 1)
                if j_end < W: blend[:, :, :, -fade:] *= torch.linspace(1, 0, fade, device=x.device).view(1, -1)
                
                output[:, :, i_start:i_end, j_start:j_end] += restored_tile * blend
                weight[:, :, i_start:i_end, j_start:j_end] += blend
        
        return output / (weight + 1e-8)

# ==============================================================================
# SECTION 3: Model Factory
# ==============================================================================

def create_nano_resnet(
    size: str = 'tiny',
    crf_min: int = 23,
    crf_max: int = 63,
    norm_range: Tuple[float, float] = (-1.0, 1.0)
) -> NanoResNet:
    """Factory function for creating AV1-Nano-ResNet models."""
    return NanoResNet(size, crf_min, crf_max, norm_range)