# av1_restorer/models/av1_fbcnn_restorer.py
"""
AV1-FBCNN-Restorer: SOTA Ultra-Lightweight Artifact Removal (v2.1)

FBCNN-inspired shallow U-Net for specialized CRF range artifact removal.
This architecture is a U-Net, but built with hyper-efficient blocks
and SOTA stability features.

Architecture:
- 3-level shallow U-Net
- GroupNorm (replaces BatchNorm for small-batch stability)
- Upsample + DepthwiseSeparable (no checkerboard artifacts)
- Efficient Channel Attention (ECA)
- Residual scaling for stable training

Author: Soham Mukherjee
Version: 2.1 (Class rename)
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

class MicroRes(nn.Module):
    """
    Lightweight residual block using DWS convs + ECA.
    """
    def __init__(self, ch: int):
        super().__init__()
        self.conv1 = DepthwiseSeparable(ch, ch)
        self.conv2 = DepthwiseSeparable(ch, ch)
        self.attn = ECA(ch)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.attn(out)
        return x + out

class MultiScaleFusion(nn.Module):
    """Fuses features from decoder upsampling and encoder skip connection."""
    def __init__(self, channels: int, num_groups: int = 16):
        super().__init__()
        if channels % num_groups != 0: # Auto-adjust num_groups
            num_groups = 1
        self.fusion = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size=1, bias=False),
            nn.GroupNorm(num_groups, channels),
            nn.GELU()
        )
    def forward(self, x_up: torch.Tensor, x_skip: torch.Tensor) -> torch.Tensor:
        fused = torch.cat([x_up, x_skip], dim=1)
        return self.fusion(fused)

# ==============================================================================
# SECTION 2: Main Nano-FBCNN Architecture
# ==============================================================================

class AV1FBCNNRestorer(nn.Module):
    """
    Ultra-fast 3-level U-Net for specialized CRF range artifact removal.
    Inspired by FBCNN principles (efficient, gated processing) and
    SOTA stability features (GroupNorm, Upsample+Conv, ECA, Residual Scaling).
    """

    SIZE_CONFIGS: Dict[str, Dict] = {
        'tiny':  {'base_ch': 16, 'blocks': [1, 1, 2]},  # ~0.8M params
        'small': {'base_ch': 24, 'blocks': [1, 2, 2]},  # ~1.8M params
        'base':  {'base_ch': 32, 'blocks': [2, 2, 3]},  # ~3.4M params
    }

    def __init__(
        self, 
        size: str = 'small', 
        crf_min: int = 23, 
        crf_max: int = 33, 
        norm_range: Tuple[float,float]=(-1,1)
    ):
        super().__init__()
        if size not in self.SIZE_CONFIGS:
            raise ValueError(f"Unknown size: {size}. Choose from {list(self.SIZE_CONFIGS.keys())}")
        cfg = self.SIZE_CONFIGS[size]
        base, blocks = cfg['base_ch'], cfg['blocks']
        
        self.clamp_min, self.clamp_max = norm_range
        self.res_scale = 0.1 # Residual scaling for training stability

        logger.info("="*60)
        logger.info(f"Initializing AV1FBCNNRestorer (Size: {size}, CRF Range: {crf_min}-{crf_max})")

        # --- Encoder ---
        self.head = nn.Sequential(
            nn.Conv2d(3, base, 3, 1, 1, bias=False),
            nn.GroupNorm(16 if base >= 16 else 1, base),
            nn.GELU(),
            *[MicroRes(base) for _ in range(2)]
        )
        self.enc1_down = DepthwiseSeparable(base, base*2, stride=2)
        self.enc1_body = nn.Sequential(*[MicroRes(base*2) for _ in range(blocks[0])])
        
        self.enc2_down = DepthwiseSeparable(base*2, base*4, stride=2)
        self.enc2_body = nn.Sequential(*[MicroRes(base*4) for _ in range(blocks[1])])
        
        self.enc3_down = DepthwiseSeparable(base*4, base*8, stride=2)
        self.enc3_body = nn.Sequential(*[MicroRes(base*8) for _ in range(blocks[2])])

        # --- Gating (FBCNN-inspired) ---
        self.gate = nn.Sequential(
            nn.Conv2d(base*8, base*8, 3, 1, 1, bias=False),
            nn.GroupNorm(16, base*8), # 16 groups for base*8
            nn.Sigmoid()
        )

        # --- Decoder (UPDATED: Upsample + Conv) ---
        self.dec3_up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            DepthwiseSeparable(base*8, base*4)
        )
        self.dec3_fusion = MultiScaleFusion(base*4)
        self.dec3_body = nn.Sequential(*[MicroRes(base*4) for _ in range(blocks[1])])
        
        self.dec2_up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            DepthwiseSeparable(base*4, base*2)
        )
        self.dec2_fusion = MultiScaleFusion(base*2)
        self.dec2_body = nn.Sequential(*[MicroRes(base*2) for _ in range(blocks[0])])

        self.dec1_up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            DepthwiseSeparable(base*2, base)
        )
        self.dec1_fusion = MultiScaleFusion(base)
        self.dec1_body = nn.Sequential(*[MicroRes(base) for _ in range(2)])

        # --- Output Tail ---
        self.tail = nn.Sequential(
            MicroRes(base),
            nn.Conv2d(base, 3, 3, 1, 1)
        )
        self._init_weights()
        total_params = sum(p.numel() for p in self.parameters())
        logger.info(f"✓ Model Initialized: {total_params:,} params ({total_params/1e6:.2f}M)")
        logger.info("="*60)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
            elif isinstance(m, nn.GroupNorm):
                nn.init.ones_(m.weight); nn.init.zeros_(m.bias)
        nn.init.zeros_(self.tail[-1].weight)
        if self.tail[-1].bias is not None: nn.init.zeros_(self.tail[-1].bias)
        logger.info("✓ Weights initialized (tail zeroed, GroupNorm used)")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: input + predicted residual.
        
        Args:
            x: [B, 3, H, W] Input image in norm_range (NON-CONDITIONAL)
        
        Returns:
            [B, 3, H, W] Restored image
        """
        skip0 = self.head(x)
        skip1 = self.enc1_body(self.enc1_down(skip0))
        skip2 = self.enc2_body(self.enc2_down(skip1))
        
        enc3 = self.enc3_body(self.enc3_down(skip2))
        gated = enc3 * self.gate(enc3) # Gating mechanism
        
        dec3_up = self.dec3_up(gated)
        dec3 = self.dec3_fusion(dec3_up, skip2)
        dec3 = self.dec3_body(dec3)
        
        dec2_up = self.dec2_up(dec3)
        dec2 = self.dec2_fusion(dec2_up, skip1)
        dec2 = self.dec2_body(dec2)
        
        dec1_up = self.dec1_up(dec2)
        dec1 = self.dec1_fusion(dec1_up, skip0)
        dec1 = self.dec1_body(dec1)
        
        residual = self.tail(dec1) * self.res_scale
        out = x + residual
        return torch.clamp(out, self.clamp_min, self.clamp_max)

    @torch.no_grad()
    def inference(self, x: torch.Tensor) -> torch.Tensor:
        """
        Fast inference with automatic padding.
        
        Pads to multiple of 8 (2^3 for 3 downsample levels).
        """
        self.eval()
        B, C, H, W = x.shape
        pad_h = (8 - H % 8) % 8
        pad_w = (8 - W % 8) % 8
        if pad_h > 0 or pad_w > 0:
            x_p = F.pad(x, (0, pad_w, 0, pad_h), mode='reflect')
            out = self.forward(x_p)
            return out[:, :, :H, :W]
        return self.forward(x)

# ==============================================================================
# SECTION 3: Model Factory
# ==============================================================================

def create_av1_nano_fbcnn_restorer(
    size: str = 'small', 
    crf_min: int = 23, 
    crf_max: int = 33, 
    norm_range: Tuple[float,float]=(-1,1)
) -> AV1FBCNNRestorer:
    """Factory function for creating AV1-FBCNN-Restorer models."""
    return AV1FBCNNRestorer(size=size, crf_min=crf_min, crf_max=crf_max, norm_range=norm_range)
