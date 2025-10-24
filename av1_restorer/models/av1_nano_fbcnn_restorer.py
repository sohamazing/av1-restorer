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

from .blocks import DepthwiseSeparable, MicroResBlock, MultiScaleFusion, ECA, choose_num_groups

# ==============================================================================
# SECTION 1: Main Nano-FBCNN Architecture
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
            *[MicroResBlock(base) for _ in range(2)]
        )
        self.enc1_down = DepthwiseSeparable(base, base*2, stride=2)
        self.enc1_body = nn.Sequential(*[MicroResBlock(base*2) for _ in range(blocks[0])])
        
        self.enc2_down = DepthwiseSeparable(base*2, base*4, stride=2)
        self.enc2_body = nn.Sequential(*[MicroResBlock(base*4) for _ in range(blocks[1])])
        
        self.enc3_down = DepthwiseSeparable(base*4, base*8, stride=2)
        self.enc3_body = nn.Sequential(*[MicroResBlock(base*8) for _ in range(blocks[2])])

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
        self.dec3_body = nn.Sequential(*[MicroResBlock(base*4) for _ in range(blocks[1])])
        
        self.dec2_up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            DepthwiseSeparable(base*4, base*2)
        )
        self.dec2_fusion = MultiScaleFusion(base*2)
        self.dec2_body = nn.Sequential(*[MicroResBlock(base*2) for _ in range(blocks[0])])

        self.dec1_up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            DepthwiseSeparable(base*2, base)
        )
        self.dec1_fusion = MultiScaleFusion(base)
        self.dec1_body = nn.Sequential(*[MicroResBlock(base) for _ in range(2)])

        # --- Output Tail ---
        self.tail = nn.Sequential(
            MicroResBlock(base),
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
# SECTION 2: Model Factory
# ==============================================================================

def create_av1_nano_fbcnn_restorer(
    size: str = 'small', 
    crf_min: int = 23, 
    crf_max: int = 33, 
    norm_range: Tuple[float,float]=(-1,1)
) -> AV1FBCNNRestorer:
    """Factory function for creating AV1-FBCNN-Restorer models."""
    return AV1FBCNNRestorer(size=size, crf_min=crf_min, crf_max=crf_max, norm_range=norm_range)
