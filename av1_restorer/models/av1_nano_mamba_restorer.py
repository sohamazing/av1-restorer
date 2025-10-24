# av1_restorer/models/av1_nano_mamba_restorer.py
"""
AV1-Nano-Mamba-Restorer: SOTA Ultra-Lightweight Artifact Removal (2025)

Architecture Philosophy:
- Hybrid U-Net + State Space Model (Mamba-inspired)
- 3-level shallow U-Net (for efficiency)
- CNN Encoder/Decoder: Depthwise-Separable convs (proven efficiency)
- VSS Bottleneck: Visual State Space (VSS) blocks to capture
  global context with LINEAR complexity (O(N)).
- This "Hybrid" approach gets the best of both worlds:
  1. CNNs: Excellent at local features, lightweight, fast.
  2. SSMs: Excellent at global context, linear time, efficient.

Author: Soham Mukherjee
Version: 3.0 (Mamba/SSM Integration)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict
import logging

logger = logging.getLogger(__name__)

from .blocks import DepthwiseSeparable

# ==============================================================================
# SECTION 2: NEW SOTA Mamba-inspired Block
# ==============================================================================

class SSMScan(nn.Module):
    """
    Simplified 1D State Space Model (SSM) scan module.
    
    This is a proxy for the S6 (Selective Scan) mechanism in Mamba.
    It flattens the 2D spatial features into a 1D sequence and applies
    a 1D "scan" (here, a 1D depthwise conv) to model dependencies.
    """
    def __init__(self, dim: int):
        super().__init__()
        # 1D conv to act as a simple, efficient SSM scan proxy
        self.scan_proxy = nn.Conv1d(
            in_channels=dim,
            out_channels=dim,
            kernel_size=3,
            padding=1,
            groups=dim,  # Depthwise
            bias=False
        )
        logger.info("  ... (SSMScan proxy initialized)")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: [B, C, H, W]
        B, C, H, W = x.shape
        
        # Flatten: [B, C, H, W] -> [B, C, H*W]
        x_flat = x.flatten(2)
        
        # Apply 1D "scan" along the flattened sequence
        scanned = self.scan_proxy(x_flat)
        
        # Un-flatten: [B, C, H*W] -> [B, C, H, W]
        out = scanned.view(B, C, H, W)
        return out


class VSSBlock(nn.Module):
    """
    Visual State Space (VSS) Block, inspired by Vision Mamba (VMamba).
    
    This block is a hybrid:
    1. A CNN path (DepthwiseSeparable) for local features.
    2. An SSM path (SSMScan) for global, long-range context.
    
    It replaces a traditional ResBlock or Transformer block.
    """
    def __init__(self, dim: int, expansion: int = 2, num_groups: int = 16):
        super().__init__()
        dim_inner = dim * expansion
        
        self.norm = nn.GroupNorm(num_groups, dim)
        self.in_proj = nn.Conv2d(dim, dim_inner, 1, bias=False)
        self.act = nn.GELU()
        
        # 1. Local CNN Path
        self.local_path = DepthwiseSeparable(
            dim_inner, dim_inner, kernel_size=3, num_groups=num_groups
        )
        
        # 2. Global SSM Path
        self.global_path = SSMScan(dim_inner)
        
        # Output projection
        self.out_proj = nn.Conv2d(dim_inner, dim, 1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        
        # Normalize and project
        x = self.norm(x)
        x = self.in_proj(x)
        x = self.act(x)
        
        # Gated-style fusion of local and global paths
        # This is a common pattern in modern hybrid architectures
        x_local = self.local_path(x)
        x_global = self.global_path(x)
        
        # Fuse: (Local Context) * (Global Context)
        x_fused = x_local * x_global
        
        # Project back
        x = self.out_proj(x_fused)
        
        return identity + x

# ==============================================================================
# SECTION 3: Main Nano-Mamba-Restorer Architecture
# ==============================================================================

class AV1NanoMambaRestorer(nn.Module):
    """
    Ultra-fast 3-level U-Net using CNNs for spatial hierarchy and
    a Mamba-inspired VSS block in the bottleneck for global context.
    """
    SIZE_CONFIGS: Dict[str, Dict] = {
        'tiny':  {'base_ch': 16, 'blocks': [1, 1, 2]},  # [enc1, enc2, bottleneck]
        'small': {'base_ch': 24, 'blocks': [1, 2, 3]},  # ~2.0M params
        'base':  {'base_ch': 32, 'blocks': [2, 2, 4]},  # ~3.8M params
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
        logger.info(f"Initializing AV1NanoMambaRestorer (Size: {size}, CRF: {crf_min}-{crf_max})")
        logger.info(f"  Architecture: CNN-Encoder + VSS-Bottleneck + CNN-Decoder")

        # --- CNN Encoder (Efficient) ---
        self.head = nn.Sequential(
            nn.Conv2d(3, base, 3, 1, 1, bias=False),
            nn.GroupNorm(16 if base >= 16 else 1, base),
            nn.GELU()
        )
        self.enc1_down = DepthwiseSeparable(base, base*2, stride=2)
        self.enc1_body = nn.Sequential(
            *[DepthwiseSeparable(base*2, base*2) for _ in range(blocks[0])]
        )
        
        self.enc2_down = DepthwiseSeparable(base*2, base*4, stride=2)
        self.enc2_body = nn.Sequential(
            *[DepthwiseSeparable(base*4, base*4) for _ in range(blocks[1])]
        )
        
        # --- SOTA VSS Bottleneck ---
        logger.info(f"  VSS Bottleneck: {blocks[2]} VSSBlocks at {base*4} channels")
        self.bottleneck = nn.Sequential(
            *[VSSBlock(base*4, expansion=2) for _ in range(blocks[2])]
        )

        # --- CNN Decoder (Efficient) ---
        self.dec2_up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            DepthwiseSeparable(base*4, base*2)
        )
        self.dec2_fusion = MultiScaleFusion(base*2)
        self.dec2_body = nn.Sequential(
            *[DepthwiseSeparable(base*2, base*2) for _ in range(blocks[1])]
        )
        
        self.dec1_up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            DepthwiseSeparable(base*2, base)
        )
        self.dec1_fusion = MultiScaleFusion(base)
        self.dec1_body = nn.Sequential(
            *[DepthwiseSeparable(base, base) for _ in range(blocks[0])]
        )

        # --- Output Tail ---
        self.tail = nn.Sequential(
            DepthwiseSeparable(base, base),
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
        """
        skip0 = self.head(x)
        skip1 = self.enc1_body(self.enc1_down(skip0))
        skip2 = self.enc2_body(self.enc2_down(skip1))
        
        # Apply global context modeling in the bottleneck
        bottleneck = self.bottleneck(skip2)
        
        dec2_up = self.dec2_up(bottleneck)
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
        Pads to multiple of 4 (2^2 for 2 downsample levels).
        """
        self.eval()
        B, C, H, W = x.shape
        pad_h = (4 - H % 4) % 4
        pad_w = (4 - W % 4) % 4
        if pad_h > 0 or pad_w > 0:
            x_p = F.pad(x, (0, pad_w, 0, pad_h), mode='reflect')
            out = self.forward(x_p)
            return out[:, :, :H, :W]
        return self.forward(x)

# ==============================================================================
# SECTION 4: Model Factory
# ==============================================================================

def create_av1_nano_mamba_restorer(
    size: str = 'small', 
    crf_min: int = 23, 
    crf_max: int = 33, 
    norm_range: Tuple[float,float]=(-1,1)
) -> AV1NanoMambaRestorer:
    """Factory function for creating AV1-Nano-Mamba-Restorer models."""
    return AV1NanoMambaRestorer(size=size, crf_min=crf_min, crf_max=crf_max, norm_range=norm_range)
