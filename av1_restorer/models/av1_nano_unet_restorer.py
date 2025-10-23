# av1_restorer/models/av1_nano_unet_restorer.py
"""
AV1 Nano U-Net Restorer: Ultra-Lightweight 3-Level U-Net for AV1 Artifact Removal

==============================================================================
ARCHITECTURE OVERVIEW
==============================================================================

Design Philosophy:
    • Specialized per CRF range (train separate models for 23-33, 34-43, etc.)
    • 3-level shallow U-Net (no bottleneck → faster inference)
    • Depthwise separable convolutions only (8× parameter reduction)
    • Squeeze-Excitation for cheap channel attention
    • Multi-scale skip connection fusion
    • Residual learning (predict correction, not full image)

Architecture Flow:
    Input (3ch) → Head (base_ch + 2 ResBlocks)
                → Encoder L1 (↓2×, base*2)  ─┐
                → Encoder L2 (↓2×, base*4)  ─┼─ Skip Connections
                → Encoder L3 (↓2×, base*8)  ─┘
                → Decoder L3 (↑2×, base*4) ←─┘
                → Decoder L2 (↑2×, base*2) ←─┘
                → Decoder L1 (↑2×, base)   ←─┘
                → Tail (predict residual)
                → Output = Input + Residual

==============================================================================
Author: Soham Mukherjee
Version: 2.0 (Professional)
License: MIT
==============================================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict
import logging

logger = logging.getLogger(__name__)


# ==============================================================================
# SECTION 1: Core Building Blocks
# ==============================================================================

class DepthwiseSeparable(nn.Module):
    """
    Depthwise Separable Convolution: 8× parameter reduction vs standard conv.
    
    Flow: Depthwise (spatial, per-channel) → Pointwise (1×1, channel mixing)
          → BatchNorm → GELU
    
    Args:
        in_ch: Input channels
        out_ch: Output channels
        stride: Convolution stride (1 for same size, 2 for downsampling)
    """
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_ch, in_ch, kernel_size=3, stride=stride, 
            padding=1, groups=in_ch, bias=False
        )
        self.pointwise = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.GELU()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.pointwise(self.depthwise(x))))


class SqueezeExcitation(nn.Module):
    """
    Squeeze-and-Excitation: Efficient channel attention (Hu et al., CVPR 2018).
    
    Mechanism: Global pooling → FC compress → ReLU → FC expand → Sigmoid → Scale
    Parameters: <1% of total, but improves quality by 5-10%
    
    Args:
        channels: Number of channels
        reduction: Compression ratio (higher = fewer params, less capacity)
    """
    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.excitation(self.squeeze(x))


class MicroResBlock(nn.Module):
    """
    Efficient residual block: 2× DWS Conv + SE attention.
    
    Parameters: ~2C² vs standard ResBlock: ~9C²
    
    Flow: Input → DWSConv → DWSConv → SE → Add(input)
    """
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = DepthwiseSeparable(channels, channels)
        self.conv2 = DepthwiseSeparable(channels, channels)
        self.se = SqueezeExcitation(channels, reduction=8)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.se(out)
        return identity + out


class MultiScaleFusion(nn.Module):
    """
    Multi-scale feature fusion for skip connections.
    
    Concatenates upsampled decoder features with encoder skip features,
    then fuses with 1×1 conv. Preserves both high and low frequency details.
    
    Args:
        channels: Channel count (must match for both inputs)
    """
    def __init__(self, channels: int):
        super().__init__()
        self.fusion = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.GELU()
        )
    
    def forward(self, x_up: torch.Tensor, x_skip: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x_up: Upsampled features from decoder
            x_skip: Skip connection from encoder
        Returns:
            Fused features
        """
        return self.fusion(torch.cat([x_up, x_skip], dim=1))


# ==============================================================================
# SECTION 2: Main U-Net Architecture
# ==============================================================================

class AV1NanoUnetRestorer(nn.Module):
    """
    Ultra-lightweight 3-level U-Net for specialized CRF range artifact removal.
    
    Key Design Decisions:
        • 3 encoder/decoder levels (vs 5 in full U-Net) → speed
        • No bottleneck self-attention → avoid computational cost
        • Depthwise separable convs everywhere → parameter efficiency
        • SE blocks for channel attention → cheap quality boost
        • Non-conditional → train separate models per CRF bucket
    
    Model Sizes:
        nano:  20 base channels, [2,2,2] blocks → ~0.2M params
        tiny:  24 base channels, [2,2,3] blocks → ~0.5M params
        small: 32 base channels, [2,3,4] blocks → ~1.2M params
        base:  48 base channels, [3,3,4] blocks → ~2.5M params
        large: 64 base channels, [4,4,6] blocks → ~6.2M params
        huge:  64 base channels, [4,6,12] blocks → ~11.2M params

    
    Args:
        size: Model variant ('tiny', 'small', 'base', 'large')
        crf_min: Minimum CRF value this model handles
        crf_max: Maximum CRF value this model handles
        norm_range: Image normalization range ((-1,1) or (0,1))
    
    Example:
        >>> # Create model for medium compression (CRF 34-43)
        >>> model = AV1NanoUnetRestorer('small', crf_min=34, crf_max=43)
        >>> x = torch.randn(1, 3, 1080, 1920)
        >>> restored = model.inference(x)
    """
    
    SIZE_CONFIGS: Dict[str, Dict] = {
        'nano': {
            'base_ch': 16,
            'blocks_per_level': [2, 2, 2],  # [enc1, enc2, enc3]
        },
        'tiny': {
            'base_ch': 24,
            'blocks_per_level': [2, 2, 3],  
        },
        'small': {
            'base_ch': 32,
            'blocks_per_level': [2, 3, 4],
        },
        'base': {
            'base_ch': 48,
            'blocks_per_level': [3, 3, 4],
        },
        'large': {
            'base_ch': 64,
            'blocks_per_level': [4, 4, 6],
        },
        'huge': {
            'base_ch': 64,
            'blocks_per_level': [6, 8, 12],
        },

    }
    
    def __init__(
        self,
        size: str = 'small',
        crf_min: int = 23,
        crf_max: int = 63,
        norm_range: Tuple[float, float] = (-1, 1)
    ):
        super().__init__()
        
        if size not in self.SIZE_CONFIGS:
            raise ValueError(
                f"Unknown size: '{size}'. "
                f"Choose from: {list(self.SIZE_CONFIGS.keys())}"
            )
        
        cfg = self.SIZE_CONFIGS[size]
        base = cfg['base_ch']
        blocks = cfg['blocks_per_level']
        
        self.size = size
        self.crf_min = crf_min
        self.crf_max = crf_max
        self.clamp_min, self.clamp_max = norm_range
        
        logger.info("=" * 60)
        logger.info(f"Initializing AV1NanoUnetRestorer ({size})")
        logger.info(f"  Base channels: {base}")
        logger.info(f"  Blocks per level: {blocks}")
        logger.info(f"  CRF Range: {crf_min}-{crf_max}")
        logger.info(f"  Normalization: [{norm_range[0]}, {norm_range[1]}]")
        logger.info("=" * 60)
        
        # ===== Input Head: 3ch → base_ch with initial processing =====
        self.head = nn.Sequential(
            nn.Conv2d(3, base, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(base),
            nn.GELU(),
            *[MicroResBlock(base) for _ in range(2)]
        )
        
        # ===== Encoder Path (3 levels with downsampling) =====
        # Level 1: base → base*2, H/2
        self.enc1 = nn.ModuleList([
            DepthwiseSeparable(base, base*2, stride=2),
            *[MicroResBlock(base*2) for _ in range(blocks[0])]
        ])
        
        # Level 2: base*2 → base*4, H/4
        self.enc2 = nn.ModuleList([
            DepthwiseSeparable(base*2, base*4, stride=2),
            *[MicroResBlock(base*4) for _ in range(blocks[1])]
        ])
        
        # Level 3: base*4 → base*8, H/8
        self.enc3 = nn.ModuleList([
            DepthwiseSeparable(base*4, base*8, stride=2),
            *[MicroResBlock(base*8) for _ in range(blocks[2])]
        ])
        
        # ===== Decoder Path (3 levels with upsampling + skip fusion) =====
        # Level 3: base*8 → base*4, H/4, fuse with enc2 skip
        self.dec3_up = nn.Sequential(
            nn.ConvTranspose2d(base*8, base*4, kernel_size=2, stride=2, bias=False),
            nn.BatchNorm2d(base*4),
            nn.GELU()
        )
        self.dec3_fusion = MultiScaleFusion(base*4)
        self.dec3_blocks = nn.Sequential(
            *[MicroResBlock(base*4) for _ in range(blocks[1])]
        )
        
        # Level 2: base*4 → base*2, H/2, fuse with enc1 skip
        self.dec2_up = nn.Sequential(
            nn.ConvTranspose2d(base*4, base*2, kernel_size=2, stride=2, bias=False),
            nn.BatchNorm2d(base*2),
            nn.GELU()
        )
        self.dec2_fusion = MultiScaleFusion(base*2)
        self.dec2_blocks = nn.Sequential(
            *[MicroResBlock(base*2) for _ in range(blocks[0])]
        )
        
        # Level 1: base*2 → base, H, fuse with head skip
        self.dec1_up = nn.Sequential(
            nn.ConvTranspose2d(base*2, base, kernel_size=2, stride=2, bias=False),
            nn.BatchNorm2d(base),
            nn.GELU()
        )
        self.dec1_fusion = MultiScaleFusion(base)
        self.dec1_blocks = nn.Sequential(
            *[MicroResBlock(base) for _ in range(2)]
        )
        
        # ===== Output Tail: Predict 3-channel residual =====
        self.tail = nn.Sequential(
            MicroResBlock(base),
            nn.Conv2d(base, 3, kernel_size=3, padding=1)
        )
        
        self._init_weights()
        
        # Log statistics
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logger.info(f"✓ Model initialized")
        logger.info(f"  Total: {total_params:,} ({total_params/1e6:.2f}M) params")
        logger.info(f"  Trainable: {trainable_params:,} params")
        logger.info(f"  Size (FP32): ~{total_params * 4 / 1e6:.1f} MB")
        logger.info("=" * 60)
    
    def _init_weights(self):
        """
        Initialize weights for stable training.
        
        Strategy:
            • Conv/ConvTranspose: Kaiming (He) init for ReLU-family activations
            • BatchNorm: ones (weight) + zeros (bias)
            • Final tail: zero init → network outputs 0 at start (identity map)
        """
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
        
        # Zero-init final layer for residual learning
        nn.init.zeros_(self.tail[-1].weight)
        nn.init.zeros_(self.tail[-1].bias)
        logger.info("✓ Weights initialized (tail zeroed for residual learning)")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: Input + Predicted Residual.
        
        Args:
            x: [B, 3, H, W] Input image in norm_range
        
        Returns:
            [B, 3, H, W] Restored image, clamped to norm_range
        
        Flow:
            1. Encoder: Extract multi-scale features with skip storage
            2. Decoder: Upsample and fuse with skip connections
            3. Tail: Predict residual correction
            4. Output: Input + Residual (residual learning)
        """
        # Encoder path with skip storage
        skip0 = self.head(x)  # [B, base, H, W]
        
        enc1 = skip0
        for layer in self.enc1:
            enc1 = layer(enc1)
        skip1 = enc1  # [B, base*2, H/2, W/2]
        
        enc2 = skip1
        for layer in self.enc2:
            enc2 = layer(enc2)
        skip2 = enc2  # [B, base*4, H/4, W/4]
        
        enc3 = skip2
        for layer in self.enc3:
            enc3 = layer(enc3)  # [B, base*8, H/8, W/8]
        
        # Decoder path with skip fusion
        dec3 = self.dec3_up(enc3)
        dec3 = self.dec3_fusion(dec3, skip2)
        dec3 = self.dec3_blocks(dec3)
        
        dec2 = self.dec2_up(dec3)
        dec2 = self.dec2_fusion(dec2, skip1)
        dec2 = self.dec2_blocks(dec2)
        
        dec1 = self.dec1_up(dec2)
        dec1 = self.dec1_fusion(dec1, skip0)
        dec1 = self.dec1_blocks(dec1)
        
        # Predict residual and add to input
        residual = self.tail(dec1)
        restored = x + residual
        
        return torch.clamp(restored, self.clamp_min, self.clamp_max)
    
    @torch.no_grad()
    def inference(self, x: torch.Tensor) -> torch.Tensor:
        """
        Fast inference with automatic padding.
        
        Pads input to multiple of 8 (2³ for 3 downsample levels), processes,
        then crops back to original size to avoid size mismatches.
        
        Args:
            x: [B, 3, H, W] Input image (any size)
        
        Returns:
            [B, 3, H, W] Restored image
        """
        self.eval()
        B, C, H, W = x.shape
        
        # Calculate padding needed (must be divisible by 8)
        pad_h = (8 - H % 8) % 8
        pad_w = (8 - W % 8) % 8
        
        if pad_h > 0 or pad_w > 0:
            # Pad, process, crop
            x_padded = F.pad(x, (0, pad_w, 0, pad_h), mode='reflect')
            output = self.forward(x_padded)
            return output[:, :, :H, :W]
        
        return self.forward(x)


# ==============================================================================
# SECTION 3: Model Factory
# ==============================================================================

def create_av1_nano_unet_restorer(
    size: str = 'small',
    crf_min: int = 23,
    crf_max: int = 63,
    norm_range: Tuple[float, float] = (-1, 1)
) -> AV1NanoUnetRestorer:
    """
    Factory function for creating AV1 Nano U-Net models.
    
    Recommended Strategy: Train separate models per CRF bucket
        • Low:    CRF 23-33 (light artifacts)
        • Medium: CRF 34-43 (moderate artifacts)
        • High:   CRF 44-53 (heavy artifacts)
        • Extreme: CRF 54-63 (severe artifacts)
    
    Args:
        size: 'tiny' (0.8M), 'small' (1.5M), or 'base' (2.5M)
        crf_min: Minimum CRF value for specialization
        crf_max: Maximum CRF value for specialization
        norm_range: Image normalization range
    
    Returns:
        Initialized AV1NanoUnetRestorer model
    
    Example:
        >>> # Create models for each compression tier
        >>> model_med = create_av1_nano_unet_restorer(
        ...     size='small', crf_min=34, crf_max=43
        ... )
        >>> 
        >>> # At inference, select based on known CRF
        >>> if 34 <= crf <= 43:
        ...     restored = model_med.inference(compressed_image)
    """
    logger.info(f"Creating AV1NanoUnetRestorer (size={size}, CRF={crf_min}-{crf_max})")
    return AV1NanoUnetRestorer(size, crf_min, crf_max, norm_range)


# ==============================================================================
# SECTION 4: Testing & Benchmarking
# ==============================================================================

if __name__ == "__main__":
    import logging
    import time
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)s | %(message)s'
    )
    
    print("\n" + "=" * 70)
    print("AV1 Nano U-Net Restorer - Model Testing")
    print("=" * 70 + "\n")
    
    # Test all model sizes
    for size in ['nano', 'tiny', 'small', 'base', 'large', 'huge']:        
        print(f"\n{'=' * 70}")
        print(f"Testing: {size.upper()} unet variant")
        print('=' * 70)
        
        # Create model
        model = create_av1_nano_unet_restorer(
            size=size,
            crf_min=23,
            crf_max=33,
            norm_range=(-1, 1)
        )
        model.eval()
        
        # Parameter count
        total = sum(p.numel() for p in model.parameters())
        print(f"\nParameters: {total:,} ({total/1e6:.2f}M)")
        print(f"Model Size: ~{total * 4 / 1e6:.1f} MB (FP32), ~{total * 2 / 1e6:.1f} MB (FP16)")
        
        # Test forward pass
        print(f"\nForward Pass Test:")
        x = torch.randn(2, 3, 256, 256)
        with torch.no_grad():
            output = model(x)
        
        print(f"  Input:  {x.shape}, range [{x.min():.3f}, {x.max():.3f}]")
        print(f"  Output: {output.shape}, range [{output.min():.3f}, {output.max():.3f}]")
        assert output.shape == x.shape, "Shape mismatch"
        assert output.min() >= -1.0 and output.max() <= 1.0, "Range violation"
        
        # Test inference with odd dimensions
        print(f"\nOdd Dimensions Test:")
        x_odd = torch.randn(1, 3, 333, 555)
        with torch.no_grad():
            output_odd = model.inference(x_odd)
        print(f"  {x_odd.shape} → {output_odd.shape} ✓")
        assert output_odd.shape == x_odd.shape, "Inference shape mismatch"
        
        # Speed benchmark
        print(f"\nSpeed Benchmark (1080p, 10 iterations, CPU):")
        x_1080p = torch.randn(1, 3, 1080, 1920)
        
        with torch.no_grad():
            # Warmup
            _ = model.inference(x_1080p)
            
            # Benchmark
            times = []
            for _ in range(10):
                start = time.time()
                _ = model.inference(x_1080p)
                times.append(time.time() - start)
        
        avg_time = sum(times) / len(times)
        print(f"  Avg time: {avg_time * 1000:.1f} ms ({1.0/avg_time:.1f} fps)")
        
        print(f"\n✓ All tests passed for {size.upper()}")
    
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    print(f"\n{'Model':<10} {'Params':<12} {'Use Case'}")
    print("-" * 70)
    print(f"{'nano':<10} {'~0.2M':<12} {'Mobile, embedded (fastest)'}")
    print(f"{'tiny':<10} {'~0.5M':<12} {'Light tasks (e.g. CRF 23-33)'}")
    print(f"{'small':<10} {'~1.2M':<12} {'Balanced (e.g. CRF 34-43)'}")
    print(f"{'base':<10} {'~2.5M':<12} {'High quality (e.g. CRF 44-53)'}")
    print(f"{'large':<10} {'~6.0M':<12} {'Heavy tasks (e.g. CRF 54-63)'}")
    print(f"{'huge':<10} {'~11.2M':<12} {'Max quality (extreme CRF)'}")
    print("\n" + "=" * 70)
    print("✓ All tests passed!\n")