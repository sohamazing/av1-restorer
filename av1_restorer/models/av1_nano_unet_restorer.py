"""
AV1 Nano U-Net Restorer: Ultra-Lightweight 3-Level U-Net for AV1 Artifact Removal
Finalized Ultimate Lightweight Nano UNet:
  - BatchNorm -> GroupNorm (robust for small batches)
  - SE -> ECA (Efficient Channel Attention)
  - Depthwise separable convs with GroupNorm + GELU
  - Residual scaling on final predicted residual for stability

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
Version: 2.1 (Optimized)
License: MIT
"""
import logging
from typing import Tuple, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

from .blocks import DepthwiseSeparable, ECA, MicroResBlock, MultiScaleFusion, choose_num_groups

# ==============================================================================
# SECTION 2: AV1NanoUnetRestorer (uses upgraded blocks)
# ==============================================================================

class AV1NanoUnetRestorer(nn.Module):
    """
    Ultra-lightweight 3-level U-Net for specialized CRF range artifact removal.
    - Replaces BatchNorm with GroupNorm.
    - Uses ECA instead of SE for attention.
    - Depthwise separable convs everywhere.
    - Predicts a scaled residual which is added to the input.

    Key Design Decisions:
        • 3 encoder/decoder levels (vs 5 in full U-Net) → speed
        • No bottleneck self-attention → avoid computational cost
        • Depthwise separable convs everywhere → parameter efficiency
        • SE blocks for channel attention → cheap quality boost
        • Non-conditional → train separate models per CRF bucket
    """

    SIZE_CONFIGS: Dict[str, Dict] = {
        'nano':  {'base_ch': 16, 'blocks_per_level': [2, 2, 2]},
        'tiny':  {'base_ch': 24, 'blocks_per_level': [2, 2, 3]},
        'small': {'base_ch': 32, 'blocks_per_level': [2, 3, 4]},
        'base':  {'base_ch': 48, 'blocks_per_level': [3, 3, 4]},
        'large': {'base_ch': 64, 'blocks_per_level': [4, 4, 6]},
        'huge':  {'base_ch': 64, 'blocks_per_level': [6, 8, 12]},
    }

    def __init__(
        self,
        size: str = 'small',
        crf_min: int = 23,
        crf_max: int = 63,
        norm_range: Tuple[float, float] = (-1, 1),
        res_scale: float = 0.1,
        gn_groups: int = 16,
    ):
        super().__init__()

        if size not in self.SIZE_CONFIGS:
            raise ValueError(f"Unknown size: '{size}'. Choose from: {list(self.SIZE_CONFIGS.keys())}")

        cfg = self.SIZE_CONFIGS[size]
        base = cfg['base_ch']
        blocks = cfg['blocks_per_level']

        self.size = size
        self.crf_min = crf_min
        self.crf_max = crf_max
        self.clamp_min, self.clamp_max = norm_range
        self.res_scale = nn.Parameter(torch.tensor(res_scale, dtype=torch.float32))
        self._gn_groups = gn_groups

        logger.info("=" * 60)
        logger.info(f"Initializing AV1NanoUnetRestorer ({size})")
        logger.info(f"  Base channels: {base}")
        logger.info(f"  Blocks per level: {blocks}")
        logger.info(f"  CRF Range: {crf_min}-{crf_max}")
        logger.info(f"  Normalization: [{norm_range[0]}, {norm_range[1]}]")
        logger.info(f"  Residual scale: {self.res_scale}")
        logger.info("=" * 60)

        # ===== Input head =====
        head_gn = choose_num_groups(base, self._gn_groups)
        self.head = nn.Sequential(
            nn.Conv2d(3, base, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(head_gn, base),
            nn.GELU(),
            *[MicroResBlock(base, num_groups=self._gn_groups) for _ in range(2)]
        )

        # ===== Encoder =====
        # Enc1: base -> base*2 (downsample)
        self.enc1 = nn.ModuleList([
            DepthwiseSeparable(base, base * 2, stride=2, num_groups=self._gn_groups),
            *[MicroResBlock(base * 2, num_groups=self._gn_groups) for _ in range(blocks[0])]
        ])

        # Enc2: base*2 -> base*4 (downsample)
        self.enc2 = nn.ModuleList([
            DepthwiseSeparable(base * 2, base * 4, stride=2, num_groups=self._gn_groups),
            *[MicroResBlock(base * 4, num_groups=self._gn_groups) for _ in range(blocks[1])]
        ])

        # Enc3: base*4 -> base*8 (downsample)
        self.enc3 = nn.ModuleList([
            DepthwiseSeparable(base * 4, base * 8, stride=2, num_groups=self._gn_groups),
            *[MicroResBlock(base * 8, num_groups=self._gn_groups) for _ in range(blocks[2])]
        ])

        # ===== Decoder Path (UPGRADED: Upsample+Conv and GroupNorm) =====
        # Level 3: base*8 → base*4, H/4, fuse with enc2 skip
        self.dec3_up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            DepthwiseSeparable(base*8, base*4)
        )
        self.dec3_fusion = MultiScaleFusion(base*4)
        self.dec3_blocks = nn.Sequential(
            *[MicroResBlock(base*4) for _ in range(blocks[1])]
        )
        
        # Level 2: base*4 → base*2, H/2, fuse with enc1 skip
        self.dec2_up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            DepthwiseSeparable(base*4, base*2)
        )
        self.dec2_fusion = MultiScaleFusion(base*2)
        self.dec2_blocks = nn.Sequential(
            *[MicroResBlock(base*2) for _ in range(blocks[0])]
        )
        
        # Level 1: base*2 → base, H, fuse with head skip
        self.dec1_up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            DepthwiseSeparable(base*2, base)
        )
        self.dec1_fusion = MultiScaleFusion(base)
        self.dec1_blocks = nn.Sequential(
            *[MicroResBlock(base) for _ in range(2)]
        )

        # ===== Tail: predict 3-channel residual =====
        self.tail = nn.Sequential(
            MicroResBlock(base, num_groups=self._gn_groups),
            nn.Conv2d(base, 3, kernel_size=3, padding=1)
        )

        # initialize weights
        self._init_weights()

        # Logging sizes
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logger.info(f"✓ Model initialized")
        logger.info(f"  Total: {total_params:,} ({total_params/1e6:.2f}M) params")
        logger.info(f"  Trainable: {trainable_params:,} params")
        logger.info(f"  Size (FP32): ~{total_params * 4 / 1e6:.1f} MB")
        logger.info("=" * 60)

    def _init_weights(self):
        """
        Initialize weights for stable training (GroupNorm included).
        - Convs: Kaiming normal
        - GroupNorm: weight=1, bias=0
        - Final conv in tail: zero-init (so network starts as identity)
        """
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                if hasattr(m, "weight") and m.weight is not None:
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if hasattr(m, "bias") and m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.GroupNorm):
                if hasattr(m, "weight") and m.weight is not None:
                    nn.init.ones_(m.weight)
                if hasattr(m, "bias") and m.bias is not None:
                    nn.init.zeros_(m.bias)

        # Zero-init final layer of tail for residual learning (if exists)
        final_conv = None
        if isinstance(self.tail, nn.Sequential) and len(self.tail) > 0:
            # expect last module to be Conv2d
            if isinstance(self.tail[-1], nn.Conv2d):
                final_conv = self.tail[-1]
        if isinstance(final_conv, nn.Conv2d):
            if hasattr(final_conv, "weight") and final_conv.weight is not None:
                nn.init.zeros_(final_conv.weight)
            if hasattr(final_conv, "bias") and final_conv.bias is not None:
                nn.init.zeros_(final_conv.bias)

        # if hasattr(self, "res_scale"):
        #     with torch.no_grad():
        #         self.res_scale.clamp_(0.0, 1.0)

        logger.info("✓ Weights initialized (tail final conv zeroed for residual start)")

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
        skip0 = self.head(x)  # base channels, H, W

        enc1 = skip0
        for layer in self.enc1:
            enc1 = layer(enc1)
        skip1 = enc1  # base*2, H/2, W/2

        enc2 = skip1
        for layer in self.enc2:
            enc2 = layer(enc2)
        skip2 = enc2  # base*4, H/4, W/4

        enc3 = skip2
        for layer in self.enc3:
            enc3 = layer(enc3)  # base*8, H/8, W/8

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

        # Predict residual and add scaled residual to input
        residual = self.tail(dec1) * torch.clamp(self.res_scale, 0.0, 1.0)
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
# SECTION 3: Factory
# ==============================================================================

def create_av1_nano_unet_restorer(
    size: str = 'small',
    crf_min: int = 23,
    crf_max: int = 63,
    norm_range: Tuple[float, float] = (-1, 1),
    res_scale: float = 0.1,
    gn_groups: int = 16,
) -> AV1NanoUnetRestorer:
    logger.info(f"Creating AV1NanoUnetRestorer (size={size}, CRF={crf_min}-{crf_max})")
    return AV1NanoUnetRestorer(size=size, crf_min=crf_min, crf_max=crf_max,
                               norm_range=norm_range, res_scale=res_scale, gn_groups=gn_groups)


# ==============================================================================
# SECTION 4: Quick test harness (run as script)
# ==============================================================================

if __name__ == "__main__":
    import time
    logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')

    print("\n" + "=" * 70)
    print("AV1 Nano U-Net Restorer - Model Testing (GroupNorm + ECA)")
    print("=" * 70 + "\n")

    for size in ['nano', 'tiny', 'small', 'base', 'large', 'huge']:
        print(f"\n{'=' * 70}")
        print(f"Testing: {size.upper()} unet variant")
        print('=' * 70)

        model = create_av1_nano_unet_restorer(size=size, crf_min=23, crf_max=33, norm_range=(-1, 1))
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

        # CPU speed benchmark
        print(f"\nSpeed Benchmark (1080p, 5 iterations, CPU):")
        x_1080p = torch.randn(1, 3, 1080, 1920)
        with torch.no_grad():
            _ = model.inference(x_1080p)  # warmup
            times = []
            for _ in range(5):
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
