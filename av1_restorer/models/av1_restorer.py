# av1_restorer/models/av1_restorer.py
"""
AV1 U-Net Restorer (Unified SOTA Factory)

Unified factory for creating state-of-the-art AV1 conditional restorers.
Automatically selects the optimal architecture based on the requested model size:

- 'nano', 'lite', 'tiny'  : EfficientRestorer (SOTA efficiency, <10M params)
- 'small', 'base', 'big'  : BalancedRestorer (SOTA quality-efficiency balance)
- 'large', 'huge', 'pro'  : QualityRestorer (SOTA maximum quality, >30M params)

Author: Soham Mukherjee
Version: 3.1
License: MIT
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
import logging

# Import core building blocks
from .blocks import (
    DepthwiseSeparable,
    EfficientResBlock,
    SimpleSelfAttention,
    choose_num_groups,
    ConditioningEmbedder,
    FiLMLayer,
    WaveletRestorationBlock,
    SwinBottleneck,
    ProgressiveFeatureFusion,
    DenseFeatureFusion
)

logger = logging.getLogger(__name__)


# ==============================================================================
# SECTION 1: EfficientRestorer Architecture
# FiLM + EfficientResBlocks + SimpleSelfAttention Bottleneck
# ==============================================================================

class AV1_EfficientRestorer(nn.Module):
    """
    EfficientRestorer:
    - Lightweight encoder-decoder with EfficientResBlocks.
    - Bottleneck uses SimpleSelfAttention + FiLM conditioning.
    - Tail: Upsample + residual EfficientResBlocks.
    
    Best for: 'nano', 'tiny', 'lite' variants (<10M params)
    """
    def __init__(self, config: dict):
        super().__init__()
        ch, blocks = config['channels'], config['blocks']
        crf_range, preset_range = config['crf_range'], config['preset_range']
        norm_range = tuple(config.get('norm_range', (0, 1)))
        self.clamp_min, self.clamp_max = norm_range
        self.cond_dim = 128 if preset_range[0] == preset_range[1] else 192 

        logger.info("  > Instantiating EfficientRestorer (SOTA Efficiency)")
        
        # Conditioning network
        self.conditioning_embedder = ConditioningEmbedder(crf_range, preset_range)
        
        # Input head
        self.head = nn.Sequential(
            nn.Conv2d(3, ch[0], 3, 1, 1, padding_mode='reflect', bias=False),
            nn.GroupNorm(choose_num_groups(ch[0]), ch[0]),
            nn.GELU(),
            *[EfficientResBlock(ch[0]) for _ in range(blocks[0])]
        )
        
        # Encoder levels
        self.encoder1 = self._build_encoder_level(ch[0], ch[1], blocks[1], self.cond_dim)
        self.encoder2 = self._build_encoder_level(ch[1], ch[2], blocks[2], self.cond_dim)
        self.encoder3 = self._build_encoder_level(ch[2], ch[3], blocks[3], self.cond_dim)
        
        # Bottleneck with simple self-attention
        self.bottleneck_down = nn.Conv2d(ch[3], ch[4], 3, 2, 1, padding_mode='reflect', bias=False)
        self.bottleneck_gn = nn.GroupNorm(choose_num_groups(ch[4]), ch[4])
        self.bottleneck_pre_attn = nn.Sequential(*[EfficientResBlock(ch[4]) for _ in range(blocks[4] // 2)])
        self.bottleneck_attn = SimpleSelfAttention(ch[4])
        self.bottleneck_film = FiLMLayer(self.cond_dim, ch[4])
        self.bottleneck_post_attn = nn.Sequential(*[EfficientResBlock(ch[4]) for _ in range(blocks[4] - (blocks[4] // 2))])
        
        # Decoder levels
        self.decoder3 = self._build_decoder_level(ch[4], ch[3], blocks[3])
        self.decoder2 = self._build_decoder_level(ch[3], ch[2], blocks[2])
        self.decoder1 = self._build_decoder_level(ch[2], ch[1], blocks[1])
        
        # Output tail (upsample + residual)
        self.tail_upsample_op = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.tail_upsample_conv = DepthwiseSeparable(ch[1], ch[0], stride=1)
        self.tail_gn = nn.GroupNorm(choose_num_groups(ch[0]), ch[0])
        self.tail_fusion = nn.Conv2d(ch[0] * 2, ch[0], 1, bias=False)
        self.tail_fusion_gn_act = nn.Sequential(nn.GroupNorm(choose_num_groups(ch[0]), ch[0]), nn.GELU())
        self.tail_body = nn.Sequential(*[EfficientResBlock(ch[0]) for _ in range(blocks[0])])
        self.tail_pred = nn.Conv2d(ch[0], 3, 3, 1, 1, padding_mode='reflect')
        
        self._init_weights()

    # --- Helper methods for building encoder/decoder ---
    def _build_encoder_level(self, in_ch, out_ch, num_blocks, cond_dim):
        """Build a single encoder stage with downsampling, residual blocks, and FiLM conditioning."""
        return nn.ModuleDict({
            'downsample': nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, 2, 1, padding_mode='reflect', bias=False),
                nn.GroupNorm(choose_num_groups(out_ch), out_ch),
                nn.GELU()
            ),
            'body': nn.Sequential(*[EfficientResBlock(out_ch) for _ in range(num_blocks)]),
            'film': FiLMLayer(cond_dim, out_ch)
        })

    def _build_decoder_level(self, in_ch, out_ch, num_blocks):
        """Build a single decoder stage with upsampling, residual blocks, and skip fusion."""
        return nn.ModuleDict({
            'upsample_op': nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            'upsample_conv': DepthwiseSeparable(in_ch, out_ch, stride=1),
            'upsample_gn': nn.GroupNorm(choose_num_groups(out_ch), out_ch),
            'fusion': nn.Sequential(
                nn.Conv2d(out_ch * 2, out_ch, 1, bias=False),
                nn.GroupNorm(choose_num_groups(out_ch), out_ch),
                nn.GELU()
            ),
            'body': nn.Sequential(*[EfficientResBlock(out_ch) for _ in range(num_blocks)])
        })

    def _init_weights(self):
        """Initialize weights: Conv2d with Kaiming, Linear with Xavier, zero tail_pred for residual learning."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.GroupNorm, nn.LayerNorm)):
                if m.weight is not None:
                    nn.init.ones_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.5)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        nn.init.zeros_(self.tail_pred.weight)
        if self.tail_pred.bias is not None:
            nn.init.zeros_(self.tail_pred.bias)
        logger.info("✓ Weights initialized (tail_pred zeroed for residual learning)")

    # --- Forward pass ---
    def forward(self, lq_image: torch.Tensor, crf: torch.Tensor, preset: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass for EfficientRestorer.
        Args:
            lq_image : Input low-quality AV1 frame
            crf      : CRF tensor
            preset   : Optional preset tensor
        Returns:
            Restored image (same shape as input)
        """
        cond = self.conditioning_embedder(crf, preset)
        skip0 = self.head(lq_image)
        
        e1 = self.encoder1['downsample'](skip0)
        e1 = self.encoder1['body'](e1)
        e1 = self.encoder1['film'](e1, cond)
        skip1 = e1

        e2 = self.encoder2['downsample'](skip1)
        e2 = self.encoder2['body'](e2)
        e2 = self.encoder2['film'](e2, cond)
        skip2 = e2
        
        e3 = self.encoder3['downsample'](skip2)
        e3 = self.encoder3['body'](e3)
        e3 = self.encoder3['film'](e3, cond)
        skip3 = e3
        
        b = self.bottleneck_down(skip3)
        b = F.gelu(self.bottleneck_gn(b))
        b = torch.clamp(b, -10.0, 10.0)
        b = self.bottleneck_pre_attn(b)
        
        # run SimpleSelfAttention in float32 for stability 
        with torch.amp.autocast(device_type=str(lq_image.device.type), enabled=False):
            b = self.bottleneck_attn(b.float())
        
        b = self.bottleneck_film(b, cond)
        b = torch.clamp(b, -10.0, 10.0)
        b = self.bottleneck_post_attn(b)
        
        d3 = self.decoder3['upsample_op'](b)
        d3 = self.decoder3['upsample_conv'](d3)
        d3 = F.gelu(self.decoder3['upsample_gn'](d3))
        d3 = torch.cat([d3, skip3], dim=1)
        d3 = self.decoder3['fusion'](d3)
        d3 = self.decoder3['body'](d3)
        
        d2 = self.decoder2['upsample_op'](d3)
        d2 = self.decoder2['upsample_conv'](d2)
        d2 = F.gelu(self.decoder2['upsample_gn'](d2))
        d2 = torch.cat([d2, skip2], dim=1)
        d2 = self.decoder2['fusion'](d2)
        d2 = self.decoder2['body'](d2)
        
        d1 = self.decoder1['upsample_op'](d2)
        d1 = self.decoder1['upsample_conv'](d1)
        d1 = F.gelu(self.decoder1['upsample_gn'](d1))
        d1 = torch.cat([d1, skip1], dim=1); d1 = self.decoder1['fusion'](d1); d1 = self.decoder1['body'](d1)
        
        t = self.tail_upsample_op(d1)
        t = self.tail_upsample_conv(t)
        t = F.gelu(self.tail_gn(t))
        t = torch.cat([t, skip0], dim=1)
        t = self.tail_fusion(t)
        t = self.tail_fusion_gn_act(t)
        # run Wavelet block in float32 for stability
        if isinstance(self.tail_body, nn.Sequential) and any(isinstance(m, WaveletRestorationBlock) for m in self.tail_body):
            with torch.amp.autocast(device_type=str(lq_image.device.type), enabled=False):
                t = self.tail_body(t.float())
        else:
            t = self.tail_body(t)

        residual = self.tail_pred(t)
        
        return torch.clamp(lq_image + residual, self.clamp_min, self.clamp_max)

    @torch.no_grad()
    def inference(self, *args, **kwargs):
        """Tiled and optionally TTA-enhanced inference."""
        return _inference_tiled(self, *args, **kwargs)

# ==============================================================================
# SECTION 2: BalancedRestorer Architecture
# FiLM + EfficientResBlocks + SwinBottleneck + WaveletRestorationBlock
# ==============================================================================

class AV1_BalancedRestorer(nn.Module):
    """
    BalancedRestorer:
    - Encoder/Decoder: EfficientResBlocks (lightweight residual blocks)
    - Bottleneck: SwinBottleneck + FiLM conditioning
    - Tail: Fusion + WaveletRestorationBlock
    
    Best for: 'small', 'base', 'big' variants (8M - 30M params)
    """
    def __init__(self, config: dict):
        super().__init__()
        ch, blocks = config['channels'], config['blocks']
        crf_range, preset_range = config['crf_range'], config['preset_range']
        norm_range = tuple(config.get('norm_range', (0, 1)))
        self.clamp_min, self.clamp_max = norm_range
        self.cond_dim = 128 if preset_range[0] == preset_range[1] else 192 
            
        logger.info("  > Instantiating BalancedRestorer (SOTA Balanced Quality-Efficiency)")

        # --- Conditioning network ---
        self.conditioning_embedder = ConditioningEmbedder(crf_range, preset_range)
        
        # --- Input head (V1) ---
        self.head = nn.Sequential(
            nn.Conv2d(3, ch[0], 3, 1, 1, padding_mode='reflect', bias=False),
            nn.GroupNorm(choose_num_groups(ch[0]), ch[0]), nn.GELU(),
            *[EfficientResBlock(ch[0]) for _ in range(blocks[0])]
        )
        
        # --- Encoder (V1) ---
        self.encoder1 = self._build_encoder_level(ch[0], ch[1], blocks[1], self.cond_dim)
        self.encoder2 = self._build_encoder_level(ch[1], ch[2], blocks[2], self.cond_dim)
        self.encoder3 = self._build_encoder_level(ch[2], ch[3], blocks[3], self.cond_dim)
        
        # --- Bottleneck (V3) ---
        self.bottleneck_down = nn.Conv2d(ch[3], ch[4], 3, 2, 1, padding_mode='reflect', bias=False)
        self.bottleneck_gn = nn.GroupNorm(choose_num_groups(ch[4]), ch[4])
        self.bottleneck_pre_attn = nn.Sequential(*[EfficientResBlock(ch[4]) for _ in range(blocks[4] // 2)])
        self.bottleneck_attn = SwinBottleneck(channels=ch[4], depth=4, num_heads=8, window_size=8)
        self.bottleneck_film = FiLMLayer(self.cond_dim, ch[4])
        self.bottleneck_post_attn = nn.Sequential(*[EfficientResBlock(ch[4]) for _ in range(blocks[4] - (blocks[4] // 2))])
        
        # --- Decoder (V1) ---
        self.decoder3 = self._build_decoder_level(ch[4], ch[3], blocks[3])
        self.decoder2 = self._build_decoder_level(ch[3], ch[2], blocks[2])
        self.decoder1 = self._build_decoder_level(ch[2], ch[1], blocks[1])
        
        # --- Output Tail (V1 fusion + V3 Wavelet) ---
        self.tail_upsample_op = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.tail_upsample_conv = DepthwiseSeparable(ch[1], ch[0], stride=1)
        self.tail_gn = nn.GroupNorm(choose_num_groups(ch[0]), ch[0])
        self.tail_fusion = nn.Conv2d(ch[0] * 2, ch[0], 1, bias=False)
        self.tail_fusion_gn_act = nn.Sequential(nn.GroupNorm(choose_num_groups(ch[0]), ch[0]), nn.GELU())
        self.tail_body = WaveletRestorationBlock(ch[0])
        self.tail_pred = nn.Conv2d(ch[0], 3, 3, 1, 1, padding_mode='reflect')
        
        self._init_weights()
    
    # Re-use helpers from EfficientRestorer
    _build_encoder_level = AV1_EfficientRestorer._build_encoder_level
    _build_decoder_level = AV1_EfficientRestorer._build_decoder_level
    _init_weights = AV1_EfficientRestorer._init_weights
    
    # Re-use forward pass from EfficientRestorer
    forward = AV1_EfficientRestorer.forward

    @torch.no_grad()
    def inference(self, *args, **kwargs):
        """Tiled and optional TTA-enhanced inference."""
        return _inference_tiled(self, *args, **kwargs)


# ==============================================================================
# SECTION 3: QualityRestorer Architecture
# EfficientResBlocks + WaveletRestorationBlock, SwinBottleneck + FiLM, ProgressiveFeatureFusion
# ==============================================================================

class AV1_QualityRestorer(nn.Module):
    """
    QualityRestorer:
    - Encoder/Decoder: EfficientResBlocks + WaveletRestorationBlock
    - Bottleneck: SwinBottleneck + FiLM conditioning
    - Decoder: Dense Progressive Feature Fusion + Wavelet blocks
    - Tail: EfficientResBlocks + WaveletRestorationBlock
    
    Best for: 'large', 'huge', 'pro' (>30M params)
    """
    def __init__(self, config: dict):
        super().__init__()
        
        ch = config['channels']
        blocks = config['blocks']
        crf_range = config.get('crf_range', (23, 63))
        preset_range = config.get('preset_range', (0, 8))
        norm_range = tuple(config.get('norm_range', (-1, 1)))
        
        self.clamp_min, self.clamp_max = norm_range
        self.cond_dim = 128 if preset_range[0] == preset_range[1] else 192
        
        logger.info("  > Instantiating QualityRestorer (SOTA Maximum Quality)")

        # --- Conditioning network ---
        self.conditioning_embedder = ConditioningEmbedder(crf_range, preset_range)
        
        # --- Input Head ---
        self.head = nn.Sequential(
            nn.Conv2d(3, ch[0], 3, 1, 1, padding_mode='reflect', bias=False),
            nn.GroupNorm(choose_num_groups(ch[0]), ch[0]), nn.GELU(),
            *[EfficientResBlock(ch[0]) for _ in range(blocks[0])]
        )
        
        # --- Encoder levels (V3) ---
        self.encoder_levels = nn.ModuleList([
            self._build_encoder_level(ch[i], ch[i+1], blocks[i+1], self.cond_dim)
            for i in range(4)
        ])
        
        # --- Bottleneck (Swin + FiLM) ---
        self.bottleneck = SwinBottleneck(channels=ch[4], depth=6, num_heads=8, window_size=8)
        self.bottleneck_film = FiLMLayer(self.cond_dim, ch[4])
        
        # --- Decoder levels (V3) ---
        self.decoder_levels = nn.ModuleList([
            self._build_decoder_level(ch[i+1], ch[i], blocks[i])
            for i in range(3, -1, -1)
        ])
        
        # --- Progressive Fusion across decoder levels ---
        self.progressive_fusion = ProgressiveFeatureFusion([ch[3], ch[2], ch[1], ch[0]])
        
        # --- Output Tail (V3) ---
        self.tail_body = nn.Sequential(
            *[EfficientResBlock(ch[0]) for _ in range(blocks[0])],
            WaveletRestorationBlock(ch[0])
        )
        self.tail_pred = nn.Conv2d(ch[0], 3, 3, 1, 1, padding_mode='reflect')
        
        self._init_weights()

    def _build_encoder_level(self, in_ch, out_ch, num_blocks, cond_dim):
        """V3 Encoder level with residuals, FiLM conditioning, and Wavelet blocks."""
        return nn.ModuleDict({
            'downsample': nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, 2, 1, padding_mode='reflect', bias=False),
                nn.GroupNorm(choose_num_groups(out_ch), out_ch), nn.GELU()
            ),
            'body': nn.Sequential(*[EfficientResBlock(out_ch) for _ in range(num_blocks)]),
            'film': FiLMLayer(cond_dim, out_ch),
            'wavelet': WaveletRestorationBlock(out_ch)
        })

    def _build_decoder_level(self, in_ch, out_ch, num_blocks):
        """V3 Decoder level with upsample, residual body, and Wavelet."""
        return nn.ModuleDict({
            'upsample': nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                DepthwiseSeparable(in_ch, out_ch, stride=1),
                nn.GroupNorm(choose_num_groups(out_ch), out_ch), nn.GELU()
            ),
            'body': nn.Sequential(*[EfficientResBlock(out_ch) for _ in range(num_blocks)]),
            'wavelet': WaveletRestorationBlock(out_ch)
        })

    def _init_weights(self):
        """Initialize weights using EfficientRestorer V1 scheme."""
        AV1_EfficientRestorer._init_weights(self)

    def forward(self, lq_image: torch.Tensor, crf: torch.Tensor, preset: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass for QualityRestorer (V3) with Dense Progressive Fusion."""
        # 1. Generate conditioning
        cond = self.conditioning_embedder(crf, preset)
        
        # 2. Head
        skip0 = self.head(lq_image)
        
        # 3. Encoder
        x = skip0
        skips = [skip0]
        for encoder in self.encoder_levels:
            x = encoder['downsample'](x)
            x = encoder['body'](x)
            x = encoder['film'](x, cond)
            x = torch.clamp(x, -10.0, 10.0)
            with torch.amp.autocast(device_type=str(lq_image.device.type), enabled=False):
                x = encoder['wavelet'](x.float())
            skips.append(x) # skips = [s0, s1, s2, s3, s4]
            
        # 4. Bottleneck
        x = skips.pop() # x = s4 (ch[4])
        # run SwinBottleneck in float32 for stability
        with torch.amp.autocast(device_type=str(lq_image.device.type), enabled=False):
            x = self.bottleneck(x.float())

        x = self.bottleneck_film(x, cond)
        x = torch.clamp(x, -10.0, 10.0) # Stability
        
        # 5. Decoder with Progressive Fusion
        # skips = [s0, s1, s2, s3]
        x_pff_prev = None # Stores output of previous (deeper) level
        
        for i, decoder in enumerate(self.decoder_levels):
            # i = 0: 4->3. i = 1: 3->2. i = 2: 2->1. i = 3: 1->0.
            
            x_up = decoder['upsample'](x)
            x_skip = skips[-(i+1)] # s3, then s2, then s1, then s0
            
            # Fuse at this level using the dedicated DFF block
            x_fused = self.progressive_fusion.fusions[i](x_skip, x_up) # V3 Feature
            
            # Add lateral (bottom-up) connection from previous (deeper) level
            if x_pff_prev is not None:
                bottom_up = F.interpolate(
                    x_pff_prev, size=x_fused.shape[-2:], mode='bilinear', align_corners=False
                )
                bottom_up = self.progressive_fusion.lateral[i-1](bottom_up) # V3 Feature
                x_fused = x_fused + bottom_up
                
            # Run the main body of this decoder level
            x = decoder['body'](x_fused)
            with torch.amp.autocast(device_type=str(lq_image.device.type), enabled=False):
                x = decoder['wavelet'](x.float()) # run Wavelet block in float32 for stability
                
            x_pff_prev = x # Store this output for the next iteration
            
        # 6. Tail
        # x is now the final, full-res output from the decoder loop (ch[0])
        with torch.amp.autocast(device_type=str(lq_image.device.type), enabled=False):
            x = self.tail_body(x.float()) # run Wavelet block in float32 for stability
        residual = self.tail_pred(x)
        
        # 7. Residual Connection
        restored = lq_image + residual
        return torch.clamp(restored, self.clamp_min, self.clamp_max)

    @torch.no_grad()
    def inference(self, *args, **kwargs):
        return _inference_tiled(self, *args, **kwargs)


# ==============================================================================
# SECTION 4: Shared Inference Logic (Tiling, TTA)
# ==============================================================================

def _inference_tiled(
    model: nn.Module,
    lq_image: torch.Tensor,
    crf: torch.Tensor,
    preset: Optional[torch.Tensor] = None,
    tile_size: int = 512,
    tile_overlap: int = 64,
    center_crop: bool = False,
    use_tta: bool = False
) -> torch.Tensor:
    """
    Restoration inference with robust tiling, padding, and optional test-time augmentation (TTA).
    
    Handles three cases:
    1. Center Crop: Fast preview of the center tile only.
    2. Small Image: Process the entire image at once (with padding if needed).
    3. Large Image: Split into overlapping tiles and blend results for seamless output.

    Args:
        model (nn.Module): AV1 restoration model.
        lq_image (Tensor): Low-quality input image [B, C, H, W].
        crf (Tensor): CRF conditioning tensor.
        preset (Optional[Tensor]): Optional preset conditioning tensor.
        tile_size (int): Size of the tile for tiling mode.
        tile_overlap (int): Overlap in pixels between tiles.
        center_crop (bool): If True, only process center crop (fast preview).
        use_tta (bool): If True, apply 8-fold test-time augmentation.
    
    Returns:
        Tensor: Restored image of same shape as input.
    """
    model.eval()
    B, C, H, W = lq_image.shape
    min_div = 32  # Model requires 5 levels of 2x downsampling (2^5=32)

    # --- Internal helper: Single model forward pass (handles CRF-only or CRF+Preset mode) ---
    def _model_pass(tile, base_crf, base_preset):
        is_crf_only_mode = (model.cond_dim == 128)
        num_augs = tile.shape[0] // base_crf.shape[0]

        crf_batch = base_crf.repeat_interleave(num_augs, dim=0) if num_augs > 1 else base_crf
        preset_batch = base_preset.repeat_interleave(num_augs, dim=0) if base_preset is not None and num_augs > 1 else base_preset

        if is_crf_only_mode:
            return model(tile, crf_batch)
        else:
            if preset_batch is None:
                raise ValueError("Preset input missing in CRF+Preset mode inference.")
            return model(tile, crf_batch, preset_batch)

    # --- Internal helper: Forward pass with optional TTA ---
    def _forward_pass_tta(tile, crf, preset, use_tta):
        if not use_tta:
            return _model_pass(tile, crf, preset)

        augmented_tiles = _tta_forward(tile)
        restored_augmented = _model_pass(augmented_tiles, crf, preset)
        restored_tiles = _tta_inverse(restored_augmented)
        return torch.mean(restored_tiles, dim=0, keepdim=True)

    # --- Internal helper: Pad a tensor to multiples of min_div ---
    def _pad_tensor(tensor, pad_h, pad_w):
        h, w = tensor.shape[-2:]
        padding_mode = 'reflect'
        if (pad_h > h and h > 0) or (pad_w > w and w > 0):
            logger.warning(f"Padding ({pad_h}, {pad_w}) > tile size ({h}, {w}). Falling back to 'replicate'.")
            padding_mode = 'replicate'
        return F.pad(tensor, (0, pad_w, 0, pad_h), mode=padding_mode)

    # ===== MODE 1: CENTER CROP ONLY (Fast Preview) =====
    if center_crop:
        crop_h = min(tile_size, H)
        crop_w = min(tile_size, W)
        start_h = (H - crop_h) // 2
        start_w = (W - crop_w) // 2
        center_tile = lq_image[:, :, start_h:start_h + crop_h, start_w:start_w + crop_w]

        pad_h = (min_div - crop_h % min_div) % min_div
        pad_w = (min_div - crop_w % min_div) % min_div

        if pad_h > 0 or pad_w > 0:
            center_tile_padded = _pad_tensor(center_tile, pad_h, pad_w)
            output = _forward_pass_tta(center_tile_padded, crf, preset, use_tta)
            output = output[:, :, :crop_h, :crop_w]  # Remove padding
        else:
            output = _forward_pass_tta(center_tile, crf, preset, use_tta)
        return output

    # ===== MODE 2: SMALL IMAGE (No Tiling Needed) =====
    if H <= tile_size and W <= tile_size:
        pad_h = (min_div - H % min_div) % min_div
        pad_w = (min_div - W % min_div) % min_div

        if pad_h > 0 or pad_w > 0:
            lq_padded = _pad_tensor(lq_image, pad_h, pad_w)
            output = _forward_pass_tta(lq_padded, crf, preset, use_tta)
            return output[:, :, :H, :W]  # Remove padding
        else:
            return _forward_pass_tta(lq_image, crf, preset, use_tta)

    # ===== MODE 3: LARGE IMAGE (Tiling) =====
    logger.info(f"Tiled inference: {H}×{W} image → {tile_size}×{tile_size} tiles "
                f"(overlap: {tile_overlap}px, TTA: {use_tta})")

    output = torch.zeros_like(lq_image)
    weight = torch.zeros_like(lq_image)

    stride = tile_size - tile_overlap
    h_tiles = (H + stride - 1) // stride
    w_tiles = (W + stride - 1) // stride

    for i in range(h_tiles):
        for j in range(w_tiles):
            h_start = i * stride
            w_start = j * stride
            h_end = min(h_start + tile_size, H)
            w_end = min(w_start + tile_size, W)

            tile = lq_image[:, :, h_start:h_end, w_start:w_end]
            tile_h, tile_w = tile.shape[-2:]

            pad_h = (min_div - tile_h % min_div) % min_div
            pad_w = (min_div - tile_w % min_div) % min_div

            if pad_h > 0 or pad_w > 0:
                tile_padded = _pad_tensor(tile, pad_h, pad_w)
                restored_tile = _forward_pass_tta(tile_padded, crf, preset, use_tta)
                restored_tile = restored_tile[:, :, :tile_h, :tile_w]  # Remove padding
            else:
                restored_tile = _forward_pass_tta(tile, crf, preset, use_tta)

            blend = _create_blend_mask(
                tile_h, tile_w, tile_overlap,
                is_top=(i == 0), is_bottom=(h_end >= H),
                is_left=(j == 0), is_right=(w_end >= W),
                device=lq_image.device
            )

            output[:, :, h_start:h_end, w_start:w_end] += restored_tile * blend
            weight[:, :, h_start:h_end, w_start:w_end] += blend

    output = output / (weight + 1e-8)  # Normalize by accumulated weights
    return output


def _tta_forward(x: torch.Tensor) -> torch.Tensor:
    """Applies 8 augmentations for Test-Time Augmentation (4 rotations, 4 flips)."""
    x_rot90 = x.rot90(1, [-2, -1])
    x_rot180 = x.rot90(2, [-2, -1])
    x_rot270 = x.rot90(3, [-2, -1])
    x_flip = x.flip(-1)
    x_flip_rot90 = x_flip.rot90(1, [-2, -1])
    x_flip_rot180 = x_flip.rot90(2, [-2, -1])
    x_flip_rot270 = x_flip.rot90(3, [-2, -1])
    return torch.cat([x, x_rot90, x_rot180, x_rot270, x_flip, x_flip_rot90, x_flip_rot180, x_flip_rot270], dim=0)


def _tta_inverse(x: torch.Tensor) -> torch.Tensor:
    """Reverts the 8 augmentations applied during TTA."""
    tta_outs = x.chunk(8, dim=0)
    o0 = tta_outs[0]
    o1 = tta_outs[1].rot90(-1, [-2, -1])
    o2 = tta_outs[2].rot90(-2, [-2, -1])
    o3 = tta_outs[3].rot90(-3, [-2, -1])
    o4 = tta_outs[4].flip(-1)
    o5 = tta_outs[5].rot90(-1, [-2, -1]).flip(-1)
    o6 = tta_outs[6].rot90(-2, [-2, -1]).flip(-1)
    o7 = tta_outs[7].rot90(-3, [-2, -1]).flip(-1)
    return torch.stack([o0, o1, o2, o3, o4, o5, o6, o7], dim=0)


def _create_blend_mask(tile_h, tile_w, overlap, is_top, is_bottom, is_left, is_right, device):
    """
    Creates a linear blending mask for seamless tile stitching.

    Overlap is gradually blended using ramps up to `min(overlap//2, 32)` pixels.
    """
    blend = torch.ones(1, 1, tile_h, tile_w, device=device)
    fade = min(overlap // 2, 32)

    if not is_top:
        fade_h = min(fade, tile_h)
        ramp = torch.linspace(0, 1, fade_h, device=device).view(-1, 1)
        blend[:, :, :fade_h, :] *= ramp
    if not is_bottom:
        fade_h = min(fade, tile_h)
        ramp = torch.linspace(1, 0, fade_h, device=device).view(-1, 1)
        blend[:, :, -fade_h:, :] *= ramp
    if not is_left:
        fade_w = min(fade, tile_w)
        ramp = torch.linspace(0, 1, fade_w, device=device).view(1, -1)
        blend[:, :, :, :fade_w] *= ramp
    if not is_right:
        fade_w = min(fade, tile_w)
        ramp = torch.linspace(1, 0, fade_w, device=device).view(1, -1)
        blend[:, :, :, -fade_w:] *= ramp

    return blend


# ==============================================================================
# SECTION 5: Unified Restorer Factory
# ==============================================================================

def create_av1_restorer(
    size: str = 'large',
    crf_range: Tuple[int, int] = (23, 63),
    preset_range: Tuple[int, int] = (0, 8),
    norm_range: Tuple[float, float] = (-1, 1)
) -> nn.Module:
    """
    Factory for creating AV1 Conditional U-Net optimized for the requested size.

    Selects EfficientRestorer, BalancedRestorer, or QualityRestorer depending on size.
    Supports CRF-only or CRF+Preset conditioning.

    Returns:
        nn.Module: Initialized model ready for inference/training.
    """
    # --- EfficientRestorer configs (~2M-8M params) ---
    configs_v1 = {
        'nano': {'channels': [16, 32, 64, 128, 192], 'blocks': [1, 2, 3, 3, 4]},  # ~2M
        'lite': {'channels': [16, 32, 64, 128, 256], 'blocks': [2, 2, 3, 3, 4]},  # ~4M
        'tiny': {'channels': [24, 48, 96, 192, 288], 'blocks': [2, 2, 3, 3, 6]},  # ~8M recommended
    }

    # --- BalancedRestorer configs (~12M-24M params) ---
    configs_v2 = {
        'small': {'channels': [24, 48, 96, 192, 288], 'blocks': [3, 3, 3, 3, 6]},  # ~12M
        'base':  {'channels': [30, 60, 120, 240, 320], 'blocks': [3, 3, 4, 4, 6]}, # ~16M recommended
        'big':   {'channels': [32, 64, 128, 256, 384], 'blocks': [3, 3, 4, 4, 8]}, # ~24M
    }

    # --- QualityRestorer configs (~32M-96M params) ---
    configs_v3 = {
        'large': {'channels': [24, 48, 96, 192, 288], 'blocks': [3, 3, 4, 4, 8]},   # ~32M recommended
        'huge':  {'channels': [32, 64, 128, 256, 384], 'blocks': [3, 3, 4, 4, 10]}, # ~60M
        'pro':   {'channels': [32, 64, 128, 256, 512], 'blocks': [4, 6, 6, 8, 12]}, # ~96M
    }

    # --- Select architecture ---
    if size in configs_v1:
        cfg = configs_v1[size]
        ModelClass = AV1_EfficientRestorer
    elif size in configs_v2:
        cfg = configs_v2[size]
        ModelClass = AV1_BalancedRestorer
    elif size in configs_v3:
        cfg = configs_v3[size]
        ModelClass = AV1_QualityRestorer
    else:
        valid_sizes = ", ".join(list(configs_v1.keys()) + list(configs_v2.keys()) + list(configs_v3.keys()))
        raise ValueError(f"Unknown size '{size}'. Choose from: {valid_sizes}")

    # --- Final model configuration ---
    model_config = {
        'channels': cfg['channels'],
        'blocks': cfg['blocks'],
        'crf_range': crf_range,
        'preset_range': preset_range,
        'norm_range': norm_range
    }

    # --- Logging ---
    is_crf_only = (preset_range[0] == preset_range[1])
    cond_mode = "CRF-Only" if is_crf_only else "CRF+Preset"
    logger.info("=" * 70)
    logger.info(f"Creating AV1 Restorer - Size: {size.upper()} (Mode: {cond_mode})")
    logger.info(f"  CRF Range: {crf_range}")
    logger.info(f"  Image Norm Range: {norm_range}")

    model = ModelClass(model_config)
    actual_params = sum(p.numel() for p in model.parameters())
    logger.info(f"  ✓ Model Instantiated - Params: {actual_params:,} ({actual_params/1e6:.2f}M)")
    logger.info("=" * 70)

    return model


# ==============================================================================
# SECTION 6: Example Usage & Testing
# ==============================================================================

if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)

    model_sizes = ['nano', 'lite', 'tiny', 'small', 'base', 'big', 'large', 'huge', 'pro']

    print("\n" + "=" * 70)
    print("UNIFIED AV1 RESTORER FACTORY - PARAMETER CALIBRATION TEST")
    print("=" * 70 + "\n")

    print(f"{'Size':<10} | {'Target':<10} | {'Selected Architecture':<20} | {'Actual Params':<15}")
    print("-" * 60)

    for model_size in model_sizes:
        try:
            model = create_av1_restorer(size=model_size, preset_range=(4, 4))  # CRF-Only mode
            params_m = sum(p.numel() for p in model.parameters()) / 1e6
            target_map = {'nano':'~2M','lite':'~4M','tiny':'~8M','small':'~12M','base':'~16M','big':'~24M','large':'~32M','huge':'~50-64M','pro':'~100-128M'}
            target = target_map[model_size]

            print(f"{model_size:<10} | {target:<10} | {model.__class__.__name__:<20} | {params_m:<15.2f}M")

            # Simple forward pass test
            lq = torch.randn(1, 3, 256, 256)
            crf = torch.tensor([[48.0]])
            model.eval()
            with torch.no_grad():
                restored = model(lq, crf)
            assert restored.shape == lq.shape

        except Exception as e:
            print(f"\n✗ Test FAILED for size: '{model_size}'")
            print(f"  Error: {e}")
            import traceback
            traceback.print_exc()
