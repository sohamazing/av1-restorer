# av1_restorer/models/av1_restorer.py
"""
AV1 U-Net Restorer v3.0 - SOTA Architecture with All Improvements

Integrates SOTA components for maximum restoration quality:
1. Wavelet Transform Processing (frequency-aware restoration)
2. Swin Transformer Bottleneck (global receptive field)
3. Dense Feature Fusion (refined, attention-based skip connections)

Author: Soham Mukherjee
Version: 3.0 (SOTA Complete)
License: MIT
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, Optional, List
import logging

# Import all required blocks from our new unified blocks.py
from blocks import (
    DepthwiseSeparable, 
    EfficientResBlock, 
    choose_num_groups,
    ConditioningEmbedder,
    FiLMLayer,
    WaveletRestorationBlock,
    SwinBottleneck,
    ProgressiveFeatureFusion,
    DenseFeatureFusion
)

logger = logging.getLogger(__name__)


class AV1ConditionalUNetPro(nn.Module):
    """
    SOTA AV1 U-Net with Wavelet, Swin, and Dense Fusion improvements.
    
    Architecture Flow:
    ==================
    Input [B,3,H,W] + Conditioning
      ↓
    Head (Initial feature extraction)
      ↓
    ╔═══════════════════════════════════════╗
    ║  Encoder with Wavelet Processing      ║
    ║  Level 1: Wavelet + EfficientResBlock ║ ─┐
    ║  Level 2: Wavelet + EfficientResBlock ║ ─┼─ Skip Connections
    ║  Level 3: Wavelet + EfficientResBlock ║ ─┼─ (Dense Fusion)
    ║  Level 4: Wavelet + EfficientResBlock ║ ─┘ (ch[3] -> ch[4])
    ╚═══════════════════════════════════════╝
      ↓
    ╔═══════════════════════════════════════╗
    ║  Swin Transformer Bottleneck (ch[4])  ║
    ║  - Global receptive field             ║
    ║  - Window + Shifted window attention  ║
    ║  - FiLM conditioning                  ║
    ╚═══════════════════════════════════════╝
      ↓
    ╔═══════════════════════════════════════╗
    ║  Decoder with Progressive Fusion      ║
    ║  Level 3: PFF(s3, d4_up) + Body       ║ ← Skip 3 (ch[3])
    ║  Level 2: PFF(s2, d3_up) + Body       ║ ← Skip 2 (ch[2])
    ║  Level 1: PFF(s1, d2_up) + Body       ║ ← Skip 1 (ch[1])
    ║  Level 0: PFF(s0, d1_up) + Body       ║ ← Skip 0 (ch[0])
    ╚═══════════════════════════════════════╝
      ↓
    Tail (Final refinement)
      ↓
    Output = Input + Residual
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
        
        # ===== CONDITIONING =====
        self.conditioning_embedder = ConditioningEmbedder(crf_range, preset_range)
        
        # ===== INPUT HEAD =====
        self.head = nn.Sequential(
            nn.Conv2d(3, ch[0], 3, padding=1, padding_mode='reflect', bias=False),
            nn.GroupNorm(choose_num_groups(ch[0]), ch[0]),
            nn.GELU(),
            *[EfficientResBlock(ch[0]) for _ in range(blocks[0])]
        )
        
        # ===== ENCODER WITH WAVELET PROCESSING =====
        # 4 levels: (ch[0]->ch[1]), (ch[1]->ch[2]), (ch[2]->ch[3]), (ch[3]->ch[4])
        self.encoder_levels = nn.ModuleList([
            self._build_encoder_level(ch[i], ch[i+1], blocks[i+1], self.cond_dim, use_wavelet=True)
            for i in range(4)
        ])
        
        # ===== SWIN TRANSFORMER BOTTLENECK =====
        # Note: The last encoder level (i=3) already outputs at ch[4].
        # The bottleneck_down/norm from the V3 draft are not needed.
        self.bottleneck = SwinBottleneck(
            channels=ch[4],
            depth=6,  # 6 Swin blocks for deep global context
            num_heads=8,
            window_size=8
        )
        
        self.bottleneck_film = FiLMLayer(self.cond_dim, ch[4])
        
        # ===== DECODER WITH PROGRESSIVE FUSION =====
        # 4 levels: (ch[4]->ch[3]), (ch[3]->ch[2]), (ch[2]->ch[1]), (ch[1]->ch[0])
        self.decoder_levels = nn.ModuleList([
            self._build_decoder_level(ch[i+1], ch[i], blocks[i], use_wavelet=True)
            for i in range(3, -1, -1) # Build from deep to shallow: 4->3, 3->2, 2->1, 1->0
        ])
        
        # Progressive fusion for skip connections
        # Handles skips s3, s2, s1, s0 (channels ch[3], ch[2], ch[1], ch[0])
        self.progressive_fusion = ProgressiveFeatureFusion(
            [ch[3], ch[2], ch[1], ch[0]]
        )
        
        # ===== OUTPUT TAIL =====
        # Final refinement body and prediction layer.
        # The PFF module handles the final fusion with skip0.
        self.tail_body = nn.Sequential(
            *[EfficientResBlock(ch[0]) for _ in range(blocks[0])],
            WaveletRestorationBlock(ch[0])  # Final wavelet refinement
        )
        
        self.tail_pred = nn.Conv2d(ch[0], 3, 3, padding=1, padding_mode='reflect')
        
        self._init_weights()
        
        # Log architecture details
        total_params = sum(p.numel() for p in self.parameters())
        logger.info("=" * 70)
        logger.info("AV1ConditionalUNetPro - SOTA Architecture")
        logger.info("=" * 70)
        logger.info(f"  Channels: {ch}")
        logger.info(f"  Blocks: {blocks}")
        logger.info(f"  Parameters: {total_params:,} ({total_params/1e6:.2f}M)")
        logger.info("  Key Features:")
        logger.info("    ✓ Wavelet transform processing")
        logger.info("    ✓ Swin transformer bottleneck")
        logger.info("    ✓ Dense feature fusion")
        logger.info("    ✓ Progressive multi-scale learning")
        logger.info("=" * 70)
    
    def _build_encoder_level(
        self, 
        in_ch: int, 
        out_ch: int, 
        num_blocks: int, 
        cond_dim: int,
        use_wavelet: bool = True
    ) -> nn.ModuleDict:
        """Build encoder level with optional wavelet processing."""
        modules = nn.ModuleDict({
            'downsample': nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, stride=2, padding=1, padding_mode='reflect', bias=False),
                nn.GroupNorm(choose_num_groups(out_ch), out_ch),
                nn.GELU()
            ),
            'body': nn.Sequential(
                *[EfficientResBlock(out_ch) for _ in range(num_blocks)]
            ),
            'film': FiLMLayer(cond_dim, out_ch)
        })
        
        if use_wavelet:
            modules['wavelet'] = WaveletRestorationBlock(out_ch)
        
        return modules
    
    def _build_decoder_level(
        self,
        in_ch: int,
        out_ch: int,
        num_blocks: int,
        use_wavelet: bool = True
    ) -> nn.ModuleDict:
        """Build decoder level with optional wavelet processing."""
        modules = nn.ModuleDict({
            'upsample': nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                DepthwiseSeparable(in_ch, out_ch, stride=1),
                nn.GroupNorm(choose_num_groups(out_ch), out_ch),
                nn.GELU()
            ),
            'body': nn.Sequential(
                *[EfficientResBlock(out_ch) for _ in range(num_blocks)]
            )
        })
        
        if use_wavelet:
            modules['wavelet'] = WaveletRestorationBlock(out_ch)
        
        return modules
    
    def _init_weights(self):
        """Initialize weights with special handling for wavelet filters."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.GroupNorm, nn.LayerNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.5)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        
        # Zero-initialize final prediction layer
        nn.init.zeros_(self.tail_pred.weight)
        if self.tail_pred.bias is not None:
            nn.init.zeros_(self.tail_pred.bias)
    
    def forward(
        self,
        lq_image: torch.Tensor,
        crf: torch.Tensor,
        preset: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass with Wavelet, Swin, and Progressive Fusion.
        
        Args:
            lq_image: [B, 3, H, W] Low-quality input
            crf: [B, 1] CRF value
            preset: [B, 1] Optional preset value
        
        Returns:
            [B, 3, H, W] Restored image
        """
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
            if 'wavelet' in encoder:
                x = encoder['wavelet'](x)
            skips.append(x) # skips = [s0, s1, s2, s3, s4]
            
        # 4. Bottleneck
        x = skips.pop() # x = s4 (ch[4])
        x = torch.clamp(x, -10.0, 10.0) # Stability
        x = self.bottleneck(x)
        x = self.bottleneck_film(x, cond)
        x = torch.clamp(x, -10.0, 10.0) # Stability
        
        # 5. Decoder with Progressive Fusion
        # skips = [s0, s1, s2, s3]
        
        x_pff_prev = None # Stores output of previous (deeper) level for lateral connection
        
        for i, decoder in enumerate(self.decoder_levels):
            # i = 0: 4->3. i = 1: 3->2. i = 2: 2->1. i = 3: 1->0.
            
            x_up = decoder['upsample'](x)
            x_skip = skips[-(i+1)] # s3, then s2, then s1, then s0
            
            # Fuse at this level using the dedicated DFF block
            x_fused = self.progressive_fusion.fusions[i](x_skip, x_up)
            
            # Add lateral (bottom-up) connection from previous (deeper) level
            if x_pff_prev is not None:
                bottom_up = F.interpolate(
                    x_pff_prev, 
                    size=x_fused.shape[-2:], 
                    mode='bilinear', 
                    align_corners=False
                )
                # Use the correct lateral block (i-1)
                bottom_up = self.progressive_fusion.lateral[i-1](bottom_up)
                x_fused = x_fused + bottom_up
                
            # Run the main body of this decoder level
            x = decoder['body'](x_fused)
            if 'wavelet' in decoder:
                x = decoder['wavelet'](x)
                
            x_pff_prev = x # Store this output for the next iteration's lateral connection
            
        # 6. Tail
        # x is now the final, full-res output from the decoder loop (ch[0])
        x = self.tail_body(x)
        residual = self.tail_pred(x)
        
        # 7. Residual Connection
        restored = lq_image + residual
        return torch.clamp(restored, self.clamp_min, self.clamp_max)

    # ==========================================================================
    # --- SOTA INFERENCE & HELPER METHODS (Copied from V2)
    # ==========================================================================

    def _model_pass(self, tile: torch.Tensor, base_crf: torch.Tensor, base_preset: Optional[torch.Tensor]):
        """
        Internal helper for `inference`. Handles conditioning tensor batching.
        
        This function correctly expands the (B=1) CRF/Preset tensors to match
        the (B=8) batch size used during Test-Time Augmentation (TTA).
        """
        is_crf_only_mode = (self.cond_dim == 128)
        
        # Check if tile batch is larger than cond batch (i.e., TTA is active)
        num_augs = tile.shape[0] // base_crf.shape[0]
        
        if num_augs > 1:
            # TTA is active, repeat cond tensors to match the TTA batch size
            crf_batch = base_crf.repeat_interleave(num_augs, dim=0)
            preset_batch = base_preset.repeat_interleave(num_augs, dim=0) if base_preset is not None else None
        else:
            # No TTA, use cond tensors as-is (batch size is already correct)
            crf_batch = base_crf
            preset_batch = base_preset

        # Call the appropriate forward pass based on model's conditioning mode
        if is_crf_only_mode:
            # Call V3 forward pass
            return self.forward(tile, crf_batch)
        else:
            if preset_batch is None:
                 raise ValueError("Preset input missing in CRF+Preset mode inference.")
            # Call V3 forward pass
            return self.forward(tile, crf_batch, preset_batch)

    def _forward_pass_tta(self, tile: torch.Tensor, crf: torch.Tensor, preset: Optional[torch.Tensor], use_tta: bool) -> torch.Tensor:
        """
        Internal helper for `inference`. Wraps the model pass with optional TTA.
        
        If TTA is True, this runs the model 8 times (flips/rotations) and
        averages the results for a higher-quality, more stable output.
        """
        if not use_tta:
            return self._model_pass(tile, crf, preset)
        
        # 1. Augment: Create 8 versions of the tile
        augmented_tiles = self._tta_forward(tile)
        
        # 2. Predict: Run all 8 augmentations through the model in one batch
        restored_augmented = self._model_pass(augmented_tiles, crf, preset)
        
        # 3. Invert: Revert the 8 augmentations on the output
        restored_tiles = self._tta_inverse(restored_augmented)
        
        # 4. Average: Combine the 8 results
        return torch.mean(restored_tiles, dim=0, keepdim=True)

    def _pad_tensor(self, tensor: torch.Tensor, pad_h: int, pad_w: int) -> torch.Tensor:
        """
        Applies robust padding.
        
        Falls back to 'replicate' padding if the tile is smaller than the
        required padding, preventing a 'reflect' mode crash.
        """
        h, w = tensor.shape[-2:]
        padding_mode = 'reflect'
        
        # Check if padding is larger than the tile dimension
        if (pad_h > h and h > 0) or (pad_w > w and w > 0):
             logger.warning(f"Padding ({pad_h}, {pad_w}) > tile size ({h}, {w})." 
                           " Falling back to 'replicate' padding.")
             padding_mode = 'replicate'
             
        return F.pad(tensor, (0, pad_w, 0, pad_h), mode=padding_mode)

    @torch.no_grad()
    def inference(
        self,
        lq_image: torch.Tensor,
        crf: torch.Tensor,
        preset: Optional[torch.Tensor] = None,
        tile_size: int = 512,
        tile_overlap: int = 64,
        center_crop: bool = False,
        use_tta: bool = False  # TTA is off by default for speed
    ) -> torch.Tensor:
        """
        SOTA inference with robust tiling, padding, and Test-Time Augmentation (TTA).
        
        This is the main function called by inference scripts. It handles
        three cases:
        1. Center Crop: Fast preview of just the center tile.
        2. Small Image: Processes the whole image in one pass (with padding).
        3. Large Image: Splits the image into overlapping tiles and stitches
           the results seamlessly.
        """
        self.eval()
        B, C, H, W = lq_image.shape
        min_div = 32  # Model needs 5 levels of 2x downsampling (2^5 = 32)

        # ===== MODE 1: CENTER CROP ONLY (Fast Preview) =====
        if center_crop:
            crop_h = min(tile_size, H)
            crop_w = min(tile_size, W)
            start_h = (H - crop_h) // 2
            start_w = (W - crop_w) // 2
            center_tile = lq_image[:, :, start_h:start_h+crop_h, start_w:start_w+crop_w]
            
            pad_h = (min_div - crop_h % min_div) % min_div
            pad_w = (min_div - crop_w % min_div) % min_div
            
            if pad_h > 0 or pad_w > 0:
                center_tile_padded = self._pad_tensor(center_tile, pad_h, pad_w)
                output = self._forward_pass_tta(center_tile_padded, crf, preset, use_tta)
                output = output[:, :, :crop_h, :crop_w] # Un-pad
            else:
                output = self._forward_pass_tta(center_tile, crf, preset, use_tta)
            return output
        
        # ===== MODE 2: SMALL IMAGE (No Tiling Needed) =====
        if H <= tile_size and W <= tile_size:
            pad_h = (min_div - H % min_div) % min_div
            pad_w = (min_div - W % min_div) % min_div
            
            if pad_h > 0 or pad_w > 0:
                lq_padded = self._pad_tensor(lq_image, pad_h, pad_w)
                output = self._forward_pass_tta(lq_padded, crf, preset, use_tta)
                return output[:, :, :H, :W] # Un-pad
            else:
                return self._forward_pass_tta(lq_image, crf, preset, use_tta)
        
        # ===== MODE 3: LARGE IMAGE (Tiling) =====
        if not center_crop:
             logger.info(f"Tiled inference: {H}×{W} image → {tile_size}×{tile_size} tiles (overlap: {tile_overlap}px, TTA: {use_tta})")
        
        output = torch.zeros_like(lq_image)
        weight = torch.zeros_like(lq_image)
        
        stride = tile_size - tile_overlap
        h_tiles = (H + stride - 1) // stride
        w_tiles = (W + stride - 1) // stride
        
        for i in range(h_tiles):
            for j in range(w_tiles):
                # Calculate coordinates for the current tile
                h_start = i * stride
                w_start = j * stride
                h_end = min(h_start + tile_size, H)
                w_end = min(w_start + tile_size, W)
                
                # Extract the tile
                tile = lq_image[:, :, h_start:h_end, w_start:w_end]
                tile_h, tile_w = tile.shape[-2:]
                
                # Pad the tile if it's not divisible by min_div
                pad_h = (min_div - tile_h % min_div) % min_div
                pad_w = (min_div - tile_w % min_div) % min_div
                
                if pad_h > 0 or pad_w > 0:
                    tile_padded = self._pad_tensor(tile, pad_h, pad_w)
                    restored_tile = self._forward_pass_tta(tile_padded, crf, preset, use_tta)
                    restored_tile = restored_tile[:, :, :tile_h, :tile_w] # Un-pad
                else:
                    restored_tile = self._forward_pass_tta(tile, crf, preset, use_tta)
                
                # Create a blend mask for seamless stitching
                blend = self._create_blend_mask(
                    tile_h, tile_w, tile_overlap,
                    is_top=(i == 0), is_bottom=(h_end >= H),
                    is_left=(j == 0), is_right=(w_end >= W),
                    device=lq_image.device
                )
                
                # Add the blended tile to the output canvas
                output[:, :, h_start:h_end, w_start:w_end] += restored_tile * blend
                weight[:, :, h_start:h_end, w_start:w_end] += blend
        
        # Normalize the output by the blend weights
        output = output / (weight + 1e-8)
        return output

    def _tta_forward(self, x: torch.Tensor) -> torch.Tensor:
        """Helper: Applies 8 augmentations (4 rotations, 4 flips)."""
        x_rot90 = x.rot90(1, [-2, -1])
        x_rot180 = x.rot90(2, [-2, -1])
        x_rot270 = x.rot90(3, [-2, -1])
        
        x_flip = x.flip(-1)
        x_flip_rot90 = x_flip.rot90(1, [-2, -1])
        x_flip_rot180 = x_flip.rot90(2, [-2, -1])
        x_flip_rot270 = x_flip.rot90(3, [-2, -1])
        
        return torch.cat([
            x, x_rot90, x_rot180, x_rot270,
            x_flip, x_flip_rot90, x_flip_rot180, x_flip_rot270
        ], dim=0)

    def _tta_inverse(self, x: torch.Tensor) -> torch.Tensor:
        """Helper: Reverts the 8 augmentations."""
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

    def _create_blend_mask(
        self,
        tile_h: int,
        tile_w: int,
        overlap: int,
        is_top: bool,
        is_bottom: bool,
        is_left: bool,
        is_right: bool,
        device: torch.device
    ) -> torch.Tensor:
        """Creates a linear blend mask for seamless tile stitching."""
        
        blend = torch.ones(1, 1, tile_h, tile_w, device=device)
        fade = min(overlap // 2, 32) # Use a fade ramp up to 32px
        
        # Create 1D ramps
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
# SECTION 4: Unified Model Factory (V3)
# ==============================================================================

def create_av1_restorer_v3(
    size: str = 'large', # Defaulting to ~20M target
    crf_range: Tuple[int, int] = (23, 63),
    preset_range: Tuple[int, int] = (0, 8),
    norm_range: Tuple[float, float] = (-1, 1)
) -> AV1ConditionalUNetPro:
    """
    Factory for SOTA V3 AV1 Conditional U-Nets.

    Integrates Wavelet, Swin, and Dense Fusion components.
    Uses the same size definitions as the V2 models for easy comparison.

    Size Options & Target / Actual Parameters (Approx. V2):
    ----------------------------------------------------------
    nano:    Target ~2M  (Actual: ~2.3M)
    tiny:    Target ~4M  (Actual: ~5.0M)
    small:   Target ~8M  (Actual: ~9.2M)
    base:    Target ~12M (Actual: ~12.8M)
    large:   Target ~20M (Actual: ~19.7M) - RECOMMENDED
    huge:    Target ~32M (Actual: ~32.3M)
    pro:     Target ~48M (Actual: ~50.3M)
    
    (Note: V3 models will have a ~15-25% parameter increase over these)

    Args:
        size (str): Model variant ('nano' to 'pro'). Defaults to 'large'.
        crf_range (Tuple[int, int]): CRF normalization range (min, max).
        preset_range (Tuple[int, int]): Preset range. Use identical min/max
                                         (e.g., [4, 4]) for CRF-only mode.
        norm_range (Tuple[float, float]): Image normalization range (e.g., [-1, 1]).

    Returns:
        AV1ConditionalUNetPro: Initialized V3 model instance.
    """
    # Use the same battle-tested configs from V2
    configs = {
        # Target: ~2M params (Actual: ~2.3M)
        'nano':   {'channels': [16, 32, 64, 128, 192], 'blocks': [2, 2, 2, 3, 3]},
        # Target: ~4M params (Actual: ~5.0M)
        'tiny':   {'channels': [20, 40, 80, 160, 240], 'blocks': [2, 2, 3, 4, 5]},
        # Target: ~8M params (Actual: ~9.2M)
        'small':  {'channels': [24, 48, 96, 192, 288], 'blocks': [3, 3, 4, 5, 6]},
        # Target: ~12M params (Actual: ~12.8M)
        'base':   {'channels': [30, 60, 120, 240, 320], 'blocks': [3, 3, 4, 5, 6]},
        # Target: ~20M params (Actual: ~19.7M) - Recommended Default
        'large':  {'channels': [32, 64, 128, 256, 384], 'blocks': [3, 4, 4, 6, 8]},
        # Target: ~32M params (Actual: ~32.3M)
        'huge':   {'channels': [32, 64, 128, 256, 512], 'blocks': [3, 4, 6, 6, 10]},
        # Target: ~48M params (Actual: ~50.3M)
        'pro':    {'channels': [40, 80, 160, 320, 640], 'blocks': [3, 4, 6, 6, 10]},
    }


    if size not in configs:
        valid_sizes = ", ".join(configs.keys())
        raise ValueError(
            f"Unknown size '{size}'. Choose from: {valid_sizes}\n"
            f"Recommended: 'large' (Target ~20M V2 params)"
        )

    cfg = configs[size]
    model_config = {
        'channels': cfg['channels'],
        'blocks': cfg['blocks'],
        'crf_range': crf_range,
        'preset_range': preset_range,
        'norm_range': norm_range
    }

    # Determine conditioning mode for logging
    is_crf_only = (preset_range[0] == preset_range[1])
    cond_mode = "CRF-Only" if is_crf_only else "CRF+Preset"

    logger.info("=" * 70)
    logger.info(f"Creating AV1ConditionalUNetPro - Size: {size.upper()}")
    logger.info("  (SOTA: Wavelet + Swin + DenseFusion)")
    logger.info(f"  Conditioning Mode: {cond_mode}")
    logger.info(f"  Channel Progression: {cfg['channels']}")
    logger.info(f"  Blocks per Level: {cfg['blocks']}")
    logger.info(f"  CRF Normalization Range: {crf_range}")
    if not is_crf_only:
        logger.info(f"  Preset Normalization Range: {preset_range}")
    logger.info(f"  Image Normalization Range: {norm_range}")
    logger.info("=" * 70)

    # Instantiate the V3 model
    model = AV1ConditionalUNetPro(model_config)

    # --- Verification Step (Shows ACTUAL count) ---
    actual_params = sum(p.numel() for p in model.parameters())
    logger.info(f"  ✓ V3 Model Instantiated - Actual Parameters: {actual_params:,} ({actual_params/1e6:.2f}M)")
    # --- End Verification ---

    return model


# ==============================================================================
# SECTION 5: Example Usage & Testing
# ==============================================================================

if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)
    
    model_sizes = ['nano', 'tiny', 'small', 'base', 'large', 'huge', 'pro']
    
    for model_size in model_sizes:
        print("\n" + "=" * 70)
        print(f"Testing AV1ConditionalUNetPro - Size: '{model_size}'")
        print("=" * 70 + "\n")
        
        try:
            # --- Test Case 1: CRF-Only ---
            model_crf_only = create_av1_restorer_v3(
                size=model_size,
                crf_range=(44, 53),
                preset_range=(4, 4), # Single value preset range
                norm_range=(0, 1)
            )
            model_crf_only.eval()

            # Test forward pass for CRF-Only model
            batch_size = 1
            # Use 256x256 to ensure window_size=8 and 5 downsamples (256/32=8) works
            lq = torch.randn(batch_size, 3, 256, 256) 
            crf = torch.tensor([[48.0]])

            print(f"CRF-Only Input shapes:")
            print(f"  LQ: {lq.shape}")
            print(f"  CRF: {crf.shape} - values: {crf.squeeze().tolist()}")

            with torch.no_grad():
                restored = model_crf_only(lq, crf) # Correctly call with only lq, crf

            print(f"\nCRF-Only Output:")
            print(f"  Shape: {restored.shape}")
            print(f"  Range: [{restored.min():.3f}, {restored.max():.3f}]")
            
            assert restored.shape == lq.shape
            
            # --- Test Case 2: CRF + Preset ---
            model_std = create_av1_restorer_v3(
                size=model_size,
                crf_range=(23, 63),
                preset_range=(0, 8), # Variable preset range
                norm_range=(-1, 1)
            )
            model_std.eval()
            
            lq_std = torch.randn(batch_size, 3, 256, 256)
            crf_std = torch.tensor([[30.0]])
            preset_std = torch.tensor([[5.0]])
            
            with torch.no_grad():
                restored_std = model_std(lq_std, crf_std, preset_std)
                
            print(f"\nCRF+Preset Output:")
            print(f"  Shape: {restored_std.shape}")
            print(f"  Range: [{restored_std.min():.3f}, {restored_std.max():.3f}]")
            
            assert restored_std.shape == lq_std.shape

            print(f"\n✓ Test PASSED for size: '{model_size}'")
            
        except Exception as e:
            print(f"\n✗ Test FAILED for size: '{model_size}'")
            print(f"  Error: {e}")
            import traceback
            traceback.print_exc()
