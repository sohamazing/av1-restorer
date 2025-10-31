# av1_restorer/models/av1_unet_restorer.py
"""
AV1 U-Net Restorer

Efficient U-Net with FiLM conditioning for CRF and Preset-adaptive restoration.

Architecture Highlights:
  - U-Net backbone with 5 levels (4 downsample + bottleneck)
  - EfficientResBlocks (depthwise separable + SE attention)
  - SimpleSelfAttention at bottleneck (channel attention)
  - FiLM conditioning at encoder levels and bottleneck
  - Residual learning (predicts artifact correction)
  - SOTA Upsampling: Bilinear Interpolation + Conv

Author: Soham Mukherjee
Version: 2.1 
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List, Union, Optional

import logging
logger = logging.getLogger(__name__)

from .blocks import DepthwiseSeparable, ECA, EfficientResBlock, choose_num_groups

# ==============================================================================
# SECTION 2: Conditioning + Channel Attention 
# ==============================================================================

class SimpleSelfAttention(nn.Module):
    """
    Efficient channel-wise self-attention for bottleneck.
    
    Optimizations:
      - Spatial reduction (2×) → 4× fewer pixels
      - Channel reduction (8×) → 8× fewer channels  
      - Channel attention (not spatial) → cheaper computation
    
    Complexity: O(C²) instead of O((HW)²)
    """
    def __init__(self, channels: int, reduction_factor: int = 2, channel_reduction: int = 8):
        super().__init__()
        self.channels = channels
        self.reduction_factor = reduction_factor
        self.hidden_ch = channels // channel_reduction
        
        # Spatial reduction
        self.pool = nn.AvgPool2d(kernel_size=reduction_factor, stride=reduction_factor)
        
        # Q, K, V projections (reduced channels)
        self.to_q = nn.Conv2d(channels, self.hidden_ch, 1)
        self.to_k = nn.Conv2d(channels, self.hidden_ch, 1)
        self.to_v = nn.Conv2d(channels, self.hidden_ch, 1)
        
        # Output projection
        self.out_proj = nn.Conv2d(self.hidden_ch, channels, 1)
        
        # Upsample back
        self.upsample = nn.Upsample(scale_factor=reduction_factor, mode='bilinear', align_corners=False)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        B, C, H, W = x.shape
        
        # Spatial reduction
        x_reduced = self.pool(x)  # [B, C, H/2, W/2]
        
        # Generate Q, K, V
        q = self.to_q(x_reduced)  # [B, hidden_ch, H/2, W/2]
        k = self.to_k(x_reduced)
        v = self.to_v(x_reduced)
        
        # Reshape for channel attention: [B, C, H, W] → [B, C, N]
        B, C_h, H_r, W_r = q.shape
        N = H_r * W_r
        
        q = q.view(B, C_h, N)  # [B, hidden_ch, N]
        k = k.view(B, C_h, N)
        v = v.view(B, C_h, N)
        
        # Channel attention: Q @ K^T (attend in channel space)
        attn_scores = torch.bmm(q, k.transpose(1, 2))  # [B, hidden_ch, hidden_ch]
        attn_scores = attn_scores / (C_h ** 0.5)  # Scale
        attn_weights = F.softmax(attn_scores, dim=-1)
        
        # Apply attention to values
        attended = torch.bmm(attn_weights, v)  # [B, hidden_ch, N]
        attended = attended.view(B, C_h, H_r, W_r)
        
        # Project back to original channels
        out = self.out_proj(attended)  # [B, C, H/2, W/2]
        
        # Upsample to original resolution
        out = self.upsample(out)  # [B, C, H, W]
        
        return identity + out  # Residual connection


class ConditioningEmbedder(nn.Module):
    """
    Unified Embedder: Generates the appropriate vector (128-dim or 192-dim) 
    based on the presence of the 'preset' input.
    """
    def __init__(
        self,
        crf_range: Tuple[float, float],
        preset_range: Tuple[float, float]
    ):
        super().__init__()
        self.crf_min, self.crf_max = crf_range
        self.preset_min, self.preset_max = preset_range
        
        # CRF embedder: 1 → 64 → 128
        self.crf_embedder = nn.Sequential(nn.Linear(1, 64), nn.GELU(), nn.Linear(64, 128))
        
        # Preset embedder: 1 → 32 → 64
        self.preset_embedder = nn.Sequential(nn.Linear(1, 32), nn.GELU(), nn.Linear(32, 64))
        
    def _normalize(self, val: torch.Tensor, min_val: float, max_val: float) -> torch.Tensor:
        """Helper for robust normalization."""
        range_val = max_val - min_val
        if range_val == 0:
             return torch.zeros_like(val)
        return (val - min_val) / range_val

    def forward(self, crf: torch.Tensor, preset: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Generates the conditioning vector (128-dim or 192-dim).
        """
        crf_norm = self._normalize(crf, self.crf_min, self.crf_max)
        crf_emb = self.crf_embedder(crf_norm)
        
        # Determine conditioning mode
        if preset is not None and self.preset_max != self.preset_min:
            # CRF + Preset Mode (192-dim)
            preset_norm = self._normalize(preset, self.preset_min, self.preset_max)
            preset_emb = self.preset_embedder(preset_norm)
            cond = torch.cat([crf_emb, preset_emb], dim=1)
        else:
            # CRF-Only Mode (128-dim)
            cond = crf_emb
        
        # CRITICAL: Clamp embeddings for stability
        return torch.clamp(cond, min=-10.0, max=10.0)


class FiLMLayer(nn.Module):
    """Feature-wise Linear Modulation with stability constraints."""
    def __init__(self, cond_dim: int, feature_channels: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(cond_dim, cond_dim),
            nn.GELU(),
            nn.Linear(cond_dim, feature_channels * 2)
        )
    
    def forward(self, features: torch.Tensor, conditioning: torch.Tensor) -> torch.Tensor:
        B, C, H, W = features.shape
        params = self.mlp(conditioning)
        gamma, beta = params.chunk(2, dim=1)
        
        # ========== CRITICAL FIX: Clamp FiLM parameters ==========
        gamma = torch.clamp(gamma, min=-5.0, max=5.0)
        beta = torch.clamp(beta, min=-5.0, max=5.0)
        # =========================================================
        
        gamma = gamma.view(B, C, 1, 1)
        beta = beta.view(B, C, 1, 1)
        return gamma * features + beta


# ==============================================================================
# SECTION 3: Main U-Net Architecture
# ==============================================================================

class AV1ConditionalUNet(nn.Module):
    """
    Complete AV1 artifact removal U-Net with FiLM conditioning.
    Unified U-Net capable of CRF (128-dim) or CRF-Preset (192-dim) conditioning.    

    Architecture:
      - 5-level U-Net with skip connections.
      - EfficientResBlocks with GroupNorm for stable, high-performance training.
      - FiLM conditioning for adapting to CRF and Preset values.
      - SimpleSelfAttention at the bottleneck to capture global context.
      - SOTA Upsampling (Bilinear + DepthwiseConv) to prevent checkerboard artifacts.
      - Residual learning (model predicts the artifacts to be removed).
    
    Args:
        config (dict): Configuration with keys:
          - channels (list): Channel counts per level, e.g., [24, 48, 96, 192, 256]
          - blocks (list): Number of EfficientResBlocks per level, e.g., [2, 2, 3, 3, 6]
          - crf_range (tuple): (min, max) for CRF normalization
          - preset_range (tuple): (min, max) for Preset normalization
    
    Example config:
        config = {
            'channels': [24, 48, 96, 192, 256],
            'blocks': [2, 2, 3, 3, 6],
            'crf_range': (23, 63),
            'preset_range': (4, 4)
            'norm_range': (-1, 1)
        }
    """
    def __init__(self, config: dict):
        super().__init__()
        
        ch = config['channels']  # e.g., [24, 48, 96, 192, 256]
        blocks = config['blocks']  # e.g., [2, 2, 3, 3, 6]
        crf_range = config.get('crf_range', (23, 63))
        preset_range = config.get('preset_range', (0, 8))
        norm_range = tuple(config.get('norm_range', (0, 1)))
        self.clamp_min, self.clamp_max = norm_range
        # Model adapts its conditioning dimension based on the preset_range
        self.cond_dim = 128 if preset_range[0] == preset_range[1] else 192 
            
        logger.info("="*60)
        logger.info("Initializing AV1ConditionalUNet")
        logger.info(f"  Cond Dim: {self.cond_dim}")
        logger.info(f"  Channels: {ch}")
        logger.info(f"  Blocks: {blocks}")
        logger.info(f"  CRF Range: {crf_range}")
        logger.info(f"  Norm Range: [{self.clamp_min}, {self.clamp_max}]")
        logger.info(f"  Upsampling: Bilinear + DepthwiseSeparable (SOTA)")
        logger.info("="*60)
        
        # ===== CONDITIONING SYSTEM =====
        self.conditioning_embedder = ConditioningEmbedder(
            crf_range=crf_range, preset_range=preset_range
        )
        
        # ===== INPUT HEAD =====
        self.head = nn.Sequential(
            nn.Conv2d(3, ch[0], kernel_size=3, padding=1, padding_mode='reflect', bias=False),
            nn.GroupNorm(choose_num_groups(ch[0]), ch[0]),
            nn.GELU(),
            *[EfficientResBlock(ch[0]) for _ in range(blocks[0])]
        )
        
        # ===== ENCODER =====
        self.encoder1 = self._build_encoder_level(ch[0], ch[1], blocks[1], self.cond_dim)
        self.encoder2 = self._build_encoder_level(ch[1], ch[2], blocks[2], self.cond_dim)
        self.encoder3 = self._build_encoder_level(ch[2], ch[3], blocks[3], self.cond_dim)
        
        # ===== BOTTLENECK =====
        self.bottleneck_down = nn.Conv2d(
            ch[3], ch[4], kernel_size=3, stride=2, padding=1, padding_mode='reflect', bias=False
        )

        self.bottleneck_gn = nn.GroupNorm(choose_num_groups(ch[4]), ch[4])
        
        self.bottleneck_pre_attn = nn.Sequential(
            *[EfficientResBlock(ch[4]) for _ in range(blocks[4] // 2)]
        )
        self.bottleneck_attn = SimpleSelfAttention(ch[4])
        self.bottleneck_film = FiLMLayer(self.cond_dim, ch[4])
        self.bottleneck_post_attn = nn.Sequential(
            *[EfficientResBlock(ch[4]) for _ in range(blocks[4] // 2)]
        )
        
        # ===== DECODER =====
        self.decoder3 = self._build_decoder_level(ch[4], ch[3], blocks[3])
        self.decoder2 = self._build_decoder_level(ch[3], ch[2], blocks[2])
        self.decoder1 = self._build_decoder_level(ch[2], ch[1], blocks[1])
        
        # ===== OUTPUT TAIL =====
        self.tail_upsample_op = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.tail_upsample_conv = DepthwiseSeparable(ch[1], ch[0], stride=1)
        self.tail_gn = nn.GroupNorm(choose_num_groups(ch[0]), ch[0])
        self.tail_fusion = nn.Conv2d(ch[0] * 2, ch[0], kernel_size=1, bias=False)
        self.tail_fusion_gn_act = nn.Sequential(
            nn.GroupNorm(choose_num_groups(ch[0]), ch[0]),
            nn.GELU()
        )
        self.tail_body = nn.Sequential(
            *[EfficientResBlock(ch[0]) for _ in range(blocks[0])]
        )
        self.tail_pred = nn.Conv2d(ch[0], 3, kernel_size=3, padding=1, padding_mode='reflect')
        
        self._init_weights()
        
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logger.info(f"✓ Model initialized")
        logger.info(f"  Total parameters: {total_params:,} ({total_params/1e6:.2f}M)")
        logger.info(f"  Trainable parameters: {trainable_params:,} ({trainable_params/1e6:.2f}M)")
        logger.info("="*60)


    def _build_encoder_level(self, in_ch: int, out_ch: int, num_blocks: int, cond_dim: int) -> nn.ModuleDict:
        """Helper to construct one level of the U-Net encoder."""
        return nn.ModuleDict({
            'downsample': nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=2, padding=1, padding_mode='reflect', bias=False),
                nn.GroupNorm(choose_num_groups(out_ch), out_ch),
                nn.GELU()
            ),
            'body': nn.Sequential(
                *[EfficientResBlock(out_ch) for _ in range(num_blocks)]
            ),
            'film': FiLMLayer(cond_dim, out_ch)
        })

    def _build_decoder_level(self, in_ch: int, out_ch: int, num_blocks: int) -> nn.ModuleDict:
        """Helper to construct one level of the U-Net decoder."""
        return nn.ModuleDict({
            'upsample_op': nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            'upsample_conv': DepthwiseSeparable(in_ch, out_ch, stride=1),
            'upsample_gn': nn.GroupNorm(choose_num_groups(out_ch), out_ch),
            
            'fusion': nn.Sequential(
                nn.Conv2d(out_ch * 2, out_ch, kernel_size=1, bias=False),
                nn.GroupNorm(choose_num_groups(out_ch), out_ch),
                nn.GELU()
            ),
            
            'body': nn.Sequential(
                *[EfficientResBlock(out_ch) for _ in range(num_blocks)]
            )
        })

    def _init_weights(self):
        """Initializes model weights, zeroing the final prediction layer."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.5)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        
        # Initialize the final layer to zero for residual learning
        nn.init.zeros_(self.tail_pred.weight)
        nn.init.zeros_(self.tail_pred.bias)
        logger.info("✓ Weights initialized (tail_pred zeroed for residual learning)")

    def forward(self, lq_image: torch.Tensor, crf: torch.Tensor, preset: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Defines the full forward pass, used for training (CRF or CRF-Preset conditioninal).
        
        Args:
            lq_image: [B, 3, H, W] Low-quality compressed image
            crf: [B, 1] CRF value, range [23, 63]
            preset: [B, 1] Optional preset value
        
        Returns:
            [B, 3, H, W] Restored image
        """
        # 1. Generate conditioning vector
        cond = self.conditioning_embedder(crf, preset)

        # 2. Encoder Path
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
        
        # 3. Bottleneck
        b = self.bottleneck_down(skip3)
        b = F.gelu(self.bottleneck_gn(b))
        b = torch.clamp(b, min=-10.0, max=10.0)
        
        b = self.bottleneck_pre_attn(b)
        b = self.bottleneck_attn(b)
        b = self.bottleneck_film(b, cond)
        b = torch.clamp(b, min=-10.0, max=10.0)
        b = self.bottleneck_post_attn(b)
        
        # 4. Decoder Path (with skip connections)
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
        d1 = torch.cat([d1, skip1], dim=1)
        d1 = self.decoder1['fusion'](d1)
        d1 = self.decoder1['body'](d1)
        
        # 5. Output Tail
        t = self.tail_upsample_op(d1)
        t = self.tail_upsample_conv(t)
        t = F.gelu(self.tail_gn(t))
        t = torch.cat([t, skip0], dim=1)
        t = self.tail_fusion(t)
        t = self.tail_fusion_gn_act(t)
        t = self.tail_body(t)
        residual = self.tail_pred(t)
        
        # 6. Residual Connection
        restored = lq_image + residual
        restored = torch.clamp(restored, self.clamp_min, self.clamp_max)
        
        return restored

    # ==========================================================================
    # --- SOTA INFERENCE & HELPER METHODS
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
            return self.forward(tile, crf_batch)
        else:
            if preset_batch is None:
                 raise ValueError("Preset input missing in CRF+Preset mode inference.")
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
# SECTION 4: Unified Model Factory
# ==============================================================================

def create_av1_restorer(
    size: str = 'large', # Defaulting to ~20M target
    crf_range: Tuple[int, int] = (23, 63),
    preset_range: Tuple[int, int] = (0, 8),
    norm_range: Tuple[float, float] = (-1, 1)
) -> AV1ConditionalUNet:
    """
    Factory for SOTA-inspired, empirically calibrated AV1 Conditional U-Nets.

    Creates architectures based on measured parameter counts, labeled with clean,
    rounded target sizes (multiples of 2M or 5M). Uses GroupNorm, efficient
    blocks, and bottleneck attention, ensuring Base Width >= 16.

    Size Options & Target / Actual Parameters (Approx. CRF-Only):
    ----------------------------------------------------------
    nano:    Target ~2M  (Actual: ~2.3M)  - Minimal viable conditional
    tiny:    Target ~4M  (Actual: ~5.0M)  - Lightweight conditional
    small:   Target ~8M (Actual: ~9.2M)  - Standard balanced conditional (Note: Actual < Target)
    base:   Target ~12M (Actual: ~12.8M) - Enhanced standard conditional
    large:  Target ~20M (Actual: ~19.7M) - RECOMMENDED High quality default
    huge:   Target ~32M (Actual: ~32.3M) - High quality conditional
    pro:    Target ~48M (Actual: ~50.3M) - Maximum quality conditional

    Args:
        size (str): Model variant ('nano' to 'pro'). Defaults to 'large'.
        crf_range (Tuple[int, int]): CRF normalization range (min, max).
        preset_range (Tuple[int, int]): Preset range. Use identical min/max
                                         (e.g., [4, 4]) for CRF-only mode.
        norm_range (Tuple[float, float]): Image normalization range (e.g., [-1, 1]).

    Returns:
        AV1ConditionalUNet: Initialized model instance with GroupNorm.

    Raises:
        ValueError: If an unknown size string is provided.

    Example:
        >>> # Create the recommended ~20M target model for CRF-only
        >>> model = create_av1_restorer(size='large', preset_range=[4, 4])
        >>> params = sum(p.numel() for p in model.parameters()) / 1e6
        >>> print(f"Model size 'large' has {params:.2f}M parameters.") # Actual count
        Model size 'large' has 19.69M parameters.
    """
    # FINAL Configurations - Based on Empirical Parameter Counts (Oct 25 Run)
    # Comments reflect clean targets; architecture yields known actual counts.
    configs = {
        # Target: ~2M params (Actual: ~2.3M) - Minimum Viable Conditional
        'nano':   {'channels': [16, 32, 64, 128, 192], 'blocks': [2, 2, 2, 3, 3]},
        # Target: ~4M params (Actual: ~5.0M) - Improved Lightweight
        'tiny':   {'channels': [20, 40, 80, 160, 240], 'blocks': [2, 2, 3, 4, 5]},
        # Target: ~8M params (Actual: ~9.2M) - Standard Balanced (Slightly under target)
        'small':  {'channels': [24, 48, 96, 192, 288], 'blocks': [3, 3, 4, 5, 6]},
        # Target: ~12M params (Actual: ~12.8M) - Enhanced Standard
        'base':   {'channels': [30, 60, 120, 240, 320], 'blocks': [3, 3, 4, 5, 6]},
        # Target: ~20M params (Actual: ~19.7M) - Recommended Default
        'large':  {'channels': [32, 64, 128, 256, 384], 'blocks': [3, 4, 4, 6, 8]},
        # Target: ~32M params (Actual: ~32.3M) - High Quality
        'huge':   {'channels': [32, 64, 128, 256, 512], 'blocks': [3, 4, 6, 6, 10]},
        # Target: ~48M params (Actual: ~50.3M) - Max Quality / Research
        'pro':    {'channels': [40, 80, 160, 320, 640], 'blocks': [3, 4, 6, 6, 10]},
    }


    if size not in configs:
        valid_sizes = ", ".join(configs.keys())
        raise ValueError(
            f"Unknown size '{size}'. Choose from: {valid_sizes}\n"
            f"Recommended: 'large' (Target ~20M / Actual ~19.7M params)" # Updated recommendation
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

    # Professional Logging Output using Clean Target Parameter Counts
    # Mapping clean targets for logging
    param_targets = {
        'nano': '~2M', 
        'tiny': '~4M', 
        'small': '~8M', 
        'base': '~12M',
        'large': '~20M',
        'huge': '~32M',
        'pro': '~48M'
    }
    param_target_log = param_targets.get(size, 'Unknown Target')

    logger.info("=" * 70)
    logger.info(f"Creating AV1ConditionalUNet - Size: {size.upper()} (Target: {param_target_log} params - FINAL EMPIRICAL ARCH)") # Updated note
    logger.info(f"  Conditioning Mode: {cond_mode}")
    logger.info(f"  Channel Progression: {cfg['channels']}")
    logger.info(f"  Blocks per Level: {cfg['blocks']}")
    logger.info(f"  CRF Normalization Range: {crf_range}")
    if not is_crf_only:
        logger.info(f"  Preset Normalization Range: {preset_range}")
    logger.info(f"  Image Normalization Range: {norm_range}")
    logger.info("=" * 70)

    # Instantiate the model (ensure AV1ConditionalUNet uses GroupNorm)
    model = AV1ConditionalUNet(model_config)

    # --- Verification Step (Shows ACTUAL count) ---
    actual_params = sum(p.numel() for p in model.parameters())
    logger.info(f"  ✓ Model Instantiated - Actual Parameters: {actual_params:,} ({actual_params/1e6:.2f}M)")
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
        # --- Test Case 1: Standard CRF + Preset ---
        print("\n--- Testing Standard Model (CRF + Preset) ---")
        model_std = create_av1_restorer(
            size=model_size,
            crf_range=(23, 63),
            preset_range=(0, 8), # Variable preset range
            norm_range=(-1, 1)
        )
        print(f"Instantiated model type: {type(model_std).__name__}")
        # (Optional: Add forward pass test for model_std if needed)

        # --- Test Case 2: CRF-Only ---
        print("\n--- Testing CRF-Only Model ---")
        model_crf_only = create_av1_restorer(
            size=model_size,
            crf_range=(44, 53),
            preset_range=(4, 4), # Single value preset range
            norm_range=(0, 1)
        )
        print(f"Instantiated model type: {type(model_crf_only).__name__}")

        # Test forward pass for CRF-Only model
        batch_size = 1
        lq = torch.randn(batch_size, 3, 128, 128)
        crf = torch.tensor([[48.0]])

        print(f"\nCRF-Only Input shapes:")
        print(f"  LQ: {lq.shape}")
        print(f"  CRF: {crf.shape} - values: {crf.squeeze().tolist()}")

        with torch.no_grad():
            model_crf_only.eval() # Ensure model is in eval mode for testing
            restored = model_crf_only(lq, crf) # Correctly call with only lq, crf

        print(f"\nCRF-Only Output:")
        print(f"  Shape: {restored.shape}")
        print(f"  Range: [{restored.min():.3f}, {restored.max():.3f}]")

        # --- Parameter Count Check (Example using the CRF-Only model) ---
        print(f"\n--- Parameter Count Check (using CRF-Only '{model_size}' model) ---")
        total_params = sum(p.numel() for p in model_crf_only.parameters())
        trainable_params = sum(p.numel() for p in model_crf_only.parameters() if p.requires_grad)
        print(f"  Total parameters: {total_params:,} ({total_params/1e6:.2f}M)")
        print(f"  Trainable parameters: {trainable_params:,} ({trainable_params/1e6:.2f}M)")
