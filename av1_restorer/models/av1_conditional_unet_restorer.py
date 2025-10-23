# av1_restorer/models/av1_unet_restorer.py
"""
AV1 U-Net Restorer - Production Implementation

Efficient U-Net with FiLM conditioning for CRF and Preset-adaptive restoration.

Architecture Highlights:
  - U-Net backbone with 5 levels (4 downsample + bottleneck)
  - EfficientResBlocks (depthwise separable + SE attention)
  - SimpleSelfAttention at bottleneck (channel attention)
  - FiLM conditioning at encoder levels and bottleneck
  - Residual learning (predicts artifact correction)

Author: Soham Mukherjee
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List, Union

import logging
logger = logging.getLogger(__name__)


# ==============================================================================
# SECTION 1: Core Building Blocks
# ==============================================================================

class DepthwiseSeparableConv(nn.Module):
    """
    Depthwise Separable Convolution: ~8× parameter reduction vs standard conv.
    
    Flow: Depthwise (spatial) → Pointwise (channel mixing) → BatchNorm → GELU
    """
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3, stride: int = 1):
        super().__init__()
        padding = kernel_size // 2
        self.depthwise = nn.Conv2d(
            in_ch, in_ch, kernel_size, stride, padding, 
            groups=in_ch, bias=False
        )
        self.pointwise = nn.Conv2d(in_ch, out_ch, 1, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.gelu = nn.GELU()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.gelu(self.bn(self.pointwise(self.depthwise(x))))


class SqueezeExcitation(nn.Module):
    """
    Squeeze-and-Excitation: Channel-wise attention mechanism.
    
    Flow: Global Pool → FC (compress) → ReLU → FC (expand) → Sigmoid → Scale
    """
    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1),
            nn.GELU(),
            nn.Conv2d(channels // reduction, channels, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.excitation(self.squeeze(x))


class EfficientResBlock(nn.Module):
    """
    MobileNetV3-inspired inverted residual block with SE attention.
    
    Flow: Expand → Depthwise → SE → Project → Residual Add
    Parameters: ~4C² (vs 9C² for standard ResNet block)
    """
    def __init__(self, channels: int, expansion_ratio: int = 2):
        super().__init__()
        hidden_dim = channels * expansion_ratio
        
        # Expansion
        self.expand = nn.Sequential(
            nn.Conv2d(channels, hidden_dim, 1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.GELU()
        )
        
        # Depthwise convolution
        self.dwconv = DepthwiseSeparableConv(hidden_dim, hidden_dim, 3)
        
        # Squeeze-Excitation
        self.se = SqueezeExcitation(hidden_dim, reduction=4)
        
        # Projection
        self.project = nn.Sequential(
            nn.Conv2d(hidden_dim, channels, 1, bias=False),
            nn.BatchNorm2d(channels)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.expand(x)
        out = self.dwconv(out)
        out = self.se(out)
        out = self.project(out)
        return out + identity  # Residual connection


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


# ==============================================================================
# SECTION 2: Conditioning Modules
# ==============================================================================
class CRFConditioningEmbedder(nn.Module):
    """
    Embeds ONLY CRF values into a conditioning vector.
    
    Args:
        crf_min, crf_max: CRF normalization range (default: 23-63)
    
    Output: [B, 128] CRF embedding
    """
    def __init__(
        self,
        crf_min: float = 23.0,
        crf_max: float = 63.0,
    ):
        super().__init__()
        self.crf_min = crf_min
        self.crf_max = crf_max
        
        # CRF embedder: 1 -> 64 -> 128
        self.crf_embedder = nn.Sequential(
            nn.Linear(1, 64),
            nn.GELU(),
            nn.Linear(64, 128)
        )
    
    def forward(self, crf: torch.Tensor) -> torch.Tensor:
        """
        Args:
            crf: [B, 1] range [crf_min, crf_max]
        
        Returns:
            [B, 128] combined embedding
        """
        # Normalize to [0, 1]
        crf_range = self.crf_max - self.crf_min
        crf_norm = (crf - self.crf_min) / crf_range if crf_range > 0 else torch.zeros_like(crf)
        
        # Embed
        crf_emb = self.crf_embedder(crf_norm)  # [B, 128]

        # CRITICAL: Clamp embeddings
        crf_emb = torch.clamp(crf_emb, -10.0, 10.0)
        
        return crf_emb

class ConditioningEmbedder(nn.Module):
    """
    Embeds CRF and Preset values into a shared conditioning vector.
    
    Args:
        crf_min, crf_max: CRF normalization range (default: 23-63)
        preset_min, preset_max: Preset normalization range (default: 0-8)
    
    Output: [B, 192] combined embedding (128 from CRF + 64 from Preset)
    """
    def __init__(
        self,
        crf_min: float = 23.0,
        crf_max: float = 63.0,
        preset_min: float = 0.0,
        preset_max: float = 8.0
    ):
        super().__init__()
        self.crf_min = crf_min
        self.crf_max = crf_max
        self.preset_min = preset_min
        self.preset_max = preset_max
        
        # CRF embedder: 1 → 64 → 128
        self.crf_embedder = nn.Sequential(
            nn.Linear(1, 64),
            nn.GELU(),
            nn.Linear(64, 128)
        )
        
        # Preset embedder: 1 → 32 → 64
        self.preset_embedder = nn.Sequential(
            nn.Linear(1, 32),
            nn.GELU(),
            nn.Linear(32, 64)
        )
    
    def forward(self, crf: torch.Tensor, preset: torch.Tensor) -> torch.Tensor:
        """
        Args:
            crf: [B, 1] range [crf_min, crf_max]
            preset: [B, 1] range [preset_min, preset_max]
        
        Returns:
            [B, 192] combined embedding
        """
        # Normalize to [0, 1]
        # --- ROBUST NORMALIZATION ---
        crf_range = self.crf_max - self.crf_min
        preset_range = self.preset_max - self.preset_min

        # Handle division-by-zero if range is a single value (e.g., [4, 4])
        # If range is 0, the normalized value should be 0.
        crf_norm = (crf - self.crf_min) / crf_range if crf_range > 0 else torch.zeros_like(crf)
        preset_norm = (preset - self.preset_min) / preset_range if preset_range > 0 else torch.zeros_like(preset)
        
        # Embed
        crf_emb = self.crf_embedder(crf_norm)      # [B, 128]
        preset_emb = self.preset_embedder(preset_norm)  # [B, 64]

        # CRITICAL: Clamp embeddings to prevent NaN propagation in AMP
        crf_emb = torch.clamp(crf_emb, -10.0, 10.0)
        preset_emb = torch.clamp(preset_emb, -10.0, 10.0)
        
        # Concatenate
        return torch.cat([crf_emb, preset_emb], dim=1)  # [B, 192]


class FiLMLayer(nn.Module):
    """
    Feature-wise Linear Modulation: Applies affine transformation conditioned on CRF+Preset.
    
    Math: output = γ(cond) ⊙ features + β(cond)
    
    Args:
        cond_dim: Dimension of conditioning vector (default: 192)
        feature_channels: Number of feature channels to modulate
    """
    def __init__(self, cond_dim: int, feature_channels: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(cond_dim, cond_dim),
            nn.GELU(),
            nn.Linear(cond_dim, feature_channels * 2)  # γ and β
        )
    
    def forward(self, features: torch.Tensor, conditioning: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: [B, C, H, W]
            conditioning: [B, cond_dim]
        
        Returns:
            [B, C, H, W] modulated features
        """
        B, C, H, W = features.shape
        
        # Generate scale (γ) and shift (β)
        params = self.mlp(conditioning)  # [B, 2C]
        gamma, beta = params.chunk(2, dim=1)  # [B, C], [B, C]
        
        # Reshape for broadcasting: [B, C] → [B, C, 1, 1]
        gamma = gamma.view(B, C, 1, 1)
        beta = beta.view(B, C, 1, 1)
        
        # Apply affine transformation
        return gamma * features + beta


# ==============================================================================
# SECTION 3: Main U-Net Architecture
# ==============================================================================

class AV1ConditionalUNetCRF(nn.Module):
    """
    CRF-ONLY Conditional U-Net.
    
    This version is identical to AV1ConditionalUNet but ONLY conditions
    on CRF, removing all 'preset' dependencies.
    
    cond_dim is 128 (from CRF) instead of 192 (CRF + Preset).
    """
    
    def __init__(self, config: dict):
        super().__init__()
        
        ch = config['channels']  # e.g., [24, 48, 96, 192, 256]
        blocks = config['blocks']  # e.g., [2, 2, 3, 3, 6]
        crf_range = config.get('crf_range', (23, 63))
        norm_range = tuple(config.get('norm_range', (0, 1)))
        self.clamp_min, self.clamp_max = norm_range
        cond_dim = 128  # Fixed: 128 (CRF-only)
            
        logger.info("="*60)
        logger.info("Initializing AV1ConditionalUNetCRF") # <-- MODIFIED
        logger.info(f"  Channels: {ch}")
        logger.info(f"  Blocks: {blocks}")
        logger.info(f"  CRF Range: {crf_range}")
        logger.info(f"  Norm Range: [{self.clamp_min}, {self.clamp_max}]")
        logger.info("="*60)
        
        # ===== CONDITIONING SYSTEM =====
        # --- MODIFIED ---
        self.conditioning_embedder = CRFConditioningEmbedder(
            crf_min=crf_range[0],
            crf_max=crf_range[1]
        )
        # --- END MODIFIED ---
        
        # ===== INPUT HEAD (Level 0) =====
        self.head = nn.Sequential(
            nn.Conv2d(3, ch[0], kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(ch[0]),
            nn.GELU(),
            *[EfficientResBlock(ch[0]) for _ in range(blocks[0])]
        )
        
        # ===== ENCODER =====
        # All FiLM layers are now created with cond_dim=128
        self.encoder1 = self._build_encoder_level(ch[0], ch[1], blocks[1], cond_dim)
        self.encoder2 = self._build_encoder_level(ch[1], ch[2], blocks[2], cond_dim)
        self.encoder3 = self._build_encoder_level(ch[2], ch[3], blocks[3], cond_dim)
        
        # ===== BOTTLENECK =====
        self.bottleneck_down = nn.Conv2d(ch[3], ch[4], kernel_size=3, stride=2, padding=1, bias=False)
        self.bottleneck_bn = nn.BatchNorm2d(ch[4])
        
        self.bottleneck_pre_attn = nn.Sequential(
            *[EfficientResBlock(ch[4]) for _ in range(blocks[4] // 2)]
        )
        self.bottleneck_film1 = FiLMLayer(cond_dim, ch[4]) # cond_dim=128
        
        self.bottleneck_attn = SimpleSelfAttention(ch[4])
        
        self.bottleneck_film2 = FiLMLayer(cond_dim, ch[4]) # cond_dim=128
        self.bottleneck_post_attn = nn.Sequential(
            *[EfficientResBlock(ch[4]) for _ in range(blocks[4] // 2)]
        )
        
        # ===== DECODER =====
        self.decoder3 = self._build_decoder_level(ch[4], ch[3], blocks[3])
        self.decoder2 = self._build_decoder_level(ch[3], ch[2], blocks[2])
        self.decoder1 = self._build_decoder_level(ch[2], ch[1], blocks[1])
        
        # ===== OUTPUT TAIL =====
        self.tail_up = nn.ConvTranspose2d(ch[1], ch[0], kernel_size=2, stride=2, bias=False)
        self.tail_bn = nn.BatchNorm2d(ch[0])
        self.tail_fusion = nn.Conv2d(ch[0] * 2, ch[0], kernel_size=1, bias=False)
        self.tail_body = nn.Sequential(
            *[EfficientResBlock(ch[0]) for _ in range(blocks[0])]
        )
        self.tail_pred = nn.Conv2d(ch[0], 3, kernel_size=3, padding=1)
        
        self._init_weights()
        
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logger.info(f"✓ Model initialized (CRF-Only)")
        logger.info(f"  Total parameters: {total_params:,} ({total_params/1e6:.2f}M)")
        logger.info(f"  Trainable parameters: {trainable_params:,} ({trainable_params/1e6:.2f}M)")
        logger.info("="*60)

    # _build_encoder_level, _build_decoder_level, _init_weights are inherited
    # We can just copy them for simplicity, or rely on inheritance.
    # For a self-contained class, let's copy them.
    
    def _build_encoder_level(self, in_ch: int, out_ch: int, num_blocks: int, cond_dim: int) -> nn.ModuleDict:
        return nn.ModuleDict({
            'downsample': nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.GELU()
            ),
            'body': nn.Sequential(
                *[EfficientResBlock(out_ch) for _ in range(num_blocks)]
            ),
            'film': FiLMLayer(cond_dim, out_ch) # cond_dim will be 128
        })

    def _build_decoder_level(self, in_ch: int, out_ch: int, num_blocks: int) -> nn.ModuleDict:
        return nn.ModuleDict({
            'upsample': nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2, bias=False),
            'upsample_bn': nn.BatchNorm2d(out_ch),
            'fusion': nn.Sequential(
                nn.Conv2d(out_ch * 2, out_ch, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.GELU()
            ),
            'body': nn.Sequential(
                *[EfficientResBlock(out_ch) for _ in range(num_blocks)]
            )
        })

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.5)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        
        nn.init.zeros_(self.tail_pred.weight)
        nn.init.zeros_(self.tail_pred.bias)
        logger.info("✓ Weights initialized (tail_pred zeroed for residual learning)")

    # --- MODIFIED forward ---
    def forward(self, lq_image: torch.Tensor, crf: torch.Tensor) -> torch.Tensor:
        """
        Full forward pass through the network (CRF-Only).
        
        Args:
            lq_image: [B, 3, H, W] Low-quality compressed image
            crf: [B, 1] CRF value, range [23, 63]
        
        Returns:
            [B, 3, H, W] Restored image
        """
        # ===== 1. GENERATE CONDITIONING VECTOR =====
        cond = self.conditioning_embedder(crf)  # [B, 128] <-- MODIFIED
        
        # ===== 2. ENCODER PATH =====
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
        
        # ===== 3. BOTTLENECK =====
        b = self.bottleneck_down(skip3)
        b = F.gelu(self.bottleneck_bn(b))
        
        b = self.bottleneck_pre_attn(b)
        b = self.bottleneck_film1(b, cond)
        
        b = self.bottleneck_attn(b)
        
        b = self.bottleneck_film2(b, cond)
        b = self.bottleneck_post_attn(b)
        
        # ===== 4. DECODER PATH =====
        d3 = self.decoder3['upsample'](b)
        d3 = F.gelu(self.decoder3['upsample_bn'](d3))
        d3 = torch.cat([d3, skip3], dim=1)
        d3 = self.decoder3['fusion'](d3)
        d3 = self.decoder3['body'](d3)
        
        d2 = self.decoder2['upsample'](d3)
        d2 = F.gelu(self.decoder2['upsample_bn'](d2))
        d2 = torch.cat([d2, skip2], dim=1)
        d2 = self.decoder2['fusion'](d2)
        d2 = self.decoder2['body'](d2)
        
        d1 = self.decoder1['upsample'](d2)
        d1 = F.gelu(self.decoder1['upsample_bn'](d1))
        d1 = torch.cat([d1, skip1], dim=1)
        d1 = self.decoder1['fusion'](d1)
        d1 = self.decoder1['body'](d1)
        
        # ===== 5. OUTPUT TAIL =====
        t = self.tail_up(d1)
        t = F.gelu(self.tail_bn(t))
        t = torch.cat([t, skip0], dim=1)
        t = self.tail_fusion(t)
        t = self.tail_body(t)
        residual = self.tail_pred(t)
        
        # ===== 6. RESIDUAL LEARNING =====
        restored = lq_image + residual
        restored = torch.clamp(restored, self.clamp_min, self.clamp_max)
        
        return restored

    # --- MODIFIED inference ---
    @torch.no_grad()
    def inference(
        self,
        lq_image: torch.Tensor,
        crf: torch.Tensor,
        tile_size: int = 512,
        tile_overlap: int = 64,
        center_crop: bool = False
    ) -> torch.Tensor:
        """
        Memory-efficient tiled inference (CRF-Only).
        
        Args:
            lq_image: [B, 3, H, W] Input image
            crf: [B, 1] CRF value
            tile_size: ...
            tile_overlap: ...
            center_crop: ...
        
        Returns:
            [B, 3, H, W] Restored image
        """
        self.eval()
        B, C, H, W = lq_image.shape
        
        # ===== MODE 1: CENTER CROP ONLY =====
        if center_crop:
            crop_h = min(tile_size, H)
            crop_w = min(tile_size, W)
            start_h = (H - crop_h) // 2
            start_w = (W - crop_w) // 2
            center_tile = lq_image[:, :, start_h:start_h+crop_h, start_w:start_w+crop_w]
            
            pad_h = (32 - crop_h % 32) % 32
            pad_w = (32 - crop_w % 32) % 32
            
            if pad_h > 0 or pad_w > 0:
                center_tile = F.pad(center_tile, (0, pad_w, 0, pad_h), mode='reflect')
                output = self.forward(center_tile, crf) # <-- MODIFIED
                output = output[:, :, :crop_h, :crop_w]
            else:
                output = self.forward(center_tile, crf) # <-- MODIFIED
            return output
        
        # ===== MODE 2: SMALL IMAGE (FITS IN SINGLE TILE) =====
        if H <= tile_size and W <= tile_size:
            pad_h = (32 - H % 32) % 32
            pad_w = (32 - W % 32) % 32
            
            if pad_h > 0 or pad_w > 0:
                lq_padded = F.pad(lq_image, (0, pad_w, 0, pad_h), mode='reflect')
                output = self.forward(lq_padded, crf) # <-- MODIFIED
                return output[:, :, :H, :W]
            else:
                return self.forward(lq_image, crf) # <-- MODIFIED
        
        # ===== MODE 3: LARGE IMAGE (REQUIRES TILING) =====
        logger.info(f"Tiled inference: {H}×{W} image → {tile_size}×{tile_size} tiles (overlap: {tile_overlap}px)")
        
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
                tile_h, tile_w = tile.shape[2:]
                
                pad_h = (32 - tile_h % 32) % 32
                pad_w = (32 - tile_w % 32) % 32
                
                if pad_h > 0 or pad_w > 0:
                    tile_padded = F.pad(tile, (0, pad_w, 0, pad_h), mode='reflect')
                    restored_tile = self.forward(tile_padded, crf) # <-- MODIFIED
                    restored_tile = restored_tile[:, :, :tile_h, :tile_w]
                else:
                    restored_tile = self.forward(tile, crf) # <-- MODIFIED
                
                blend = self._create_blend_mask(
                    tile_h, tile_w, tile_overlap,
                    is_top=(i == 0), is_bottom=(h_end >= H),
                    is_left=(j == 0), is_right=(w_end >= W),
                    device=lq_image.device
                )
                
                output[:, :, h_start:h_end, w_start:w_end] += restored_tile * blend
                weight[:, :, h_start:h_end, w_start:w_end] += blend
        
        output = output / (weight + 1e-8)
        return output

    # _create_blend_mask is identical, so we can copy it
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


class AV1ConditionalUNet(nn.Module):
    """
    Complete AV1 artifact removal U-Net with FiLM conditioning.
    
    Architecture:
      - 5 levels: Head + 3 encoder levels + bottleneck + 3 decoder levels + tail
      - Skip connections at all encoder-decoder pairs
      - FiLM conditioning at: Encoder (3 levels) + Bottleneck (2 points)
      - SimpleSelfAttention at bottleneck only
      - Residual learning: output = input + predicted_residual
    
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
            'preset_range': (0, 8)
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
        cond_dim = 192  # Fixed: 128 (CRF) + 64 (Preset)
            
        logger.info("="*60)
        logger.info("Initializing AV1ConditionalUNet")
        logger.info(f"  Channels: {ch}")
        logger.info(f"  Blocks: {blocks}")
        logger.info(f"  CRF Range: {crf_range}")
        logger.info(f"  Preset Range: {preset_range}")
        logger.info(f"  Norm Range: [{self.clamp_min}, {self.clamp_max}]")
        logger.info("="*60)
        
        # ===== CONDITIONING SYSTEM =====
        self.conditioning_embedder = ConditioningEmbedder(
            crf_min=crf_range[0],
            crf_max=crf_range[1],
            preset_min=preset_range[0],
            preset_max=preset_range[1]
        )
        
        # ===== INPUT HEAD (Level 0) =====
        self.head = nn.Sequential(
            nn.Conv2d(3, ch[0], kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(ch[0]),
            nn.GELU(),
            *[EfficientResBlock(ch[0]) for _ in range(blocks[0])]
        )
        
        # ===== ENCODER =====
        self.encoder1 = self._build_encoder_level(ch[0], ch[1], blocks[1], cond_dim)
        self.encoder2 = self._build_encoder_level(ch[1], ch[2], blocks[2], cond_dim)
        self.encoder3 = self._build_encoder_level(ch[2], ch[3], blocks[3], cond_dim)
        
        # ===== BOTTLENECK =====
        self.bottleneck_down = nn.Conv2d(ch[3], ch[4], kernel_size=3, stride=2, padding=1, bias=False)
        self.bottleneck_bn = nn.BatchNorm2d(ch[4])
        
        # Pre-attention processing
        self.bottleneck_pre_attn = nn.Sequential(
            *[EfficientResBlock(ch[4]) for _ in range(blocks[4] // 2)]
        )
        self.bottleneck_film1 = FiLMLayer(cond_dim, ch[4])
        
        # Self-attention
        self.bottleneck_attn = SimpleSelfAttention(ch[4])
        
        # Post-attention processing
        self.bottleneck_film2 = FiLMLayer(cond_dim, ch[4])
        self.bottleneck_post_attn = nn.Sequential(
            *[EfficientResBlock(ch[4]) for _ in range(blocks[4] // 2)]
        )
        
        # ===== DECODER =====
        self.decoder3 = self._build_decoder_level(ch[4], ch[3], blocks[3])
        self.decoder2 = self._build_decoder_level(ch[3], ch[2], blocks[2])
        self.decoder1 = self._build_decoder_level(ch[2], ch[1], blocks[1])
        
        # ===== OUTPUT TAIL =====
        self.tail_up = nn.ConvTranspose2d(ch[1], ch[0], kernel_size=2, stride=2, bias=False)
        self.tail_bn = nn.BatchNorm2d(ch[0])
        self.tail_fusion = nn.Conv2d(ch[0] * 2, ch[0], kernel_size=1, bias=False)
        self.tail_body = nn.Sequential(
            *[EfficientResBlock(ch[0]) for _ in range(blocks[0])]
        )
        self.tail_pred = nn.Conv2d(ch[0], 3, kernel_size=3, padding=1)
        
        # Initialize weights
        self._init_weights()
        
        # Log parameter count
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logger.info(f"✓ Model initialized")
        logger.info(f"  Total parameters: {total_params:,} ({total_params/1e6:.2f}M)")
        logger.info(f"  Trainable parameters: {trainable_params:,} ({trainable_params/1e6:.2f}M)")
        logger.info("="*60)

    def _build_encoder_level(self, in_ch: int, out_ch: int, num_blocks: int, cond_dim: int) -> nn.ModuleDict:
        """Build one encoder level with downsampling, processing blocks, and FiLM."""
        return nn.ModuleDict({
            'downsample': nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.GELU()
            ),
            'body': nn.Sequential(
                *[EfficientResBlock(out_ch) for _ in range(num_blocks)]
            ),
            'film': FiLMLayer(cond_dim, out_ch)
        })

    def _build_decoder_level(self, in_ch: int, out_ch: int, num_blocks: int) -> nn.ModuleDict:
        """Build one decoder level with upsampling, skip fusion, and processing blocks."""
        return nn.ModuleDict({
            'upsample': nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2, bias=False),
            'upsample_bn': nn.BatchNorm2d(out_ch),
            'fusion': nn.Sequential(
                nn.Conv2d(out_ch * 2, out_ch, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.GELU()
            ),
            'body': nn.Sequential(
                *[EfficientResBlock(out_ch) for _ in range(num_blocks)]
            )
        })

    def _init_weights(self):
        """Initialize network weights for stable training."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                # CRITICAL FIX: Use smaller init for Linear layers to avoid NaN on MPS
                nn.init.xavier_uniform_(m.weight, gain=0.5)  # <-- Changed from kaiming
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        
        # Zero-initialize final prediction layer for residual learning
        nn.init.zeros_(self.tail_pred.weight)
        nn.init.zeros_(self.tail_pred.bias)
        logger.info("✓ Weights initialized (tail_pred zeroed for residual learning)")

    def forward(self, lq_image: torch.Tensor, crf: torch.Tensor, preset: torch.Tensor) -> torch.Tensor:
        """
        Full forward pass through the network.
        
        Args:
            lq_image: [B, 3, H, W] Low-quality compressed image, range [0, 1]
            crf: [B, 1] CRF value, range [23, 63]
            preset: [B, 1] Preset value, range [0, 8]
        
        Returns:
            [B, 3, H, W] Restored image, range [0, 1]
        """
        # ===== 1. GENERATE CONDITIONING VECTOR =====
        cond = self.conditioning_embedder(crf, preset)  # [B, 192]
        
        # ===== 2. ENCODER PATH =====
        # Head (Level 0)
        skip0 = self.head(lq_image)  # [B, ch[0], H, W]
        
        # Encoder Level 1
        e1 = self.encoder1['downsample'](skip0)  # [B, ch[1], H/2, W/2]
        e1 = self.encoder1['body'](e1)
        e1 = self.encoder1['film'](e1, cond)  # FiLM conditioning
        skip1 = e1
        
        # Encoder Level 2
        e2 = self.encoder2['downsample'](skip1)  # [B, ch[2], H/4, W/4]
        e2 = self.encoder2['body'](e2)
        e2 = self.encoder2['film'](e2, cond)
        skip2 = e2
        
        # Encoder Level 3
        e3 = self.encoder3['downsample'](skip2)  # [B, ch[3], H/8, W/8]
        e3 = self.encoder3['body'](e3)
        e3 = self.encoder3['film'](e3, cond)
        skip3 = e3
        
        # ===== 3. BOTTLENECK =====
        b = self.bottleneck_down(skip3)  # [B, ch[4], H/16, W/16]
        b = F.gelu(self.bottleneck_bn(b))
        
        # Pre-attention
        b = self.bottleneck_pre_attn(b)
        b = self.bottleneck_film1(b, cond)  # FiLM #1
        
        # Self-attention (global context)
        b = self.bottleneck_attn(b)
        
        # Post-attention
        b = self.bottleneck_film2(b, cond)  # FiLM #2
        b = self.bottleneck_post_attn(b)
        
        # ===== 4. DECODER PATH =====
        # Decoder Level 3
        d3 = self.decoder3['upsample'](b)  # [B, ch[3], H/8, W/8]
        d3 = F.gelu(self.decoder3['upsample_bn'](d3))
        d3 = torch.cat([d3, skip3], dim=1)  # Skip connection
        d3 = self.decoder3['fusion'](d3)
        d3 = self.decoder3['body'](d3)
        
        # Decoder Level 2
        d2 = self.decoder2['upsample'](d3)  # [B, ch[2], H/4, W/4]
        d2 = F.gelu(self.decoder2['upsample_bn'](d2))
        d2 = torch.cat([d2, skip2], dim=1)
        d2 = self.decoder2['fusion'](d2)
        d2 = self.decoder2['body'](d2)
        
        # Decoder Level 1
        d1 = self.decoder1['upsample'](d2)  # [B, ch[1], H/2, W/2]
        d1 = F.gelu(self.decoder1['upsample_bn'](d1))
        d1 = torch.cat([d1, skip1], dim=1)
        d1 = self.decoder1['fusion'](d1)
        d1 = self.decoder1['body'](d1)
        
        # ===== 5. OUTPUT TAIL =====
        t = self.tail_up(d1)  # [B, ch[0], H, W]
        t = F.gelu(self.tail_bn(t))
        t = torch.cat([t, skip0], dim=1)  # Skip connection
        t = self.tail_fusion(t)
        t = self.tail_body(t)
        residual = self.tail_pred(t)  # [B, 3, H, W] - predicted artifact correction
        
        # ===== 6. RESIDUAL LEARNING =====
        restored = lq_image + residual
        
        # Optional: Clamp to valid range [0, 1]
        restored = torch.clamp(restored, self.clamp_min, self.clamp_max)
        
        return restored

    @torch.no_grad()
    def inference(
        self,
        lq_image: torch.Tensor,
        crf: torch.Tensor,
        preset: torch.Tensor,
        tile_size: int = 512,
        tile_overlap: int = 64,
        center_crop: bool = False
    ) -> torch.Tensor:
        """
        Memory-efficient tiled inference for images of any size with seamless blending.
        
        This method handles three scenarios:
        1. Center crop mode: Extract and process only the center tile
        2. Small images (≤ tile_size): Process entire image with padding
        3. Large images (> tile_size): Tiled processing with smooth blending
        
        Args:
            lq_image: [B, 3, H, W] Input image (any size)
            crf: [B, 1] CRF value for conditioning
            preset: [B, 1] Preset value for conditioning
            tile_size: Maximum tile dimensions for processing (default: 512)
            tile_overlap: Overlap pixels between tiles for blending (default: 64)
            center_crop: If True, only process center tile_size×tile_size region
        
        Returns:
            [B, 3, H, W] Restored image (full size or center crop)
        
        Notes:
            - All images are padded to multiples of 32 (2^5 for 5 downsample layers)
            - Tiles use Gaussian-like edge blending for seamless stitching
            - Center crop is useful for quick quality checks on large images
        """
        self.eval()
        B, C, H, W = lq_image.shape
        
        # ===== MODE 1: CENTER CROP ONLY =====
        if center_crop:
            # Extract center tile
            crop_h = min(tile_size, H)
            crop_w = min(tile_size, W)
            
            start_h = (H - crop_h) // 2
            start_w = (W - crop_w) // 2
            
            center_tile = lq_image[:, :, start_h:start_h+crop_h, start_w:start_w+crop_w]
            
            # Pad to multiple of 32
            pad_h = (32 - crop_h % 32) % 32
            pad_w = (32 - crop_w % 32) % 32
            
            if pad_h > 0 or pad_w > 0:
                center_tile = F.pad(center_tile, (0, pad_w, 0, pad_h), mode='reflect')
                output = self.forward(center_tile, crf, preset)
                # Crop back to original tile size
                output = output[:, :, :crop_h, :crop_w]
            else:
                output = self.forward(center_tile, crf, preset)
            
            return output
        
        # ===== MODE 2: SMALL IMAGE (FITS IN SINGLE TILE) =====
        if H <= tile_size and W <= tile_size:
            # Pad to multiple of 32 for U-Net compatibility
            pad_h = (32 - H % 32) % 32
            pad_w = (32 - W % 32) % 32
            
            if pad_h > 0 or pad_w > 0:
                lq_padded = F.pad(lq_image, (0, pad_w, 0, pad_h), mode='reflect')
                output = self.forward(lq_padded, crf, preset)
                # Crop back to original size
                return output[:, :, :H, :W]
            else:
                return self.forward(lq_image, crf, preset)
        
        # ===== MODE 3: LARGE IMAGE (REQUIRES TILING) =====
        logger.info(f"Tiled inference: {H}×{W} image → {tile_size}×{tile_size} tiles (overlap: {tile_overlap}px)")
        
        # Initialize output accumulators
        output = torch.zeros_like(lq_image)
        weight = torch.zeros_like(lq_image)
        
        # Calculate grid parameters
        stride = tile_size - tile_overlap
        h_tiles = (H + stride - 1) // stride
        w_tiles = (W + stride - 1) // stride
        
        # Process each tile
        for i in range(h_tiles):
            for j in range(w_tiles):
                # Calculate tile boundaries
                h_start = i * stride
                w_start = j * stride
                h_end = min(h_start + tile_size, H)
                w_end = min(w_start + tile_size, W)
                
                # Extract tile
                tile = lq_image[:, :, h_start:h_end, w_start:w_end]
                tile_h, tile_w = tile.shape[2:]
                
                # Pad tile to multiple of 32
                pad_h = (32 - tile_h % 32) % 32
                pad_w = (32 - tile_w % 32) % 32
                
                if pad_h > 0 or pad_w > 0:
                    tile_padded = F.pad(tile, (0, pad_w, 0, pad_h), mode='reflect')
                    restored_tile = self.forward(tile_padded, crf, preset)
                    # Crop back to original tile size
                    restored_tile = restored_tile[:, :, :tile_h, :tile_w]
                else:
                    restored_tile = self.forward(tile, crf, preset)
                
                # Create smooth blending weights
                blend = self._create_blend_mask(
                    tile_h, tile_w, tile_overlap,
                    is_top=(i == 0),
                    is_bottom=(h_end >= H),
                    is_left=(j == 0),
                    is_right=(w_end >= W),
                    device=lq_image.device
                )
                
                # Accumulate weighted tiles
                output[:, :, h_start:h_end, w_start:w_end] += restored_tile * blend
                weight[:, :, h_start:h_end, w_start:w_end] += blend
        
        # Normalize by total weight
        output = output / (weight + 1e-8)
        
        return output

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
        """
        Create a smooth blending mask for seamless tile stitching.
        
        Uses linear gradients at tile edges to avoid visible seams.
        Edge tiles (touching image boundaries) don't fade on those edges.
        
        Args:
            tile_h, tile_w: Tile dimensions
            overlap: Overlap width in pixels
            is_top, is_bottom, is_left, is_right: Edge flags
            device: Tensor device
        
        Returns:
            [1, 1, tile_h, tile_w] blending weight mask
        """
        # Start with uniform weights
        blend = torch.ones(1, 1, tile_h, tile_w, device=device)
        
        # Determine fade width (half of overlap, capped at 32px)
        fade = min(overlap // 2, 32)
        
        # Apply linear fade at each edge (skip if at image boundary)
        # Top edge
        if not is_top:
            fade_h = min(fade, tile_h)
            ramp = torch.linspace(0, 1, fade_h, device=device).view(-1, 1)
            blend[:, :, :fade_h, :] *= ramp
        
        # Bottom edge
        if not is_bottom:
            fade_h = min(fade, tile_h)
            ramp = torch.linspace(1, 0, fade_h, device=device).view(-1, 1)
            blend[:, :, -fade_h:, :] *= ramp
        
        # Left edge
        if not is_left:
            fade_w = min(fade, tile_w)
            ramp = torch.linspace(0, 1, fade_w, device=device).view(1, -1)
            blend[:, :, :, :fade_w] *= ramp
        
        # Right edge
        if not is_right:
            fade_w = min(fade, tile_w)
            ramp = torch.linspace(1, 0, fade_w, device=device).view(1, -1)
            blend[:, :, :, -fade_w:] *= ramp
        
        return blend

# ==============================================================================
# SECTION 4: Unified Model Factory
# ==============================================================================

def create_av1_restorer(
    size: str = 'base',
    crf_range: Tuple[int, int] = (23, 63),
    preset_range: Tuple[int, int] = (0, 8),
    norm_range: Tuple[int, int] = (-1, 1)
) -> Union[AV1ConditionalUNet, AV1ConditionalUNetCRF]: # <-- Updated return type hint
    """
    Unified factory function for Conditional AV1 U-Net Restorers.

    Automatically creates either:
        - AV1ConditionalUNet (CRF+Preset): If preset_range has different min/max.
        - AV1ConditionalUNetCRF: If preset_range has the same min/max.

    Args:
        size: Model size variant ('tiny', 'small', 'base', 'large')
        crf_range: (min, max) for CRF normalization
        preset_range: (min, max) for Preset normalization (determines model type)
        norm_range: (min, max) for output clamping

    Returns:
        Initialized AV1ConditionalUNet or AV1ConditionalUNetCRF model.
    """

    configs = {
        'tiny': { # ~4M
            'channels': [24, 48, 64, 128, 224],
            'blocks': [2, 2, 2, 2, 4]
        },
        'small': { # ~10M
            'channels': [20, 40, 80, 160, 320],
            'blocks': [2, 2, 3, 3, 6]
        },
        'base': { # ~16M
            'channels': [48, 64, 128, 224, 384],
            'blocks': [2, 2, 3, 3, 6]
        },
        'large': { # ~32M
            'channels': [64, 96, 128, 256, 512],
            'blocks': [2, 3, 4, 4, 8]
        }
    }

    if size not in configs:
        raise ValueError(f"Unknown size '{size}'. Choose from: {list(configs.keys())}")

    # --- Prepare base config ---
    config = configs[size].copy() # Use .copy() to avoid modifying the template
    config['crf_range'] = crf_range
    config['norm_range'] = norm_range
    # preset_range is handled below

    # --- Logic to select model type ---
    if preset_range[0] == preset_range[1]:
        # CRF-Only Mode
        logger.info(
            f"Preset range is single value ({preset_range[0]}). "
            f"Creating AV1ConditionalUNetCRF (size={size})"
        )
        # Note: preset_range is NOT added to config for CRFOnly constructor
        return AV1ConditionalUNetCRF(config)
    else:
        # Standard CRF + Preset Mode
        logger.info(
            f"Preset range is variable {preset_range}. "
            f"Creating AV1ConditionalUNet (size={size})"
        )
        config['preset_range'] = preset_range # Add preset_range for standard constructor
        return AV1ConditionalUNet(config)

# ==============================================================================
# SECTION 5: Example Usage & Testing
# ==============================================================================

if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)
    model_size = 'tiny' # 'small', 'base', 'large'

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
    print("\n--- Parameter Count Check (using CRF-Only 'tiny' model) ---")
    total_params = sum(p.numel() for p in model_crf_only.parameters())
    trainable_params = sum(p.numel() for p in model_crf_only.parameters() if p.requires_grad)
    print(f"  Total parameters: {total_params:,} ({total_params/1e6:.2f}M)")
    print(f"  Trainable parameters: {trainable_params:,} ({trainable_params/1e6:.2f}M)")
