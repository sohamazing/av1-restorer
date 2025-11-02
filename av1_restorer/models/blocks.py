# av1_restorer/models/blocks.py
"""
Ultimate Shared Building Blocks for AV1 Restorers (v3.0 - SOTA Complete)

Consolidated module containing all building blocks including:
- Standard efficient blocks (GroupNorm + ECA + GELU)
- Conditioning modules (FiLM)
- Swin Transformer components (global receptive field)
- Wavelet Transform blocks (frequency-aware processing)
- Dense Feature Fusion (advanced skip connections)
- CBAM attention modules
- Legacy blocks (RDB, RRDB, etc.)

Author: Soham Mukherjee
Version: 3.0 (SOTA Complete)
License: MIT
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, Optional, List
import logging

logger = logging.getLogger(__name__)

# ==============================================================================
# SECTION 1: Core Helper Functions
# ==============================================================================

def choose_num_groups(channels: int, preferred: int = 16) -> int:
    """
    Helper to select a stable number of groups for GroupNorm.
    Falls back to 1 (LayerNorm-like) if channel count is not divisible.
    
    Args:
        channels (int): The number of channels to normalize.
        preferred (int, optional): The ideal number of groups. Defaults to 16.

    Returns:
        int: A valid number of groups (a divisor of channels).
    """
    if channels == 0: return 1
    # Check preferred, then common divisors
    for g in [preferred, 8, 4, 2, 1]:
        if channels % g == 0:
            return g
    return 1 # Fallback


def window_partition(x: torch.Tensor, window_size: int) -> torch.Tensor:
    """
    Partitions a tensor into non-overlapping windows (for Swin Transformer).

    Args:
        x (torch.Tensor): Input tensor of shape (B, H, W, C).
        window_size (int): The height and width of the window.

    Returns:
        torch.Tensor: Windows tensor of shape (B*num_windows, window_size, window_size, C).
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows: torch.Tensor, window_size: int, H: int, W: int) -> torch.Tensor:
    """
    Reverses the window partitioning (for Swin Transformer).

    Args:
        windows (torch.Tensor): Windows tensor of shape (B*num_windows, window_size, window_size, C).
        window_size (int): The height and width of the window.
        H (int): Original image height.
        W (int): Original image width.

    Returns:
        torch.Tensor: Reconstructed tensor of shape (B, H, W, C).
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


# ==============================================================================
# SECTION 2: Conditioning Modules (FiLM)
# ==============================================================================

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
# SECTION 3: Standard Efficient Blocks (GroupNorm + ECA)
# ==============================================================================

class DepthwiseSeparable(nn.Module):
    """
    Standardized Depthwise Separable Conv.
    Uses GroupNorm (for small-batch stability) + GELU (smooth activation).
    This is the core efficient convolution unit.
    
    Flow: Depthwise(3x3) -> Pointwise(1x1) -> GroupNorm -> GELU

    Args:
        in_ch (int): Input channels.
        out_ch (int): Output channels.
        stride (int, optional): Stride for depthwise conv. Defaults to 1.
        num_groups (int, optional): Preferred groups for GroupNorm. Defaults to 16.
    """
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1, num_groups: int = 16):
        super().__init__()
        self.depthwise = nn.Conv2d(in_ch, in_ch, 3, stride, 1, padding_mode='reflect', groups=in_ch, bias=False)
        self.pointwise = nn.Conv2d(in_ch, out_ch, 1, bias=False)
        self.norm = nn.GroupNorm(choose_num_groups(out_ch, num_groups), out_ch)
        self.act = nn.GELU()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.norm(self.pointwise(self.depthwise(x))))


class ECA(nn.Module):
    """
    Standardized Efficient Channel Attention (ECA-Net).
    Learns channel weights using an efficient 1D convolution after
    global average pooling.
    
    Args:
        channels (int): Input channels.
        kernel_size (int, optional): Kernel size for the 1D conv. Defaults to 3.
    """
    def __init__(self, channels: int, kernel_size: int = 3):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size, padding=(kernel_size - 1) // 2, bias=False)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.shape
        # (B, C, 1, 1) -> (B, 1, C)
        y = self.avg_pool(x).view(b, 1, c)
        # (B, 1, C) -> (B, 1, C)
        y = self.conv(y)
        # (B, 1, C) -> (B, C, 1, 1)
        y = y.view(b, c, 1, 1)
        # Apply sigmoid and scale input
        return x * y.sigmoid()


class MicroResBlock(nn.Module):
    """
    Standardized Efficient residual block (non-inverted).
    Processes features at their original channel depth.
    
    Flow: x -> DWS -> DWS -> ECA -> x + out
    (Used by Nano U-Net and FBCNN)

    Args:
        channels (int): Input/output channels.
        num_groups (int, optional): Preferred groups for GroupNorm. Defaults to 16.
        eca_kernel (int, optional): Kernel size for ECA. Defaults to 3.
    """
    def __init__(self, channels: int, num_groups: int = 16, eca_kernel: int = 3):
        super().__init__()
        self.conv1 = DepthwiseSeparable(channels, channels, stride=1, num_groups=num_groups)
        self.conv2 = DepthwiseSeparable(channels, channels, stride=1, num_groups=num_groups)
        self.attn = ECA(channels, kernel_size=eca_kernel)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.attn(out)
        return identity + out


class EfficientResBlock(nn.Module):
    """
    MobileNetV3-Inspired Inverted Residual Block with ECA Attention.
    (Used by Nano-ResNet and Conditional U-Net)
    
    Combines three proven techniques:
    1. Inverted Residual (expand → process → project)
    2. Depthwise Separable Convolutions (efficiency)
    3. ECA Attention (channel recalibration)
    
    Args:
         channels: Input/output channel count (must match for residual)
         expansion: Hidden dimension multiplier (2-4 typical)
         num_groups: GroupNorm groups (auto-adjusted by helper)
    
    Parameters: ~4C² vs Standard ResBlock: ~9C²
    """
    def __init__(self, channels: int, expansion: int = 2, num_groups: int = 16):
        super().__init__()
        hidden = channels * expansion
        
        # Expansion: channels → hidden
        self.conv1 = nn.Sequential(
            nn.Conv2d(channels, hidden, 1, bias=False),
            nn.GroupNorm(choose_num_groups(hidden, num_groups), hidden),
            nn.GELU()
        )
        
        # Depthwise spatial processing
        self.dwconv = DepthwiseSeparable(hidden, hidden, stride=1, num_groups=num_groups)

        # Channel attention
        self.attn = ECA(hidden)
        
        # Projection: hidden → channels
        self.conv2 = nn.Sequential(
            nn.Conv2d(hidden, channels, 1, bias=False),
            nn.GroupNorm(choose_num_groups(channels, num_groups), channels)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.conv1(x)      # Expand
        out = self.dwconv(out)   # Process spatially
        out = self.attn(out)     # Recalibrate channels
        out = self.conv2(out)    # Project back
        return identity + out    # Residual connection


class SimpleSelfAttention(nn.Module):
    """
    Efficient channel-wise self-attention for bottleneck.
    (Used by Conditional U-Net V2)
    
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


class MultiScaleFusion(nn.Module):
    """
    Standardized Multi-scale feature fusion for U-Net skip connections.
    Concatenates features from the upsampling path and the skip connection,
    then fuses them with a 1x1 convolution.
    
    (Used by Nano U-Net and FBCNN)

    Args:
        channels (int): Channel count of the *skip connection* layer.
        num_groups (int, optional): Preferred groups for GroupNorm. Defaults to 16.
    """
    def __init__(self, channels: int, num_groups: int = 16):
        super().__init__()
        gn_groups = choose_num_groups(channels, num_groups)
        self.fusion = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size=1, bias=False),
            nn.GroupNorm(gn_groups, channels),
            nn.GELU()
        )

    def forward(self, x_up: torch.Tensor, x_skip: torch.Tensor) -> torch.Tensor:
        # x_up:   [B, C, H, W] (from decoder)
        # x_skip: [B, C, H, W] (from encoder)
        return self.fusion(torch.cat([x_up, x_skip], dim=1))


class MultiScaleFeatureExtractor(nn.Module):
    """
    Parallel Multi-Scale Feature Extraction Head.
    (Used by Nano-ResNet)
    
    Captures features at multiple receptive field sizes simultaneously.
    
    Three Parallel Paths:
         1. Fine:   3×3 kernel (local details, textures)
         2. Medium: 2× 3×3 (5×5 effective, edges, patterns)
         3. Coarse: 3× 3×3 (7×7 effective, structures, context)
    
    Args:
         in_ch: Input channels
         out_ch: Output channels
    """
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        split = out_ch // 3
        
        # Fine-scale path (1 layer, 3×3 receptive field)
        self.fine = DepthwiseSeparable(in_ch, split, 3)
        
        # Medium-scale path (2 layers, ~5×5 receptive field)
        self.medium = nn.Sequential(
            DepthwiseSeparable(in_ch, split, 3),
            DepthwiseSeparable(split, split, 3)
        )
        
        # Coarse-scale path (3 layers, ~7×7 receptive field)
        coarse_ch = out_ch - 2 * split 
        self.coarse = nn.Sequential(
            DepthwiseSeparable(in_ch, coarse_ch, 3),
            DepthwiseSeparable(coarse_ch, coarse_ch, 3),
            DepthwiseSeparable(coarse_ch, coarse_ch, 3)
        )
        
        # Fusion: combine all scales
        self.fusion = nn.Conv2d(out_ch, out_ch, 1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f1 = self.fine(x)      # Local details
        f2 = self.medium(x)    # Mid-range patterns
        f3 = self.coarse(x)    # Global context
        
        # Concatenate along channel dimension and fuse
        return self.fusion(torch.cat([f1, f2, f3], dim=1))


# ==============================================================================
# SECTION 4: Swin Transformer Components (Global Receptive Field)
# ==============================================================================

class WindowAttention(nn.Module):
    """
    Window-based multi-head self-attention with relative position bias.
    
    Core component of Swin Transformer for efficient global modeling.
    Complexity: O(M²) where M = window_size² << image_size²
    
    Args:
        dim: Feature dimension
        window_size: Window size for local attention
        num_heads: Number of attention heads
    """
    def __init__(self, dim: int, window_size: int = 8, num_heads: int = 8):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        
        # Relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size - 1) * (2 * window_size - 1), num_heads)
        )
        
        # QKV projection
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim)
        
        # Initialize relative position index
        coords_h = torch.arange(self.window_size)
        coords_w = torch.arange(self.window_size)
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing='ij'))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += self.window_size - 1
        relative_coords[:, :, 1] += self.window_size - 1
        relative_coords[:, :, 0] *= 2 * self.window_size - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)
        
        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: [B*num_windows, window_size*window_size, C]
            mask: Attention mask for shifted windows
        """
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))
        
        # Add relative position bias
        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)
        ].view(self.window_size * self.window_size, self.window_size * self.window_size, -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)
        
        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N)
            attn = attn + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
        
        attn = F.softmax(attn, dim=-1)
        
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        return x


class SwinTransformerBlock(nn.Module):
    """
    Swin Transformer Block with shifted window attention.
    
    Alternates between:
    1. Window attention (local modeling)
    2. Shifted window attention (cross-window communication)
    
    This provides global receptive field with linear complexity.
    
    Args:
        dim: Feature dimension
        num_heads: Number of attention heads
        window_size: Window size for local attention
        shift_size: Shift size for shifted window attention
        mlp_ratio: Ratio of mlp hidden dim to embedding dim
    """
    def __init__(
        self, 
        dim: int, 
        num_heads: int = 8, 
        window_size: int = 8,
        shift_size: int = 0,
        mlp_ratio: float = 4.0
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        
        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(dim, window_size, num_heads)
        
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Linear(mlp_hidden_dim, dim)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, C, H, W]
        Returns:
            [B, C, H, W]
        """
        B, C, H, W = x.shape
        
        # Permute to [B, H, W, C] for attention
        x = x.permute(0, 2, 3, 1).contiguous()
        
        shortcut = x
        x = self.norm1(x)
        
        # Cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x
        
        # Partition windows
        x_windows = window_partition(shifted_x, self.window_size)
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)
        
        # Window attention
        attn_windows = self.attn(x_windows)
        
        # Merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)
        
        # Reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
        
        # FFN
        x = shortcut + x
        x = x + self.mlp(self.norm2(x))
        
        # Permute back to [B, C, H, W]
        x = x.permute(0, 3, 1, 2).contiguous()
        return x


class SwinBottleneck(nn.Module):
    """
    Stack of Swin Transformer blocks for bottleneck processing.
    
    Provides true global receptive field with linear complexity.
    This is a drop-in replacement for SimpleSelfAttention.
    
    Args:
        channels: Feature dimension
        depth: Number of Swin blocks
        num_heads: Number of attention heads
        window_size: Window size for local attention
    """
    def __init__(self, channels: int, depth: int = 6, num_heads: int = 8, window_size: int = 8):
        super().__init__()
        
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(
                dim=channels,
                num_heads=num_heads,
                window_size=window_size,
                shift_size=0 if (i % 2 == 0) else window_size // 2,
                mlp_ratio=4.0
            )
            for i in range(depth)
        ])
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, C, H, W]
        Returns:
            [B, C, H, W]
        """
        for block in self.blocks:
            x = block(x)
        return x


# ==============================================================================
# SECTION 5: Wavelet Transform Components (Frequency-Aware Processing)
# ==============================================================================

class LearnableWaveletTransform(nn.Module):
    """
    Learnable Discrete Wavelet Transform (DWT).
    
    Decomposes input into 4 subbands:
    - LL (low-low): Approximation/smooth content
    - LH (low-high): Horizontal edges
    - HL (high-low): Vertical edges  
    - HH (high-high): Diagonal edges/textures
    
    Unlike fixed Haar wavelets, filters are learned during training.
    
    Args:
        channels: Number of input/output channels
    """
    def __init__(self, channels: int):
        super().__init__()
        
        # Learnable analysis filters (decomposition)
        self.conv_ll = nn.Conv2d(channels, channels, 3, stride=2, padding=1, groups=channels)
        self.conv_lh = nn.Conv2d(channels, channels, 3, stride=2, padding=1, groups=channels)
        self.conv_hl = nn.Conv2d(channels, channels, 3, stride=2, padding=1, groups=channels)
        self.conv_hh = nn.Conv2d(channels, channels, 3, stride=2, padding=1, groups=channels)
        
        # Initialize with Haar-like patterns
        self._init_wavelet_filters()
    
    def _init_wavelet_filters(self):
        """Initialize filters with Haar wavelet patterns (will be refined during training)."""
        with torch.no_grad():
            # LL: Low-pass (averaging)
            self.conv_ll.weight.fill_(0.25)
            
            # LH: Horizontal high-pass
            lh_pattern = torch.tensor([[-1, -1, -1], [0, 0, 0], [1, 1, 1]]) / 6.0
            for i in range(self.conv_lh.weight.size(0)):
                self.conv_lh.weight[i, 0] = lh_pattern
            
            # HL: Vertical high-pass  
            hl_pattern = torch.tensor([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]]) / 6.0
            for i in range(self.conv_hl.weight.size(0)):
                self.conv_hl.weight[i, 0] = hl_pattern
            
            # HH: Diagonal high-pass
            hh_pattern = torch.tensor([[-1, 0, 1], [0, 0, 0], [1, 0, -1]]) / 4.0
            for i in range(self.conv_hh.weight.size(0)):
                self.conv_hh.weight[i, 0] = hh_pattern
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Decompose input into 4 wavelet subbands.
        
        Args:
            x: [B, C, H, W]
        
        Returns:
            (ll, lh, hl, hh): Each is [B, C, H/2, W/2]
        """
        ll = self.conv_ll(x)  # Approximation
        lh = self.conv_lh(x)  # Horizontal details
        hl = self.conv_hl(x)  # Vertical details
        hh = self.conv_hh(x)  # Diagonal details
        
        return ll, lh, hl, hh


class WaveletRestorationBlock(nn.Module):
    """
    Process each wavelet subband with specialized networks.
    
    Key Insight: Different frequency bands need different processing:
    - LL: Smooth, context-aware (large receptive field)
    - LH/HL: Edge-preserving (directional filters)
    - HH: Texture reconstruction (fine-grained)
    
    This provides frequency-adaptive restoration.
    
    Args:
        channels: Number of input/output channels
    """
    def __init__(self, channels: int):
        super().__init__()
        
        # Decomposition
        self.dwt = LearnableWaveletTransform(channels)
        
        # Subband-specific processors
        self.process_ll = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.GroupNorm(choose_num_groups(channels), channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, 3, padding=1)
        )
        
        self.process_lh = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.GroupNorm(choose_num_groups(channels), channels),
            nn.GELU()
        )
        
        self.process_hl = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.GroupNorm(choose_num_groups(channels), channels),
            nn.GELU()
        )
        
        self.process_hh = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.GroupNorm(choose_num_groups(channels), channels),
            nn.GELU()
        )
        
        # Reconstruction (inverse wavelet)
        self.reconstruct = nn.Sequential(
            nn.ConvTranspose2d(channels * 4, channels, 3, stride=2, padding=1, output_padding=1),
            nn.GroupNorm(choose_num_groups(channels), channels),
            nn.GELU()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Wavelet-based processing with frequency-adaptive restoration.
        
        Args:
            x: [B, C, H, W]
        
        Returns:
            [B, C, H, W] - Restored features
        """
        identity = x
        
        # Decompose into subbands
        ll, lh, hl, hh = self.dwt(x)
        
        # Process each subband independently
        ll_proc = self.process_ll(ll)
        lh_proc = self.process_lh(lh)
        hl_proc = self.process_hl(hl)
        hh_proc = self.process_hh(hh)
        
        # Concatenate and reconstruct
        merged = torch.cat([ll_proc, lh_proc, hl_proc, hh_proc], dim=1)
        reconstructed = self.reconstruct(merged)
        
        # Residual connection
        return identity + reconstructed


# ==============================================================================
# SECTION 6: Advanced Fusion / Attention (CBAM, DFF)
# ==============================================================================

class ChannelAttention(nn.Module):
    """
    Squeeze-and-Excitation style channel attention.
    
    Uses both average and max pooling for richer global context.
    
    Args:
        channels: Number of input channels
        reduction: Channel reduction ratio
    """
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.fc = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        return self.sigmoid(avg_out + max_out)


class SpatialAttention(nn.Module):
    """
    Spatial attention to focus on important regions.
    
    Args:
        kernel_size: Convolution kernel size
    """
    def __init__(self, kernel_size: int = 7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        cat = torch.cat([avg_out, max_out], dim=1)
        return self.sigmoid(self.conv(cat))


class CBAM(nn.Module):
    """Convolutional Block Attention Module (CBAM)."""
    def __init__(self, channels: int):
        super().__init__()
        self.channel_attn = ChannelAttention(channels)
        self.spatial_attn = SpatialAttention()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x * self.channel_attn(x)
        x = x * self.spatial_attn(x)
        return x


class DenseFeatureFusion(nn.Module):
    """
    Dense skip connection with multi-scale feature refinement.
    
    Key improvements over simple concatenation:
    1. Multi-scale pyramid pooling
    2. Attention-guided feature selection
    3. Progressive feature refinement
    4. Residual connections
    
    Input: encoder_feat (skip) + decoder_feat (upsampled)
    Output: Refined fused features
    """
    def __init__(self, channels: int):
        super().__init__()

        # --- BUG FIX: Handle channel counts not divisible by 4 ---
        # Old: c_split = channels // 4 (fails for 30, 7*4=28)
        # New: 3 splits of C//4, 1 split with remainder
        c_split = channels // 4
        c_remainder = channels - (3 * c_split)
        
        # Multi-scale context extraction
        self.pyramid_1 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            # nn.Conv2d(channels, channels // 4, 1),
            nn.Conv2d(channels, c_split, 1),
            nn.ReLU(inplace=True)
        )
        self.pyramid_2 = nn.Sequential(
            nn.AdaptiveAvgPool2d(2),
            # nn.Conv2d(channels, channels // 4, 1),
            nn.Conv2d(channels, c_split, 1),
            nn.ReLU(inplace=True)
        )
        self.pyramid_4 = nn.Sequential(
            nn.AdaptiveAvgPool2d(4),
            # nn.Conv2d(channels, channels // 4, 1),
            nn.Conv2d(channels, c_split, 1),
            nn.ReLU(inplace=True)
        )
        self.pyramid_8 = nn.Sequential(
            # nn.Conv2d(channels, channels // 4, 1),
            nn.Conv2d(channels, c_remainder, 1), # Use remainder here
            nn.ReLU(inplace=True)
        )
        
        # Pyramid fusion
        self.pyramid_fusion = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.GroupNorm(choose_num_groups(channels), channels),
            nn.ReLU(inplace=True)
        )
        
        # Feature refinement for encoder features
        self.encoder_refine = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.GroupNorm(choose_num_groups(channels), channels),
            nn.ReLU(inplace=True),
            CBAM(channels)  # Attention-based refinement
        )
        
        # Feature refinement for decoder features
        self.decoder_refine = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.GroupNorm(choose_num_groups(channels), channels),
            nn.ReLU(inplace=True),
            CBAM(channels)
        )
        
        # Final fusion
        self.fusion = nn.Sequential(
            nn.Conv2d(channels * 3, channels, 1, bias=False),  # 3 = encoder + decoder + pyramid
            nn.GroupNorm(choose_num_groups(channels), channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.GroupNorm(choose_num_groups(channels), channels)
        )
        
        # Residual scaling
        self.alpha = nn.Parameter(torch.ones(1) * 0.1)
    
    def forward(self, encoder_feat: torch.Tensor, decoder_feat: torch.Tensor) -> torch.Tensor:
        """
        Args:
            encoder_feat: [B, C, H, W] - Skip connection from encoder
            decoder_feat: [B, C, H, W] - Upsampled features from decoder
        
        Returns:
            [B, C, H, W] - Refined fused features
        """
        B, C, H, W = encoder_feat.shape
        
        # 1. Multi-scale pyramid context
        p1 = F.interpolate(self.pyramid_1(decoder_feat), size=(H, W), mode='bilinear', align_corners=False)
        p2 = F.interpolate(self.pyramid_2(decoder_feat), size=(H, W), mode='bilinear', align_corners=False)
        p4 = F.interpolate(self.pyramid_4(decoder_feat), size=(H, W), mode='bilinear', align_corners=False)
        p8 = self.pyramid_8(decoder_feat)
        
        pyramid_feat = self.pyramid_fusion(torch.cat([p1, p2, p4, p8], dim=1))
        
        # 2. Refine encoder features (remove potential artifacts)
        encoder_refined = self.encoder_refine(encoder_feat)
        
        # 3. Refine decoder features
        decoder_refined = self.decoder_refine(decoder_feat)
        
        # 4. Concatenate and fuse
        fused = self.fusion(torch.cat([encoder_refined, decoder_refined, pyramid_feat], dim=1))
        
        # 5. Residual connection with learnable scaling
        output = decoder_feat + self.alpha * fused
        
        return output


class ProgressiveFeatureFusion(nn.Module):
    """
    Progressive fusion across multiple decoder levels.
    
    Enables information flow from deeper to shallower levels.
    """
    def __init__(self, channels_list: list):
        """
        Args:
            channels_list: List of channel counts at each decoder level
                          e.g., [384, 192, 96, 48] for 4-level decoder
        """
        super().__init__()
        
        self.fusions = nn.ModuleList([
            DenseFeatureFusion(ch) for ch in channels_list
        ])
        
        # Cross-level communication (bottom-up pathway)
        self.lateral = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(channels_list[i], channels_list[i+1], 1, bias=False),
                nn.GroupNorm(choose_num_groups(channels_list[i+1]), channels_list[i+1])
            )
            for i in range(len(channels_list) - 1)
        ])
    
    def forward(self, encoder_feats: list, decoder_feats: list) -> list:
        """
        Args:
            encoder_feats: List of encoder features (deep to shallow)
            decoder_feats: List of decoder features (deep to shallow)
        
        Returns:
            List of refined decoder features
        """
        refined_feats = []
        
        # Process from deepest to shallowest
        for i, (enc_f, dec_f) in enumerate(zip(encoder_feats, decoder_feats)):
            # Fuse at current level
            fused = self.fusions[i](enc_f, dec_f)
            
            # Add bottom-up information from previous level
            if i > 0 and i < len(self.lateral):
                bottom_up = F.interpolate(
                    refined_feats[-1], 
                    size=fused.shape[-2:], 
                    mode='bilinear', 
                    align_corners=False
                )
                bottom_up = self.lateral[i-1](bottom_up)
                fused = fused + bottom_up
            
            refined_feats.append(fused)
        
        return refined_feats


# ==============================================================================
# SECTION 7: Legacy / Alternative Blocks (ESRGAN, SwinIR)
# ==============================================================================

class ConvResBlock(nn.Module):
    """
    A standard residual block with Conv -> BatchNorm -> GELU.
    Uses standard (heavy) 3x3 convolutions and BatchNorm.
    (Used by CNNFeatureExtractor for SwinIR)

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        stride (int, optional): Stride for the first convolution. Defaults to 1.
    """
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1)
        self.norm1 = nn.BatchNorm2d(out_channels)
        self.gelu = nn.GELU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1)
        self.norm2 = nn.BatchNorm2d(out_channels)

        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        shortcut = self.shortcut(x)
        x = self.gelu(self.norm1(self.conv1(x)))
        x = self.norm2(self.conv2(x))
        return self.gelu(x + shortcut)


class CNNFeatureExtractor(nn.Module):
    """
    Lightweight CNN feature extractor
    Consists of a stack of ConvResBlocks.

    Args:
        in_channels (int, optional): Input image channels. Defaults to 3.
        embed_dim (int, optional): Embedding dimension. Defaults to 180.
        num_layers (int, optional): Number of ConvResBlocks. Defaults to 4.

    Returns:
        List[torch.Tensor]: A list of feature maps from each layer.
    """
    def __init__(self, in_channels=3, embed_dim=180, num_layers=4):
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(ConvResBlock(in_channels, embed_dim, stride=1))
        for _ in range(num_layers - 1):
            self.layers.append(ConvResBlock(embed_dim, embed_dim, stride=1))

    def forward(self, x):
        feature_maps = []
        for layer in self.layers:
            x = layer(x)
            feature_maps.append(x)
        return feature_maps


class ResidualDenseBlock(nn.Module):
    """
    Residual Dense Block (RDB) used within an RRDB.
    Features from all preceding layers are concatenated and fed into
    the next layer, promoting rich feature reuse.

    Args:
        in_channels (int, optional): Input/output channels. Defaults to 64.
        growth_channels (int, optional): Channel growth rate. Defaults to 32.
    """
    def __init__(self, in_channels=64, growth_channels=32):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, growth_channels, 3, 1, 1)
        self.conv2 = nn.Conv2d(in_channels + growth_channels, growth_channels, 3, 1, 1)
        self.conv3 = nn.Conv2d(in_channels + 2 * growth_channels, growth_channels, 3, 1, 1)
        self.conv4 = nn.Conv2d(in_channels + 3 * growth_channels, growth_channels, 3, 1, 1)
        self.conv5 = nn.Conv2d(in_channels + 4 * growth_channels, in_channels, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        # Residual connection with scaling
        return x5 * 0.2 + x


class RRDB(nn.Module):
    """
    Residual in Residual Dense Block (RRDB).
    The main building block of ESRGAN-style networks. Stacks three RDBs.
    
    Args:
        in_channels (int, optional): Input/output channels. Defaults to 64.
    """
    def __init__(self, in_channels=64):
        super().__init__()
        self.rdb1 = ResidualDenseBlock(in_channels)
        self.rdb2 = ResidualDenseBlock(in_channels)
        self.rdb3 = ResidualDenseBlock(in_channels)

    def forward(self, x):
        out = self.rdb1(x)
        out = self.rdb2(out)
        out = self.rdb3(out)
        # Residual connection with scaling
        return out * 0.2 + x


class UpsampleBlock(nn.Module):
    """
    ESRGAN-style upsampling block using PixelShuffle.
    Flow: Conv(3x3) -> PixelShuffle(2x) -> LeakyReLU

    Args:
        in_channels (int): Number of input channels.
    """
    def __init__(self, in_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, in_channels * 4, 3, 1, 1)
        self.pixel_shuffle = nn.PixelShuffle(2)
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        return self.lrelu(self.pixel_shuffle(self.conv(x)))