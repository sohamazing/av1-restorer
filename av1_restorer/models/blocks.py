# av1_restorer/utils/blocks.py
"""
Ultimate Shared Building Blocks for AV1 Restorers (v3.0 - SOTA Standardized)

Consolidated module containing all unique components from all project models,
standardized on the most robust components:
- GroupNorm (for small-batch stability)
- ECA (for parameter efficiency)
- GELU (for smooth activation)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm
from typing import Tuple, Dict, Optional, List
from einops import rearrange
import logging

logger = logging.getLogger(__name__)

# ==============================================================================
# SECTION 1: Core Helper Function
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
    Partitions a tensor into non-overlapping windows (for SwinIR).

    Args:
        x (torch.Tensor): Input tensor of shape (B, H, W, C).
        window_size (int): The height and width of the window.

    Returns:
        torch.Tensor: Windows tensor of shape (B*num_windows, Ws, Ws, C).
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows

def window_reverse(windows: torch.Tensor, window_size: int, H: int, W: int) -> torch.Tensor:
    """
    Reverses the window partitioning (for SwinIR).

    Args:
        windows (torch.Tensor): Windows tensor of shape (B*num_windows, Ws, Ws, C).
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
# SECTION 2: SOTA Nano Blocks (Standardized on GroupNorm + ECA)
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
        self.depthwise = nn.Conv2d(in_ch, in_ch, 3, stride, 1, groups=in_ch, bias=False)
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
    
    Flow:
         Input (channels)
           ↓
         Expand (1×1 conv → channels × expansion_ratio)
           ↓
         Depthwise Conv (3×3, spatial filtering)
           ↓
         ECA Attention (channel recalibration)
           ↓
         Project (1×1 conv → channels)
           ↓
         Residual Add (input + processed)
    
    Design Rationale:
         - Expand first: creates rich feature space for processing
         - Depthwise: efficient spatial filtering in expanded space
         - Attention: emphasizes useful channels, suppresses noise
         - Project: compress back to original dimensions
         - Residual: gradient flow and feature reuse
    
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
        self.dwconv = DepthwiseSeparable(hidden, hidden, 3, num_groups=num_groups)
        
        # Channel attention
        self.attn = ECA(hidden)
        
        # Projection: hidden → channels
        self.conv2 = nn.Sequential(
            nn.Conv2d(hidden, channels, 1, bias=False),
            nn.GroupNorm(choose_num_groups(channels, num_groups), channels)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, channels, H, W]
        Returns:
            [B, channels, H, W]
        """
        identity = x
        out = self.conv1(x)      # Expand
        out = self.dwconv(out)   # Process spatially
        out = self.attn(out)     # Recalibrate channels
        out = self.conv2(out)    # Project back
        return identity + out    # Residual connection

class MultiScaleFusion(nn.Module):
    """
    Standardized Multi-scale feature fusion for U-Net skip connections.
    Concatenates features from the upsampling path and the skip connection,
    then fuses them with a 1x1 convolution.
    
    (Used by Nano U-Net and FBCNN)

    Args:
        channels (int): Channel count of the *skip connection* layer.
                        (Assumes upsampled layer has same channel count).
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
    
    Captures features at multiple receptive field sizes simultaneously,
    analogous to Inception modules but more efficient (using DWS convs).
    
    Three Parallel Paths:
         1. Fine:   3×3 kernel (local details, textures)
         2. Medium: 2× 3×3 (5×5 effective, edges, patterns)
         3. Coarse: 3× 3×3 (7×7 effective, structures, context)
    
    Output: Concatenate + 1×1 fusion → ensures all scales contribute
    
    This is particularly useful for artifact removal because:
         - Blocking artifacts: local (fine path)
         - Ringing: medium-range (medium path)
         - Color banding: large-scale (coarse path)
    
    Args:
         in_ch: Input channels (typically base_channels from head)
         out_ch: Output channels (typically same as in_ch)
    
    Channel Split:
         Fine:   out_ch // 3
         Medium: out_ch // 3
         Coarse: out_ch - 2*(out_ch//3)  # Takes remainder
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
        # Handle remainder channels
        coarse_ch = out_ch - 2 * split 
        self.coarse = nn.Sequential(
            DepthwiseSeparable(in_ch, coarse_ch, 3),
            DepthwiseSeparable(coarse_ch, coarse_ch, 3),
            DepthwiseSeparable(coarse_ch, coarse_ch, 3)
        )
        
        # Fusion: combine all scales
        self.fusion = nn.Conv2d(out_ch, out_ch, 1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, in_ch, H, W]
        Returns:
            [B, out_ch, H, W] with multi-scale features
        """
        f1 = self.fine(x)      # Local details
        f2 = self.medium(x)    # Mid-range patterns
        f3 = self.coarse(x)    # Global context
        
        # Concatenate along channel dimension and fuse
        return self.fusion(torch.cat([f1, f2, f3], dim=1))

# ==============================================================================
# SECTION 3: Blocks from Other Architectures
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
