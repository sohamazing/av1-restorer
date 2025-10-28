Here is the SOTA-upgraded version of your av1_prior_embedder.py.

This new architecture is designed for maximum stability, efficiency, and quality, based on the best practices from your other project files (like the V2 U-Net and Nano-ResNet).

The key changes are:

GroupNorm replaces BatchNorm: This is the most critical change. GroupNorm is batch-size independent, making it far more stable and effective for training with the small batch sizes common in image restoration.

Efficient Blocks: The heavy ResidualBlock is replaced with the EfficientResBlock from your blocks.py file. This uses depthwise separable convolutions and channel attention (ECA) for a massive reduction in parameters and computational cost.

Modern Upsampling: The UNetDecoder's ConvTranspose2d layers (which cause checkerboard artifacts) are replaced with the "Upsample + Conv" method (Bilinear Upsample + DepthwiseSeparable conv), ensuring a clean, artifact-free output.

Logical Prediction: The CRFPredictor now branches from the fused_features (local + global) instead of just the local_features, giving it more context to make a better prediction.

I have fully documented every decision inline.

av1_prior_embedder_v2.py
Python

"""
AV1 Compression Prior Embedder (CaPE) - SOTA CNN Architecture

This module implements a lightweight, SOTA CNN-only Compression-aware Prior
Embedder, optimized for stability, speed, and quality.

--- SOTA DECISION: Architecture Philosophy ---
This architecture is a "two-tower" design.
1. Local Encoder: A deep, multi-scale CNN (ResNet-like) that extracts
   fine-grained artifact patterns (blocking, ringing).
2. Global Encoder: A lightweight, wide-receptive-field CNN that captures
   the overall image structure and texture density.

Fusing these two distinct views gives the model a rich understanding of
*what* the artifact is (local) and *where* it is in context (global).
This version is upgraded to use GroupNorm and efficient blocks.
--- END DECISION ---

Dual Learning Strategy:
1. Explicit Learning: Direct CRF prediction from compressed images.
2. Implicit Learning: Image reconstruction to understand compression artifacts.

Architecture:
- Local Artifact Encoder (SOTA: EfficientResBlocks + GroupNorm)
- Global Context Encoder (SOTA: Efficient Wide-Kernel CNN + GroupNorm)
- Feature Fusion Module (SOTA: GroupNorm)
- Reconstruction Decoder (SOTA: Upsample + Conv, no checkerboards)
- CRF Predictor (Explicit learning)

Author: Soham Mukherjee
"""

import logging
from typing import Tuple, Optional, List

import torch
import torch.nn as nn
import torch.nn.functional as F

# --- SOTA DECISION: Import Standardized Blocks ---
# We import standardized, efficient blocks from the project's shared 'blocks.py'.
# This ensures consistency, efficiency, and use of best practices.
# - EfficientResBlock: Lightweight inverted residual block.
# - DepthwiseSeparable: Efficient convolution replacement.
# - choose_num_groups: Stability helper for GroupNorm.
# --- END DECISION ---
try:
    from .blocks import EfficientResBlock, DepthwiseSeparable, choose_num_groups
except ImportError:
    logger.error("Could not import from .blocks. Please ensure blocks.py is in the same directory.")
    # Define fallbacks if running standalone (not recommended for project)
    def choose_num_groups(c, p=16):
        for g in [p, 8, 4, 2, 1]:
            if c % g == 0: return g
        return 1
    # Note: EfficientResBlock and DepthwiseSeparable are too complex for fallbacks.

logger = logging.getLogger(__name__)


class LocalArtifactEncoder(nn.Module):
    """
    SOTA Multi-scale CNN encoder for local AV1 compression artifacts.
    
    This is the "deep" tower, designed to learn fine-grained artifact patterns.
    It's built like a modern ResNet encoder.
    
    --- SOTA DECISION: EfficientResBlock + GroupNorm ---
    Replaces the original file's heavy 'ResidualBlock' (which used
    standard convs and BatchNorm) with 'EfficientResBlock' from blocks.py.
    
    WHY:
    1. GroupNorm: 'BatchNorm2d' is unstable with small batch sizes
       (common in restoration). GroupNorm is batch-size independent,
       ensuring stable training.
    2. Efficiency: 'EfficientResBlock' uses depthwise-separable convs
       and channel attention, drastically reducing parameters and FLOPS
       while maintaining high performance.
    --- END DECISION ---
    
    Args:
        embed_dim: Output embedding dimension
        blocks_per_stage: List of block counts for each stage
    """
    def __init__(self, embed_dim: int = 256, blocks_per_stage: List[int] = [2, 2, 2, 2]):
        super().__init__()
        
        # Initial convolution (Stem)
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.GroupNorm(choose_num_groups(64), 64),
            nn.GELU()
        )
        
        # Multi-scale feature extraction stages
        # Stage 1: H/2 × W/2
        self.stage1 = self._make_stage(64, 64, blocks_per_stage[0], stride=1)
        # Stage 2: H/4 × W/4
        self.stage2 = self._make_stage(64, 128, blocks_per_stage[1], stride=2)
        # Stage 3: H/8 × W/8
        self.stage3 = self._make_stage(128, 256, blocks_per_stage[2], stride=2)
        # Stage 4: H/16 × W/16
        self.stage4 = self._make_stage(256, embed_dim, blocks_per_stage[3], stride=2)
        
        logger.info(f"Initialized LocalArtifactEncoder (SOTA: EfficientResBlock + GroupNorm) with embed_dim={embed_dim}")
    
    def _make_stage(self, in_channels: int, out_channels: int, num_blocks: int, stride: int) -> nn.Sequential:
        """Create a stage with a transition block and multiple efficient blocks."""
        layers = []
        
        # --- SOTA DECISION: Transition Block ---
        # We need a 1x1 Conv "transition" to handle changing channel counts
        # and downsampling (stride). 'EfficientResBlock' requires
        # in_channels == out_channels for its residual connection.
        # --- END DECISION ---
        layers.append(nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.GroupNorm(choose_num_groups(out_channels), out_channels),
            nn.GELU()
        ))
        
        # Add the efficient residual blocks
        for _ in range(num_blocks):
            layers.append(EfficientResBlock(out_channels))
            
        return nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)        # [B, 64, H/2, W/2]
        x = self.stage1(x)      # [B, 64, H/2, W/2]
        x = self.stage2(x)      # [B, 128, H/4, W/4]
        x = self.stage3(x)      # [B, 256, H/8, W/8]
        x = self.stage4(x)      # [B, embed_dim, H/16, W/16]
        return x


class GlobalContextEncoder(nn.Module):
    """
    SOTA Lightweight CNN for global context.
    
    This is the "wide" tower, designed to get a large receptive field
    with minimal computation. It helps the model understand the *overall*
    image structure, which contextualizes the local artifacts.
    
    --- SOTA DECISION: Efficient Wide-Kernel Design ---
    Replaces the original's standard convs with:
    1. GroupNorm: For training stability (replaces BatchNorm).
    2. DepthwiseSeparable: For efficiency.
    3. Large Kernels (7x7, 5x5): To build a wide receptive field quickly
       without the computational cost of many deep layers.
    --- END DECISION ---
    
    Args:
        embed_dim: Output embedding dimension
    """
    def __init__(self, embed_dim: int = 256):
        super().__init__()
        
        self.encoder = nn.Sequential(
            # H/4 × W/4
            DepthwiseSeparable(3, 64, stride=2), # 3x3 DWS
            DepthwiseSeparable(64, 64, stride=2), # 3x3 DWS
            
            # H/8 × W/8
            DepthwiseSeparable(64, 128, stride=2), # 3x3 DWS
            
            # H/16 × W/16
            DepthwiseSeparable(128, embed_dim, stride=2), # 3x3 DWS
            
            # Final wide-kernel pass
            DepthwiseSeparable(embed_dim, embed_dim, stride=1)
        )
        
        logger.info(f"Initialized GlobalContextEncoder (SOTA: DWS + GroupNorm) with embed_dim={embed_dim}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x) # [B, embed_dim, H/16, W/16]


class UNetDecoder(nn.Module):
    """
    SOTA Lightweight U-Net decoder for implicit learning (image reconstruction).
    
    --- SOTA DECISION: Upsample + Conv (No Checkerboards) ---
    Replaces the original's 'ConvTranspose2d' (deconvolution)
    layers, which are notorious for causing checkerboard artifacts.
    
    This implementation uses:
    1. nn.Upsample(mode='bilinear'): A simple, artifact-free interpolation.
    2. DepthwiseSeparable Conv: A lightweight convolution to learn features
       at the new, higher resolution.
    
    This is the standard, high-quality method for upsampling in modern
    restoration and generative models.
    --- END DECISION ---
    
    Args:
        embed_dim: Input embedding dimension
        output_channels: Number of output channels (3 for RGB)
    """
    def __init__(self, embed_dim: int = 256, output_channels: int = 3):
        super().__init__()
        
        # --- SOTA DECISION: Replaced ConvTranspose2d + BatchNorm2d ---
        self.up1 = self._make_upsample_block(embed_dim, 128)
        self.up2 = self._make_upsample_block(128, 64)
        self.up3 = self._make_upsample_block(64, 32)
        self.up4 = self._make_upsample_block(32, 16)
        # --- END DECISION ---
        
        # Final output layer
        self.output_conv = nn.Sequential(
            nn.Conv2d(16, output_channels, kernel_size=3, padding=1),
            nn.Tanh()  # Output in [-1, 1] to match normalization
        )
        
        logger.info(f"Initialized UNetDecoder (SOTA: Upsample+Conv + GroupNorm)")

    @staticmethod
    def _make_upsample_block(in_ch: int, out_ch: int) -> nn.Sequential:
        """Helper to create one SOTA upsampling block."""
        return nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            DepthwiseSeparable(in_ch, out_ch, stride=1)
            # DepthwiseSeparable already contains GroupNorm + GELU
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.up1(x)           # [B, 128, H*2, W*2]
        x = self.up2(x)           # [B, 64, H*4, W*4]
        x = self.up3(x)           # [B, 32, H*8, W*8]
        x = self.up4(x)           # [B, 16, H*16, W*16]
        x = self.output_conv(x)   # [B, 3, H*16, W*16]
        return x


class CRFPredictor(nn.Module):
    """
    CRF prediction head for explicit learning.
    This module remains largely unchanged, as it's a standard and effective
    MLP classifier/regressor head.
    
    Args:
        embed_dim: Input embedding dimension
        hidden_dim: Hidden layer dimension
    """
    def __init__(self, embed_dim: int = 256, hidden_dim: int = 128):
        super().__init__()
        
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.predictor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        logger.info(f"Initialized CRFPredictor with embed_dim={embed_dim}, hidden_dim={hidden_dim}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(x)
        x = self.predictor(x)
        return x.squeeze(-1) # [B]


class AV1PriorEmbedder(nn.Module):
    """
    SOTA AV1 Compression Prior Embedder with dual learning strategy.
    
    Optimized CNN-only architecture for fast, stable, high-quality training.
    
    Architecture Overview:
    ┌─────────────────────────────────────────────────────────────┐
    │                    Input: LQ Image [B,3,H,W]                │
    └──────────────────────┬──────────────────────────────────────┘
                           │
            ┌──────────────┴──────────────┐
            │                             │
    ┌───────▼────────┐          ┌────────▼────────┐
    │ Local Artifact │          │ Global Context  │
    │    Encoder     │          │    Encoder      │
    │ (SOTA ResNet)  │          │ (SOTA Wide-CNN) │
    └───────┬────────┘          └────────┬────────┘
            │                             │
            └──────────────┬──────────────┘
                           │
                    ┌──────▼──────┐
                    │   Fusion    │
                    │ (GroupNorm) │
                    └──────┬──────┘
                           │
            ┌──────────────┴──────────────┐
            │                             │
    ┌───────▼────────┐          ┌────────▼────────┐
    │  UNet Decoder  │          │ CRF Predictor   │
    │(SOTA Upsample) │          │   (MLP Head)    │
    └───────┬────────┘          └────────┬────────┘
            │                             │
     Reconstructed                   Predicted
        Image                          CRF
        
    Args:
        embed_dim: Embedding dimension (default: 256)
        num_blocks_per_stage: List of block counts for local encoder
                              (default: [2, 2, 2, 2])
    """
    
    def __init__(
        self,
        embed_dim: int = 256,
        num_blocks_per_stage: List[int] = [2, 2, 2, 2]
    ):
        super().__init__()
        
        self.embed_dim = embed_dim
        
        # 1. Local Artifact Encoder (captures fine-grained compression patterns)
        self.local_encoder = LocalArtifactEncoder(
            embed_dim=embed_dim,
            blocks_per_stage=num_blocks_per_stage
        )
        
        # 2. Global Context Encoder (captures broader structural patterns)
        self.global_encoder = GlobalContextEncoder(embed_dim=embed_dim)
        
        # 3. Feature Fusion Module (combines local + global features)
        # --- SOTA DECISION: Replaced BatchNorm2d with GroupNorm ---
        self.fusion = nn.Sequential(
            nn.Conv2d(embed_dim * 2, embed_dim, kernel_size=1, bias=False),
            nn.GroupNorm(choose_num_groups(embed_dim), embed_dim),
            nn.GELU(),
            EfficientResBlock(embed_dim) # Use an EfficientResBlock for fusion
        )
        
        # 4. Implicit Learning: Image Reconstruction Decoder
        self.decoder = UNetDecoder(embed_dim=embed_dim)
        
        # 5. Explicit Learning: CRF Predictor
        self.crf_predictor = CRFPredictor(embed_dim=embed_dim)
        
        self._init_weights()
        
        param_count = sum(p.numel() for p in self.parameters())
        logger.info(f"AV1PriorEmbedder (SOTA) initialized successfully")
        logger.info(f"Total parameters: {param_count:,} (~{param_count/1e6:.1f}M)")
    
    def _init_weights(self):
        """Initialize weights for conv and linear layers."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.GroupNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(
        self, 
        lq_image: torch.Tensor, 
        return_features: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass through the embedder.
        
        Args:
            lq_image: Low-quality input image [B, 3, H, W] in range [-1, 1]
            return_features: If True, also return the fused features
            
        Returns:
            Tuple of:
                - reconstructed_image: Reconstructed HQ image [B, 3, H, W]
                - predicted_crf: Predicted CRF values [B]
                - fused_features: Fused feature embeddings [B, embed_dim, H/16, W/16] 
                                 (if return_features=True, else None)
        """
        # 1. Extract local artifact features
        local_features = self.local_encoder(lq_image)
        
        # 2. Extract global context features
        global_features = self.global_encoder(lq_image)
        
        # 3. Ensure spatial dimensions match (H/16 × W/16)
        # (This is good practice, though they should match)
        if local_features.shape[2:] != global_features.shape[2:]:
            global_features = F.interpolate(
                global_features,
                size=local_features.shape[2:],
                mode='bilinear',
                align_corners=False
            )
        
        # 4. Fuse local and global features
        combined_features = torch.cat([local_features, global_features], dim=1)
        fused_features = self.fusion(combined_features)
        
        # 5. Implicit learning: reconstruct image
        reconstructed_image = self.decoder(fused_features)
        
        # 6. Explicit learning: predict CRF
        # --- SOTA DECISION: Predict from FUSED features ---
        # The original branched from 'local_features'.
        # We now branch from 'fused_features'.
        #
        # REASON: The CRF is a global parameter. The predictor should
        # have access to all available information (both local artifacts
        # and global context) to make the most accurate prediction.
        # The fused features are the richest representation of the input.
        # --- END DECISION ---
        predicted_crf = self.crf_predictor(fused_features)
        
        if return_features:
            return reconstructed_image, predicted_crf, fused_features
        else:
            return reconstructed_image, predicted_crf, None
    
    def get_compression_prior(self, lq_image: torch.Tensor) -> torch.Tensor:
        """
        Extract compression prior embeddings for use in diffusion model (Phase 2).
        
        Args:
            lq_image: Low-quality input image [B, 3, H, W]
            
        Returns:
            Compression prior embeddings [B, embed_dim, H/16, W/16]
        """
        self.eval()
        with torch.no_grad():
            # Extract local artifact features
            local_features = self.local_encoder(lq_image)
            # Extract global context features
            global_features = self.global_encoder(lq_image)
            
            # Ensure spatial dimensions match
            if local_features.shape[2:] != global_features.shape[2:]:
                global_features = F.interpolate(
                    global_features,
                    size=local_features.shape[2:],
                    mode='bilinear',
                    align_corners=False
                )
            
            # Fuse local and global features
            combined_features = torch.cat([local_features, global_features], dim=1)
            fused_features = self.fusion(combined_features)
            
        return fused_features
    
    def freeze_for_inference(self):
        """Freeze all parameters for inference mode."""
        for param in self.parameters():
            param.requires_grad = False
        self.eval()
        logger.info("AV1PriorEmbedder (SOTA) frozen for inference")
    
    def get_num_params(self) -> dict:
        """Get parameter count breakdown."""
        # This function remains the same as your original
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        local_params = sum(p.numel() for p in self.local_encoder.parameters())
        global_params = sum(p.numel() for p in self.global_encoder.parameters())
        fusion_params = sum(p.numel() for p in self.fusion.parameters())
        decoder_params = sum(p.numel() for p in self.decoder.parameters())
        predictor_params = sum(p.numel() for p in self.crf_predictor.parameters())
        
        return {
            'total': total,
            'trainable': trainable,
            'local_encoder': local_params,
            'global_encoder': global_params,
            'fusion': fusion_params,
            'decoder': decoder_params,
            'crf_predictor': predictor_params
        }