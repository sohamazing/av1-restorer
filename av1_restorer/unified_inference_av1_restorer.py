#!/usr/bin/env python3
"""
av1_restorer/unified_inference_av1_restorer.py - Unified Inference Engine for AV1 Restoration
===============================================================================

Core inference engine used by both image and video restoration scripts.
Handles all model architectures with automatic detection and configuration.

Features:
    ✓ Automatic model architecture detection from checkpoint
    ✓ EMA weight prioritization for best quality
    ✓ Memory-efficient tiled inference for large images
    ✓ Test-Time Augmentation (TTA) support
    ✓ Cross-platform device support (CUDA/MPS/CPU)
    ✓ Automatic normalization configuration

Author: Soham Mukherjee
Version: 5.0 (Production Final)
License: MIT
"""

import logging
import sys
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import torchvision.transforms.v2 as T
from PIL import Image

# Setup project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Import all supported model architectures
try:
    from av1_restorer.models.av1_restorer import create_av1_restorer
    from av1_restorer.models.av1_nano_unet_restorer import create_av1_nano_unet_restorer
    from av1_restorer.models.av1_nano_resnet_restorer import create_av1_nano_resnet_restorer
    from av1_restorer.models.av1_nano_fbcnn_restorer import create_av1_nano_fbcnn_restorer
    from av1_restorer.models.av1_nano_mamba_restorer import create_av1_nano_mamba_restorer
except ImportError as e:
    print(f"✗ Import error: {e}", file=sys.stderr)
    print("  Ensure all model modules exist in av1_restorer/models/", file=sys.stderr)
    sys.exit(1)

logger = logging.getLogger("AV1Restorer.Inference")


def setup_device(device_str: str) -> torch.device:
    """
    Setup compute device with intelligent auto-detection.
    
    Priority order: CUDA > MPS (Apple Silicon) > CPU
    
    Args:
        device_str: Device specification ('auto', 'cuda', 'mps', 'cpu')
        
    Returns:
        Configured torch.device
    """
    if device_str == 'auto':
        if torch.cuda.is_available():
            device = torch.device('cuda')
            logger.info(f"Auto-selected: CUDA ({torch.cuda.get_device_name(0)})")
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = torch.device('mps')
            logger.info("Auto-selected: Apple MPS")
        else:
            device = torch.device('cpu')
            logger.info("Auto-selected: CPU (slow, consider GPU)")
        return device
    
    logger.info(f"User-specified device: {device_str}")
    return torch.device(device_str)


class AV1RestorerInference:
    """
    Unified inference engine for all AV1 restoration architectures.
    
    This class provides a single interface for all restoration models,
    handling architecture detection, checkpoint loading, preprocessing,
    inference, and postprocessing automatically.
    
    Attributes:
        model: Loaded restoration model
        config: Training configuration from checkpoint
        model_type: Detected architecture type
        device: Compute device
        tile_size: Maximum tile dimension for processing
        tile_overlap: Overlap between tiles for seamless stitching
        norm_range: Image normalization range from training
        is_conditional: Whether model uses CRF+Preset conditioning
    
    Example:
        >>> inferenceRestorer = AV1RestorerInference("checkpoints/best.pth", device='cuda')
        >>> img = Image.open("compressed.avif")
        >>> restored = inferenceRestorer.restore(img, crf=45, preset=4)
        >>> restored.save("restored.png")
    """
    
    def __init__(
        self, 
        checkpoint_path: str, 
        device: str = 'auto',
        tile_size: int = 512,
        tile_overlap: int = 64
    ):
        """
        Initialize the restoration engine.
        
        Args:
            checkpoint_path: Path to model checkpoint (.pth file)
            device: Compute device ('auto', 'cuda', 'mps', 'cpu')
            tile_size: Maximum tile dimension for processing large images
            tile_overlap: Overlap pixels between tiles for blending
            
        Raises:
            FileNotFoundError: If checkpoint doesn't exist
            ValueError: If checkpoint is invalid or missing config
            RuntimeError: If model initialization fails
        """
        self.device = setup_device(device)
        self.tile_size = tile_size
        self.tile_overlap = tile_overlap
        
        # Image preprocessing pipeline: PIL -> Tensor [0,1]
        self.to_tensor = T.Compose([
            T.ToImage(),
            T.ToDtype(torch.float32, scale=True)
        ])
        
        # Load model and configuration from checkpoint
        self.model, self.config, self.model_type = self._load_checkpoint(checkpoint_path)
        self.model.eval().to(self.device)
        
        # Determine conditioning type
        self.is_conditional = (self.model_type == 'unet')
        
        # Configure normalization from training config
        norm_range = self.config.get('dataset', {}).get('norm_range', [-1, 1])
        self.norm_range = tuple(norm_range) if isinstance(norm_range, list) else norm_range
        
        if self.norm_range == (-1, 1):
            # Training used [-1, 1] normalization
            self.normalize = T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            self.denormalize = lambda x: (x + 1.0) / 2.0
        else:
            # Training used [0, 1] normalization
            self.normalize = lambda x: x
            self.denormalize = lambda x: x
        
        # Log initialization summary
        logger.info("✓ Inference Restorer initialized successfully")
        logger.info(f"  Architecture: {self.model_type}")
        logger.info(f"  Device: {self.device}")
        logger.info(f"  Conditioning: {'CRF+Preset' if self.is_conditional else 'CRF-Only'}")
        logger.info(f"  Normalization: {self.norm_range}")
        logger.info(f"  Tiling: {tile_size}×{tile_size} (overlap: {tile_overlap}px)")
    
    def _load_checkpoint(self, ckpt_path: str) -> Tuple[torch.nn.Module, dict, str]:
        """
        Load checkpoint and reconstruct model architecture.
        
        The checkpoint config is the "source of truth" for all model parameters,
        ensuring perfect architecture matching and portability.
        
        Args:
            ckpt_path: Path to checkpoint file
            
        Returns:
            Tuple of (model, config_dict, model_type_string)
            
        Raises:
            FileNotFoundError: If checkpoint doesn't exist
            ValueError: If checkpoint is invalid
            RuntimeError: If model reconstruction fails
        """
        # Resolve checkpoint path
        path = Path(ckpt_path).expanduser()
        if not path.is_file():
            # Try relative to project root
            path = PROJECT_ROOT / ckpt_path
            if not path.is_file():
                raise FileNotFoundError(
                    f"Checkpoint not found at '{ckpt_path}' or '{path}'"
                )
        
        logger.info(f"Loading checkpoint: {path.name}")
        
        # Load checkpoint
        try:
            checkpoint = torch.load(path, map_location='cpu', weights_only=False)
        except Exception as e:
            raise ValueError(f"Failed to load checkpoint: {e}")
        
        # Extract configuration
        config = checkpoint.get('config')
        if not config:
            raise ValueError(
                f"Checkpoint '{path.name}' missing 'config' dictionary. "
                "This may be from an older training run or corrupted."
            )
        
        # Extract model parameters
        model_cfg = config.get('model', {})
        dset_cfg = config.get('dataset', {})
        
        model_type = model_cfg.get('type', 'unet')
        size = model_cfg.get('size')
        crf_range = tuple(dset_cfg.get('crf_range', [23, 63]))
        preset_range = tuple(dset_cfg.get('preset_range', [0, 8]))
        norm_range = tuple(dset_cfg.get('norm_range', [-1, 1]))
        
        if not size:
            raise ValueError(
                f"Checkpoint config missing 'model.size'. "
                "Cannot determine model architecture."
            )
        
        logger.info(f"  Model: {model_type} (size: {size})")
        logger.info(f"  CRF range: {crf_range}")
        logger.info(f"  Preset range: {preset_range}")
        
        # Create model using appropriate factory
        try:
            if model_type == 'unet':
                model = create_av1_restorer(size, crf_range, preset_range, norm_range)
            elif model_type == 'nano_unet':
                model = create_av1_nano_unet_restorer(size, *crf_range, norm_range)
            elif model_type == 'nano_resnet':
                model = create_av1_nano_resnet_restorer(size, *crf_range, norm_range)
            elif model_type == 'nano_fbcnn':
                model = create_av1_nano_fbcnn_restorer(size, *crf_range, norm_range)
            elif model_type == 'nano_mamba':
                model = create_av1_nano_mamba_restorer(size, *crf_range, norm_range)
            else:
                raise ValueError(
                    f"Unknown model type: '{model_type}'. "
                    f"Supported: unet, nano_unet, nano_resnet, nano_fbcnn, nano_mamba"
                )
        except Exception as e:
            raise RuntimeError(f"Failed to create model: {e}")
        
        # Load weights (prioritize EMA for best quality)
        weights = (
            checkpoint.get('ema_state_dict') or 
            checkpoint.get('ema_shadow') or 
            checkpoint.get('model_state_dict')
        )
        
        if not weights:
            raise ValueError(
                "No model weights found in checkpoint. "
                "Expected 'ema_state_dict', 'ema_shadow', or 'model_state_dict'."
            )
        
        # Load state dict with flexible matching
        missing, unexpected = model.load_state_dict(weights, strict=False)
        
        if missing:
            logger.warning(f"  Missing keys: {missing}")
        if unexpected:
            logger.warning(f"  Unexpected keys: {unexpected}")
        
        # Log model statistics
        params = sum(p.numel() for p in model.parameters())
        logger.info(f"  Parameters: {params:,} ({params/1e6:.2f}M)")
        
        return model, config, model_type
    
    @torch.no_grad()
    def restore(
        self,
        image: Image.Image,
        crf: int,
        preset: int = 4,
        center_crop: bool = False,
        use_tta: bool = False
    ) -> Image.Image:
        """
        Restore a single PIL Image.
        
        Pipeline:
            1. Convert PIL Image to normalized tensor
            2. Apply model-specific conditioning
            3. Run inference (with tiling if needed)
            4. Denormalize and convert back to PIL
        
        Args:
            image: Input PIL Image (RGB)
            crf: CRF value (typically 23-63)
            preset: Preset value (0-8, only used for conditional models)
            center_crop: If True, only process center tile (faster)
            use_tta: If True, use Test-Time Augmentation (8x slower, +quality)
            
        Returns:
            Restored PIL Image (RGB)
            
        Raises:
            RuntimeError: If inference fails
        """
        # Preprocess: PIL -> [0,1] Tensor -> Normalized Tensor
        x = self.normalize(self.to_tensor(image)).unsqueeze(0).to(self.device)
        
        # Get inference function (prefer model's 'inference' method if available)
        inference_fn = getattr(self.model, 'inference', self.model.forward)
        
        try:
            # Run inference based on model type
            if self.is_conditional:
                # Conditional U-Net: requires CRF and Preset tensors
                crf_t = torch.tensor([[float(crf)]], device=self.device)
                preset_t = torch.tensor([[float(preset)]], device=self.device)
                
                output = inference_fn(
                    x, crf_t, preset_t,
                    tile_size=self.tile_size,
                    tile_overlap=self.tile_overlap,
                    center_crop=center_crop,
                    use_tta=use_tta
                )
            else:
                # Nano models: CRF-only
                # Check if inference function supports tiling/TTA
                if "tile_size" in inference_fn.__code__.co_varnames:
                    output = inference_fn(
                        x,
                        tile_size=self.tile_size,
                        tile_overlap=self.tile_overlap,
                        center_crop=center_crop,
                        use_tta=use_tta
                    )
                else:
                    # Fallback to basic forward pass
                    output = inference_fn(x)
        
        except Exception as e:
            raise RuntimeError(f"Inference failed: {e}")
        
        # Postprocess: Denormalize -> Clamp -> [0,255] uint8 -> PIL
        output = torch.clamp(self.denormalize(output), 0.0, 1.0)
        output_np = output.squeeze(0).cpu().permute(1, 2, 0).numpy()
        output_np = (output_np * 255.0).astype(np.uint8)
        
        return Image.fromarray(output_np)


def compute_mse(img1: np.ndarray, img2: np.ndarray) -> float:
    """
    Compute Mean Squared Error between two images.
    
    Args:
        img1: First image as numpy array [H, W, C]
        img2: Second image as numpy array [H, W, C]
        
    Returns:
        MSE value (lower is better)
    """
    return float(np.mean((img1.astype(np.float64) - img2.astype(np.float64)) ** 2))