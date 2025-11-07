#!/usr/bin/env python3
"""
restore_av1.py - Unified Inference Script for AV1 Artifact Restoration
========================================================================

Production-ready inference script supporting all AV1 restoration model architectures
with automatic model detection, comprehensive error handling, and flexible processing modes.

Supported Architectures:
    - Conditional U-Net (CRF + Preset conditioning)
    - Nano U-Net (CRF-specialized, lightweight)
    - Nano ResNet (Single-scale, fastest)
    - Nano FBCNN (FBCNN-inspired)
    - Nano Mamba (Hybrid CNN+SSM)

Features:
    ✓ Single-image and batch directory processing
    ✓ Automatic CRF/Preset detection from filenames
    ✓ Test mode with metrics computation (MSE vs HQ)
    ✓ Memory-efficient tiling for large images
    ✓ Safe device selection (CUDA/MPS/CPU)
    ✓ Dry-run mode for testing
    ✓ Structured directory preservation
    ✓ NEW: Reports Average FPS and Per-CRF metrics

Usage Examples:
    # Single image with auto-detection
    python restore_av1.py -c checkpoints/best.pth -i input.avif -o output.png --auto
    
    # Batch processing with manual CRF
    python restore_av1.py -c checkpoints/best.pth -d ./lq -od ./restored --crf 45 --preset 4
    
    # Test mode with metrics
    python restore_av1.py -c checkpoints/best.pth -d ./test/lq -od ./results \
        --test --hq_dir ./test/hq --auto
    
    # Large images with tiling
    python restore_av1.py -c best.pth -i large.avif -o restored.png \
        --crf 35 --tile 512 --overlap 64

Author: Soham Mukherjee
Version: 3.1 (Production)
License: MIT
"""

import argparse
import logging
import re
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Union
from dataclasses import dataclass

import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
import torchvision.transforms.v2 as T

# ==============================================================================
# SECTION 1: Project Setup & Imports
# ==============================================================================

# Add project root to path for module imports
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
    print(f"❌ Failed to import model architectures: {e}", file=sys.stderr)
    print("   Ensure you're running from the project root directory.", file=sys.stderr)
    sys.exit(1)

# ==============================================================================
# SECTION 2: Logging Configuration
# ==============================================================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger("AV1Restorer")


# ==============================================================================
# SECTION 3: Utility Functions & Constants
# ==============================================================================

# Regex pattern for extracting CRF and preset from filenames
# Matches patterns like: "image_001_crf45_p4.avif"
FILENAME_PATTERN = re.compile(r"_crf(\d+)_p(\d+)\.")

# Supported image extensions
IMAGE_EXTENSIONS = ('.avif', '.png', '.jpg', '.jpeg', '.webp', '.bmp', '.tiff', '.npy')


@dataclass
class ImageMetrics:
    """Container for image quality metrics."""
    lq_loss: float
    restored_loss: float
    improvement: float
    improvement_percent: float


def extract_params_from_filename(name: str) -> Optional[Tuple[int, int]]:
    """
    Extract CRF and preset values from a filename.
    
    Expected format: {base}_crf{CRF}_p{PRESET}.{ext}
    Example: "image_001_crf45_p4.avif" -> (45, 4)
    
    Args:
        name: Filename to parse
        
    Returns:
        Tuple of (crf, preset) if found, None otherwise
        
    Example:
        >>> extract_params_from_filename("img_crf35_p4.avif")
        (35, 4)
        >>> extract_params_from_filename("random_name.png")
        None
    """
    match = FILENAME_PATTERN.search(name)
    if match:
        crf = int(match.group(1))
        preset = int(match.group(2))
        return (crf, preset)
    return None


def find_images(path: Path, extensions: Tuple[str, ...] = IMAGE_EXTENSIONS) -> List[Path]:
    """
    Recursively find all image files in a directory.
    
    Args:
        path: Root directory to search
        extensions: Tuple of valid file extensions (case-insensitive)
        
    Returns:
        Sorted list of Path objects for found images
        
    Example:
        >>> images = find_images(Path("./dataset"))
        >>> len(images)
        1523
    """
    return sorted([
        p for p in path.rglob('*') 
        if p.is_file() and p.suffix.lower() in extensions
    ])


def setup_device(device_str: str) -> torch.device:
    """
    Setup compute device with intelligent auto-detection.
    
    Priority order: CUDA > MPS (Apple Silicon) > CPU
    
    Args:
        device_str: Device specification ('auto', 'cuda', 'mps', 'cpu')
        
    Returns:
        Configured torch.device
        
    Example:
        >>> device = setup_device('auto')
        >>> print(device)
        device(type='cuda', index=0)
    """
    if device_str == 'auto':
        if torch.cuda.is_available():
            device = torch.device('cuda')
            logger.info(f"Auto-selected device: CUDA ({torch.cuda.get_device_name(0)})")
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = torch.device('mps')
            logger.info("Auto-selected device: Apple MPS")
        else:
            device = torch.device('cpu')
            logger.info("Auto-selected device: CPU (slow, consider using GPU)")
        return device
    
    logger.info(f"User-specified device: {device_str}")
    return torch.device(device_str)


def compute_mse(img1: np.ndarray, img2: np.ndarray) -> float:
    """
    Compute Mean Squared Error (L2 loss) between two images.
    
    Args:
        img1: First image as numpy array [H, W, C]
        img2: Second image as numpy array [H, W, C]
        
    Returns:
        MSE value (lower is better)
        
    Note:
        Images are converted to float64 for precise computation.
    """
    return float(np.mean((img1.astype(np.float64) - img2.astype(np.float64)) ** 2))


# ==============================================================================
# SECTION 4: Model Loader & Inference Engine
# ==============================================================================

class AV1Restorer:
    """
    Unified inference engine for all AV1 restoration architectures.
    
    This class handles:
        - Automatic model architecture detection from checkpoint
        - Checkpoint loading with EMA weight prioritization
        - Device management and memory optimization
        - Normalization/denormalization based on training config
        - Tiled inference for large images
    
    Attributes:
        model (nn.Module): Loaded restoration model
        config (dict): Training configuration from checkpoint
        model_type (str): Detected model architecture type
        device (torch.device): Compute device
        tile_size (int): Maximum tile size for inference
        tile_overlap (int): Overlap between tiles for seamless stitching
        norm_range (tuple): Image normalization range
        is_conditional (bool): Whether model uses CRF/Preset conditioning
    
    Example:
        >>> restorer = AV1Restorer("checkpoints/best.pth", device='cuda')
        >>> img = Image.open("compressed.avif")
        >>> restored = restorer.restore(img, crf=45, preset=4)
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
            tile_size: Maximum tile dimension for processing
            tile_overlap: Overlap pixels between tiles for blending
            
        Raises:
            FileNotFoundError: If checkpoint doesn't exist
            ValueError: If checkpoint is invalid or corrupted
            RuntimeError: If model initialization fails
        """
        self.device = setup_device(device)
        self.tile_size = tile_size
        self.tile_overlap = tile_overlap
        
        # Image preprocessing pipeline
        self.to_tensor = T.Compose([
            T.ToImage(),
            T.ToDtype(torch.float32, scale=True)  # [0, 255] -> [0, 1]
        ])
        
        # Load model and configuration
        self.model, self.config, self.model_type = self._load_checkpoint(checkpoint_path)
        self.model.eval().to(self.device)
        
        # Determine if model uses conditioning
        self.is_conditional = (self.model_type == 'unet')
        
        # Configure normalization based on training config
        norm_range = self.config.get('dataset', {}).get('norm_range', [-1, 1])
        self.norm_range = tuple(norm_range) if isinstance(norm_range, list) else norm_range
        
        if self.norm_range == (-1, 1):
            # Training used [-1, 1] normalization
            self.normalize = T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            self.denormalize = lambda x: (x + 1.0) / 2.0
        else:
            # Training used [0, 1] normalization (identity)
            self.normalize = lambda x: x
            self.denormalize = lambda x: x
        
        # Log initialization summary
        logger.info("✓ Model loaded successfully")
        logger.info(f"  Architecture: {self.model_type}")
        logger.info(f"  Device: {self.device}")
        logger.info(f"  Conditioning: {'CRF + Preset' if self.is_conditional else 'CRF-Specialized'}")
        logger.info(f"  Normalization: {self.norm_range}")
        logger.info(f"  Tiling: {tile_size}×{tile_size} (overlap: {tile_overlap}px)")
    
    def _load_checkpoint(self, ckpt_path: str) -> Tuple[torch.nn.Module, dict, str]:
        """
        Load checkpoint and reconstruct appropriate model architecture.
        
        DESIGN DECISION: Architecture detection is done ENTIRELY from checkpoint config.
        This is the most robust approach because:
        
        ✓ Guarantees architecture matches training exactly
        ✓ Eliminates user error (wrong model type/size specification)
        ✓ Checkpoint is self-contained and portable
        ✓ Prevents architecture mismatch errors
        ✓ Simplifies CLI (no need for --model-type, --size flags)
        
        The checkpoint config acts as the "source of truth" for all model parameters:
        - Architecture type (unet, nano_unet, nano_resnet, etc.)
        - Model size (tiny, small, base, large, etc.)
        - CRF/Preset ranges (for model initialization)
        - Normalization range (for preprocessing)
        
        This method:
            1. Resolves checkpoint path (absolute or relative to project)
            2. Loads checkpoint data
            3. Extracts and validates config
            4. Reconstructs model using appropriate factory function
            5. Loads weights (prioritizing EMA if available)
        
        Args:
            ckpt_path: Path to checkpoint file
            
        Returns:
            Tuple of (model, config_dict, model_type_str)
            
        Raises:
            FileNotFoundError: If checkpoint doesn't exist
            ValueError: If checkpoint is invalid or missing required data
            RuntimeError: If model reconstruction fails
        """
        # Resolve checkpoint path
        path = Path(ckpt_path).expanduser()
        if not path.is_file():
            # Try relative to project root
            potential_path = PROJECT_ROOT / ckpt_path
            if potential_path.is_file():
                path = potential_path
            else:
                raise FileNotFoundError(
                    f"Checkpoint not found at '{ckpt_path}' or '{potential_path}'"
                )
        
        logger.info(f"Loading checkpoint: {path.name}")
        
        # Load checkpoint data
        try:
            checkpoint = torch.load(path, map_location='cpu', weights_only=False)
        except Exception as e:
            raise ValueError(f"Failed to load checkpoint: {e}")
        
        # Extract configuration (CRITICAL: This is our source of truth)
        config = checkpoint.get('config')
        if not config:
            raise ValueError(
                "Checkpoint missing 'config' dictionary. "
                "This checkpoint may be from an older training run or corrupted."
            )
        
        # Extract model parameters from config
        model_config = config.get('model', {})
        dataset_config = config.get('dataset', {})
        
        model_type = model_config.get('type', 'unet')
        size = model_config.get('size')
        crf_range = tuple(dataset_config.get('crf_range', [23, 63]))
        preset_range = tuple(dataset_config.get('preset_range', [0, 8]))
        norm_range = tuple(dataset_config.get('norm_range', [-1, 1]))
        
        # Validate required parameters
        if not size:
            raise ValueError(
                "Checkpoint config missing 'model.size'. "
                "Cannot determine model architecture."
            )
        
        logger.info(f"Detected architecture: {model_type} (size: {size})")
        logger.info(f"  CRF range: {crf_range}")
        logger.info(f"  Preset range: {preset_range}")
        logger.info(f"  Norm range: {norm_range}")
        
        # Create model based on architecture type
        # Each factory function recreates the EXACT architecture used during training
        try:
            if model_type == 'unet':
                # Conditional U-Net (CRF + Preset)
                model = create_av1_restorer(size, crf_range, preset_range, norm_range)
            
            elif model_type == 'nano_unet':
                # Lightweight U-Net (CRF-specialized)
                crf_min, crf_max = crf_range
                model = create_av1_nano_unet_restorer(size, crf_min, crf_max, norm_range)
            
            elif model_type == 'nano_resnet':
                # Single-scale ResNet (fastest)
                crf_min, crf_max = crf_range
                model = create_av1_nano_resnet_restorer(size, crf_min, crf_max, norm_range)
            
            elif model_type == 'nano_fbcnn':
                # FBCNN-inspired architecture
                crf_min, crf_max = crf_range
                model = create_av1_nano_fbcnn_restorer(size, crf_min, crf_max, norm_range)
            
            elif model_type == 'nano_mamba':
                # Hybrid CNN+SSM architecture
                crf_min, crf_max = crf_range
                model = create_av1_nano_mamba_restorer(size, crf_min, crf_max, norm_range)
            
            else:
                raise ValueError(
                    f"Unknown model type: '{model_type}'. "
                    f"Supported types: unet, nano_unet, nano_resnet, nano_fbcnn, nano_mamba"
                )
        
        except Exception as e:
            raise RuntimeError(f"Failed to create model architecture: {e}")
        
        # Load model weights (prioritize EMA for best quality)
        # EMA (Exponential Moving Average) weights typically perform better than
        # the final training weights due to averaging over many steps
        weights = None
        if 'ema_state_dict' in checkpoint:
            weights = checkpoint['ema_state_dict']
            logger.info("  Using EMA weights (best quality)")
        elif 'ema_shadow' in checkpoint:
            weights = checkpoint['ema_shadow']
            logger.info("  Using EMA shadow weights")
        elif 'model_state_dict' in checkpoint:
            weights = checkpoint['model_state_dict']
            logger.info("  Using standard model weights")
        else:
            raise ValueError(
                "Checkpoint contains no model weights. "
                "Expected 'ema_state_dict', 'ema_shadow', or 'model_state_dict'."
            )
        
        # Load state dict with flexible key matching
        # strict=False allows minor mismatches (e.g., missing/extra keys from different PyTorch versions)
        missing_keys, unexpected_keys = model.load_state_dict(weights, strict=False)
        
        if missing_keys:
            logger.warning(f"  Missing keys in checkpoint: {missing_keys}")
        if unexpected_keys:
            logger.warning(f"  Unexpected keys in checkpoint: {unexpected_keys}")
        
        # Log model statistics
        total_params = sum(p.numel() for p in model.parameters())
        logger.info(f"  Parameters: {total_params:,} ({total_params/1e6:.2f}M)")
        
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
        
        This method:
            1. Converts PIL Image to normalized tensor
            2. Applies appropriate conditioning (if needed)
            3. Runs inference (with tiling for large images)
            4. Denormalizes and converts back to PIL
        
        Args:
            image: Input PIL Image in RGB format
            crf: CRF value (23-63 typically)
            preset: Preset value (0-8 for libaom-av1, only used for conditional models)
            center_crop: If True, only process center tile (faster for large images)
            use_tta: If True, uses Test-Time Augmentation (8x slower, higher quality)
            
        Returns:
            Restored PIL Image
            
        Raises:
            RuntimeError: If inference fails
            
        Example:
            >>> img = Image.open("compressed.avif")
            >>> restored = restorer.restore(img, crf=45, preset=4)
        """
        # Prepare input tensor
        x = self.normalize(self.to_tensor(image)).unsqueeze(0).to(self.device)
        
        # Get inference function
        # The 'inference' method on the model is the SOTA one with TTA
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
                # Nano models: Call inference fn (if it exists) or forward
                # Note: We must check if the nano model's fn supports all args
                if "tile_size" in inference_fn.__code__.co_varnames:
                     output = inference_fn(
                        x,
                        tile_size=self.tile_size,
                        tile_overlap=self.tile_overlap,
                        center_crop=center_crop,
                        use_tta=use_tta
                    )
                else:
                    # Fallback to basic forward pass if fn signature doesn't match
                    output = inference_fn(x)
        
        except Exception as e:
            raise RuntimeError(f"Inference failed: {e}")
        
        # Denormalize and convert to PIL
        output = torch.clamp(self.denormalize(output), 0.0, 1.0)
        output_np = output.squeeze(0).cpu().permute(1, 2, 0).numpy()
        output_np = (output_np * 255.0).astype(np.uint8)
        
        return Image.fromarray(output_np)


# ==============================================================================
# SECTION 5: Processing Functions
# ==============================================================================

def process_single_image(
    restorer: AV1Restorer,
    img_path: Path,
    output_path: Path,
    crf: Optional[int],
    preset: Optional[int],
    auto_detect: bool,
    dry_run: bool = False,
    overwrite: bool = False,
    center_crop: bool = False,
    use_tta: bool = False,
    hq_path: Optional[Path] = None
) -> Optional[ImageMetrics]:
    """
    Process a single image file with comprehensive error handling.
    
    Args:
        restorer: Configured AV1Restorer instance
        img_path: Path to input image
        output_path: Path to save restored image
        crf: Manual CRF override (None to use auto-detection)
        preset: Manual preset override (None to use auto-detection)
        auto_detect: Enable auto-detection from filename
        dry_run: If True, only log actions without processing
        overwrite: If True, overwrite existing output files
        center_crop: If True, only process center tile
        use_tta: If True, enables Test-Time Augmentation
        hq_path: Optional path to HQ image for metrics computation
        
    Returns:
        ImageMetrics if HQ image provided, None otherwise
        
    Raises:
        Does not raise exceptions; logs errors and returns None on failure
    """
    # Check if output already exists
    if not overwrite and output_path.exists():
        logger.debug(f"Skip (already exists): {output_path.name}")
        return None
    
    # Auto-detect parameters from filename if enabled
    if auto_detect and (crf is None or preset is None):
        detected_params = extract_params_from_filename(img_path.name)
        if detected_params:
            detected_crf, detected_preset = detected_params
            if crf is None:
                crf = detected_crf
            if preset is None:
                preset = detected_preset
    
    # Validate CRF parameter
    if crf is None:
        logger.warning(f"Skipping {img_path.name}: CRF not specified or detected")
        return None
    
    # Default preset if not specified
    if preset is None:
        preset = 4
        logger.debug(f"Using default preset=4 for {img_path.name}")
    
    # Dry run mode: just log what would happen
    if dry_run:
        logger.info(
            f"[DRY RUN] {img_path.name} -> {output_path.name} "
            f"(CRF={crf}, Preset={preset}, TTA={use_tta})"
        )
        return None
    
    # Process the image
    try:
        # Load input image
        if img_path.suffix.lower() == '.npy':
            lq_np = np.load(img_path)
            if lq_np.dtype == np.float32 or lq_np.dtype == np.float64:
                # Assumes .npy is saved in [0, 1] range
                lq_img = Image.fromarray((lq_np * 255.0).astype(np.uint8))
            else:
                # Assumes [0, 255] uint8
                lq_img = Image.fromarray(lq_np)
        else:
            lq_img = Image.open(img_path).convert('RGB')
        
        # Run restoration
        restored_img = restorer.restore(lq_img, crf, preset, center_crop=center_crop, use_tta=use_tta)
        
        # Save output
        output_path.parent.mkdir(parents=True, exist_ok=True)
        restored_img.save(output_path, quality=95, optimize=True)
        
        # Compute metrics if HQ image provided
        if hq_path and hq_path.exists():
            # Load HQ image (handles .npy and standard images)
            if hq_path.suffix.lower() == '.npy':
                hq_np = np.load(hq_path)
                if hq_np.dtype == np.float32 or hq_np.dtype == np.float64:
                    hq_img = Image.fromarray((hq_np * 255.0).astype(np.uint8))
                else:
                    hq_img = Image.fromarray(hq_np)
            else:
                hq_img = Image.open(hq_path).convert('RGB')
            
            # Ensure matching sizes
            if hq_img.size != restored_img.size:
                if center_crop:
                    # Crop HQ to match restored center region
                    crop_h, crop_w = restored_img.size[1], restored_img.size[0]
                    start_h = (hq_img.height - crop_h) // 2
                    start_w = (hq_img.width - crop_w) // 2
                    hq_img = hq_img.crop((
                        start_w, start_h,
                        start_w + crop_w, start_h + crop_h
                    ))
                else:
                    # Resize HQ to match restored
                    logger.warning(
                        f"Size mismatch: Restored={restored_img.size}, "
                        f"HQ={hq_img.size}. Resizing HQ."
                    )
                    hq_img = hq_img.resize(restored_img.size, Image.LANCZOS)
            
            # Also crop LQ if needed for fair comparison
            if center_crop and lq_img.size != restored_img.size:
                crop_h, crop_w = restored_img.size[1], restored_img.size[0]
                start_h = (lq_img.height - crop_h) // 2
                start_w = (lq_img.width - crop_w) // 2
                lq_img = lq_img.crop((
                    start_w, start_h,
                    start_w + crop_w, start_h + crop_h
                ))
            
            # Convert to numpy arrays
            lq_np = np.array(lq_img)
            restored_np = np.array(restored_img)
            hq_np = np.array(hq_img)
            
            # Compute MSE metrics
            lq_mse = compute_mse(lq_np, hq_np)
            restored_mse = compute_mse(restored_np, hq_np)
            improvement = lq_mse - restored_mse
            improvement_pct = (improvement / lq_mse * 100.0) if lq_mse > 1e-9 else 0.0
            
            return ImageMetrics(
                lq_loss=lq_mse,
                restored_loss=restored_mse,
                improvement=improvement,
                improvement_percent=improvement_pct
            )
        
        return None
    
    except Exception as e:
        logger.error(f"Failed to process {img_path.name}: {e}", exc_info=True)
        return None


def process_directory(
    restorer: AV1Restorer,
    input_dir: Path,
    output_dir: Path,
    crf: Optional[int],
    preset: Optional[int],
    auto_detect: bool,
    test_mode: bool,
    dry_run: bool,
    overwrite: bool,
    center_crop: bool = False,
    use_tta: bool = False,
    hq_dir: Optional[Path] = None
) -> None:
    """
    Process all images in a directory with optional test mode metrics.
    
    In test mode, expects structure:
        input_dir/
        ├── crf_23/
        │   ├── preset_4/
        │   │   ├── img_001_crf23_p4.avif
        │   │   └── ...
        │   └── ...
        └── ...
    
    Args:
        restorer: Configured AV1Restorer instance
        input_dir: Root directory containing images
        output_dir: Directory to save restored images
        crf: Manual CRF override (acts as filter in test mode)
        preset: Manual preset override (acts as filter in test mode)
        auto_detect: Enable auto-detection from filenames
        test_mode: Enable structured directory processing
        dry_run: If True, only log actions without processing
        overwrite: If True, overwrite existing output files
        center_crop: If True, only process center tiles
        use_tta: If True, enables Test-Time Augmentation
        hq_dir: Optional directory containing HQ images for metrics
    """
    # Build list of images to process
    if test_mode:
        logger.info(f"Test mode: Scanning structured directories in {input_dir}")
        search_paths = []
        
        # Apply filters based on CRF and Preset arguments
        if crf is not None and preset is not None:
            # Specific CRF and Preset
            search_path = input_dir / f"crf_{crf}" / f"preset_{preset}"
            if search_path.exists():
                search_paths.append(search_path)
                logger.info(f"  Filter: crf_{crf}/preset_{preset}")
            else:
                logger.warning(f"  Path not found: {search_path}")
        
        elif crf is not None:
            # All presets for specific CRF
            crf_dir = input_dir / f"crf_{crf}"
            if crf_dir.exists():
                preset_dirs = [
                    p for p in crf_dir.iterdir()
                    if p.is_dir() and p.name.startswith('preset_')
                ]
                search_paths.extend(preset_dirs)
                logger.info(f"  Filter: crf_{crf}/preset_* ({len(preset_dirs)} presets)")
            else:
                logger.warning(f"  CRF directory not found: {crf_dir}")
        
        elif preset is not None:
            # All CRFs for specific preset
            for crf_dir in input_dir.glob("crf_*"):
                preset_dir = crf_dir / f"preset_{preset}"
                if preset_dir.exists():
                    search_paths.append(preset_dir)
            logger.info(f"  Filter: crf_*/preset_{preset} ({len(search_paths)} CRF levels)")
        
        else:
            # All CRFs and presets
            search_paths.extend(list(input_dir.rglob("preset_*")))
            logger.info(f"  No filter: processing all preset directories")
        
        # Collect all images from search paths
        image_files = []
        for search_path in search_paths:
            if search_path.exists():
                found = find_images(search_path)
                image_files.extend(found)
                logger.debug(f"  Found {len(found)} images in {search_path}")
            else:
                logger.warning(f"  Search path not found: {search_path}")
    
    else:
        # Standard mode: recursive search
        logger.info(f"Standard mode: Recursively scanning {input_dir}")
        image_files = find_images(input_dir)
    
    # Check if any images were found
    if not image_files:
        logger.warning("No images found matching the specified criteria")
        return
    
    logger.info(f"Found {len(image_files)} images to process")
    
    # Process all images with progress bar
    metrics_list: List[ImageMetrics] = []
    per_crf_metrics = defaultdict(list)
    start_time = time.perf_counter()
    
    for img_path in tqdm(image_files, desc="Restoring images", unit="img"):
        # Preserve relative directory structure
        try:
            rel_path = img_path.relative_to(input_dir)
        except ValueError:
            # Fallback if image is not relative to input_dir (e.g., test mode)
            # Try to find 'crf_XX/preset_Y' structure
            try:
                # Assumes structure like /base/crf_XX/preset_Y/img.png
                # and input_dir is /base/
                rel_path = img_path.relative_to(input_dir.parent.parent)
            except ValueError:
                # Last resort, just use the name
                rel_path = Path(img_path.name)
                
        # Define output path, preserving structure and changing extension
        output_path = (output_dir / rel_path).with_suffix('.png')
        
        # Determine CRF/Preset for this specific image
        img_crf, img_preset = crf, preset # Start with manual/default args
        if auto_detect:
            params = extract_params_from_filename(img_path.name)
            if params:
                img_crf, img_preset = params # Override with detected params
        
        # Find corresponding HQ image if in test mode
        hq_path = None
        if test_mode and hq_dir:
            # HQ path logic: hq_dir / (img_name - suffixes) .png
            # e.g., lq/crf_45/preset_4/img_001_crf45_p4.avif -> hq/img_001.png
            hq_base_name = img_path.stem.split('_crf')[0] # Get 'image_001'
            
            # Search recursively in hq_dir for a matching base name
            potential_hq = list(hq_dir.glob(f"**/{hq_base_name}.*"))
            if potential_hq:
                hq_path = potential_hq[0] # Take first match
            else:
                logger.warning(f"No HQ match found for {img_path.name} (looked for {hq_base_name}.*)")

        # Process the single image
        metrics = process_single_image(
            restorer, img_path, output_path, img_crf, img_preset,
            auto_detect, dry_run, overwrite, center_crop, use_tta, hq_path
        )
        
        if metrics:
            # Log metrics per image and add to list
            tqdm.write(
                f"Metrics for {rel_path.name}: "
                f"L2(LQ): {metrics.lq_loss:.4f}, "
                f"L2(Restored): {metrics.restored_loss:.4f}, "
                f"Improvement: {metrics.improvement_percent:.2f}%"
            )
            metrics_list.append(metrics)
            if img_crf is not None:
                per_crf_metrics[img_crf].append(metrics)
    
    # --- TOTALS (AFTER LOOP) ---
    end_time = time.perf_counter()
    total_time = end_time - start_time
    total_images = len(image_files)
    avg_fps = total_images / total_time if total_time > 0 else 0.0

    # Print summary metrics if any were computed
    if metrics_list:
        avg_lq_loss = np.mean([m.lq_loss for m in metrics_list])
        avg_restored_loss = np.mean([m.restored_loss for m in metrics_list])
        avg_improvement = np.mean([m.improvement for m in metrics_list])
        avg_improvement_pct = (avg_improvement / avg_lq_loss * 100.0) if avg_lq_loss > 1e-9 else 0.0

        logger.info("\n" + "=" * 60)
        logger.info("INFERENCE SUMMARY")
        logger.info("=" * 60)

        logger.info("--- 🚀 Overall Performance ---")
        logger.info(f"Total images processed: {total_images}")
        logger.info(f"Total time:           {total_time:.2f} seconds")
        logger.info(f"Average FPS:          {avg_fps:.2f} img/sec")

        logger.info("\n--- 📊 Total Average Metrics ---")
        logger.info(f"Images with metrics:  {len(metrics_list)}")
        logger.info(f"Average LQ Loss (MSE):       {avg_lq_loss:.6f}")
        logger.info(f"Average Restored Loss (MSE): {avg_restored_loss:.6f}")
        logger.info(f"Average Improvement (MSE):   {avg_improvement:.6f}")
        logger.info(f"Improvement % (vs MSE):      {avg_improvement_pct:.2f}%")
        
        if per_crf_metrics:
            logger.info("\n--- 📈 Metrics per CRF ---")
            for crf_val in sorted(per_crf_metrics.keys()):
                crf_metrics = per_crf_metrics[crf_val]
                count = len(crf_metrics)
                avg_lq = np.mean([m.lq_loss for m in crf_metrics])
                avg_restored = np.mean([m.restored_loss for m in crf_metrics])
                avg_improve_pct = np.mean([m.improvement_percent for m in crf_metrics])
                
                logger.info(f"  CRF {crf_val} (n={count}):")
                logger.info(f"    Avg LQ Loss:       {avg_lq:.4f}")
                logger.info(f"    Avg Restored Loss: {avg_restored:.4f}")
                logger.info(f"    Avg Improvement %: {avg_improve_pct:.2f}%")

        logger.info("=" * 60 + "\n")
    
    else:
        # Print summary even if no metrics were calculated
        logger.info("\n" + "=" * 60)
        logger.info("INFERENCE SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Total images processed: {total_images}")
        logger.info(f"Total time:           {total_time:.2f} seconds")
        logger.info(f"Average FPS:          {avg_fps:.2f} img/sec")
        if hq_dir:
            logger.warning("No metrics were computed. Ensure HQ files are correctly named and found.")
        logger.info("=" * 60 + "\n")


# ==============================================================================
# SECTION 6: Command-Line Interface
# ==============================================================================

def main():
    """Parses arguments and orchestrates the inference process."""
    parser = argparse.ArgumentParser(
        description="Unified inference script for AV1 Artifact Restoration (v3.1)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # --- Required Arguments ---
    parser.add_argument('--checkpoint', '-c', required=True, 
                       help='Path to model checkpoint (e.g., "best.pth", "checkpoints/run_name/best.pth").')
    
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('--input', '-i', type=str, 
                            help='Path to a single input image.')
    input_group.add_argument('--input_dir', '-d', type=str, 
                            help='Path to an input directory.')

    # --- Output Arguments ---
    parser.add_argument('--output', '-o', type=str, 
                       help='Path to save the single restored image.')
    parser.add_argument('--output_dir', '-od', type=str, 
                       help='Directory to save restored images (required for directory mode).')
    
    # --- Parameter Control ---
    parser.add_argument('--crf', type=int, 
                       help='Manual CRF value. In --test mode, acts as a filter.')
    parser.add_argument('--preset', type=int, 
                       help='Manual preset value. In --test mode, acts as a filter. Defaults to 4 if not set in manual mode.')
    parser.add_argument('--auto', action='store_true', 
                       help='Auto-detect CRF/preset from filenames (e.g., "_crfXX_pY"). Overrides manual --crf/--preset.')
    
    # --- Test Mode ---
    parser.add_argument('--test', action='store_true', 
                       help='Enable test mode for structured directories (lq/crf_*/preset_*/).')
    parser.add_argument('--hq_dir', type=str,
                       help='Path to HQ directory for computing metrics in test mode.')

    # --- General Options ---
    parser.add_argument('--device', default='auto', 
                       choices=['auto', 'cuda', 'mps', 'cpu'], 
                       help='Compute device.')
    parser.add_argument('--tile', type=int, default=512, 
                       help='Tile size for large images to save memory.')
    parser.add_argument('--overlap', type=int, default=64, 
                       help='Overlap between tiles for seamless stitching.')
    parser.add_argument('--crop', action='store_true',
                        help='Only process center tile of image (faster for large images)')
    parser.add_argument('--dry_run', action='store_true', 
                       help="Print actions without processing files.")
    parser.add_argument('--overwrite', action='store_true', 
                       help="Overwrite existing files in the output directory.")
    parser.add_argument('--tta', action='store_true',
                        help="Enable Test-Time Augmentation (TTA) for higher quality (8x slower).")
    
    args = parser.parse_args()

    # --- Argument Validation ---
    if args.input_dir and not args.output_dir:
        parser.error("--output_dir (-od) is required when using --input_dir (-d).")
    if args.input and not args.output:
        parser.error("--output (-o) is required when using --input (-i).")
    
    # Logic for manual mode (not auto or test)
    if not args.auto and not args.test:
        if args.crf is None:
            parser.error("Must provide --crf for manual mode (when not using --auto or --test).")
        if args.preset is None:
            logger.warning("No --preset specified in manual mode, defaulting to 4.")
            args.preset = 4
            
    if args.tta:
        logger.info("🔥 Test-Time Augmentation (TTA) enabled. This will be ~8x slower but yield higher quality.")

    # --- Execution ---
    try:
        restorer = AV1Restorer(args.checkpoint, args.device, args.tile, args.overlap)

        if args.input:
            # Single image processing
            hq_path = None
            if args.hq_dir:
                hq_path_obj = Path(args.hq_dir)
                # Try to find match by stem
                potential_hq = list(hq_path_obj.glob(f"**/{Path(args.input).stem.split('_crf')[0]}.*"))
                if potential_hq:
                    hq_path = potential_hq[0]
                else:
                    logger.warning(f"No HQ match found for {Path(args.input).name}")

            metrics = process_single_image(
                restorer, Path(args.input).expanduser(), Path(args.output).expanduser(), 
                args.crf, args.preset, args.auto, args.dry_run, args.overwrite, args.crop, args.tta, hq_path
            )
            
            if metrics:
                logger.info(f"\nMetrics for {Path(args.input).name}:")
                logger.info(f"  LQ Loss (MSE):       {metrics.lq_loss:.6f}")
                logger.info(f"  Restored Loss (MSE): {metrics.restored_loss:.6f}")
                logger.info(f"  Improvement (MSE):   {metrics.improvement:.6f}")
                logger.info(f"  Improvement %:       {metrics.improvement_percent:.2f}%")

        else:
            # Directory processing
            hq_dir_path = Path(args.hq_dir).expanduser() if args.hq_dir else None
            process_directory(
                restorer, Path(args.input_dir).expanduser(), Path(args.output_dir).expanduser(), 
                args.crf, args.preset, args.auto, args.test, 
                args.dry_run, args.overwrite, args.crop, args.tta, hq_dir_path
            )
        
        logger.info("✓ Inference complete!")

    except Exception as e:
        logger.error(f"❌ An unexpected error occurred: {e}", exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    main()

"""
These commands assume you are in your root project directory.

1. Test a Single Image (Auto-Detect CRF/Preset)
This is a great way to quickly check a single file. The --auto flag will read the CRF and preset from the filename (e.g., ...img110_crf30_p4.avif).

python scripts/restore_av1.py \
    --checkpoint checkpoints/conditional_unet_tiny_M4/best.pth \
    --input /Users/soham/Documents/aura/av1_data/test/lq/crf_30/preset_4/YOUR_IMAGE_NAME_crf30_p4.avif \
    --output /Users/soham/Documents/aura/results/tiny_test/restored_image.png \
    --auto \
    --device mps

2. Process a Full Directory (Auto-Detect CRF/Preset)
This will find all images in the input directory, restore them using auto-detected parameters, and save them to the output directory while keeping the same folder structure.

python scripts/restore_av1.py \
    --checkpoint checkpoints/conditional_unet_tiny_M4/best.pth \
    --input_dir /Users/soham/Documents/aura/av1_data/test/lq \
    --output_dir /Users/soham/Documents/aura/results/tiny_test_all_images \
    --auto \
    --device mps

3. Run a Full Test with Metrics (Recommended)
This uses the --test and --hq_dir flags to process the structured lq directory and compare each restored image against its corresponding high-quality version in the hq directory, printing a summary of the L2 (MSE) improvement.

python scripts/restore_av1.py \
    --checkpoint checkpoints/conditional_unet_tiny_M4/best.pth \
    --input_dir /Users/soham/Documents/aura/av1_data/test/lq \
    --output_dir /Users/soham/Documents/aura/results/tiny_test_full_metrics \
    --hq_dir /Users/soham/Documents/aura/av1_data/test/hq \
    --test \
    --auto \
    --device mps

4. Run Full Test with TTA (Highest Quality)
Same as #3, but adds the --tta flag. This will be ~8x slower but will produce the best possible images.

python scripts/restore_av1.py \
    --checkpoint checkpoints/conditional_unet_tiny_M4/best.pth \
    --input_dir /Users/soham/Documents/aura/av1_data/test/lq \
    --output_dir /Users/soham/Documents/aura/results/tiny_test_full_metrics_TTA \
    --hq_dir /Users/soham/Documents/aura/av1_data/test/hq \
    --test \
    --auto \
    --device mps \
    --tta
"""