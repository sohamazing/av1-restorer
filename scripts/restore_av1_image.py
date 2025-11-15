#!/usr/bin/env python3
"""
restore_image.py - Image Inference Script for AV1 Artifact Restoration
======================================================================

Production-ready inference script for processing single images or entire
directories with comprehensive metrics and test mode support.

Features:
    ✓ Single-image and batch directory processing
    ✓ Automatic CRF/Preset detection from filenames
    ✓ Test mode with per-CRF metrics
    ✓ Memory-efficient tiling for large images
    ✓ Test-Time Augmentation (TTA) support
    ✓ FPS reporting and quality metrics
    ✓ Structured test directory handling

Usage Examples:
    # Single image with auto-detection
    python scripts/restore_image.py -c checkpoints/best.pth \
        -i input_crf45_p4.avif -o output.png --auto
    
    # Batch processing with manual CRF
    python scripts/restore_image.py -c checkpoints/best.pth \
        -d ./test/lq -od ./results --crf 45 --preset 4
    
    # Full test with metrics
    python scripts/restore_image.py -c checkpoints/best.pth \
        -d ./test/lq -od ./results --hq_dir ./test/hq \
        --test --auto --device cuda
    
    # Highest quality with TTA
    python scripts/restore_image.py -c checkpoints/best.pth \
        -d ./test/lq -od ./results --hq_dir ./test/hq \
        --test --auto --tta --device cuda

Author: Soham Mukherjee
Version: 5.0 (Production Final)
License: MIT
"""

import argparse
import logging
import re
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Optional, Tuple, List
from dataclasses import dataclass

import numpy as np
from PIL import Image
from tqdm import tqdm

# ==============================================================================
# SECTION 1: Project Setup
# ==============================================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

try:
    from av1_restorer.unified_inference_av1_restorer import AV1RestorerInference, compute_mse
except ImportError as e:
    print(f"✗ Import error: {e}", file=sys.stderr)
    print("  Ensure av1_restorer/unified_inference_av1_restorer.py exists", file=sys.stderr)
    sys.exit(1)

# ==============================================================================
# SECTION 2: Logging & Constants
# ==============================================================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger("AV1Restorer.Image")

# Regex for extracting CRF and preset from filenames
# Matches: "image_001_crf45_p4.avif" -> (45, 4)
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


# ==============================================================================
# SECTION 3: Utility Functions
# ==============================================================================

def extract_params_from_filename(name: str) -> Optional[Tuple[int, int]]:
    """
    Extract CRF and preset from filename.
    
    Args:
        name: Filename to parse
        
    Returns:
        Tuple of (crf, preset) if found, None otherwise
        
    Example:
        >>> extract_params_from_filename("img_crf35_p4.avif")
        (35, 4)
    """
    match = FILENAME_PATTERN.search(name)
    return (int(match.group(1)), int(match.group(2))) if match else None


def find_images(path: Path) -> List[Path]:
    """
    Recursively find all image files in directory.
    
    Args:
        path: Root directory to search
        
    Returns:
        Sorted list of image file paths
    """
    return sorted([
        p for p in path.rglob('*') 
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    ])


def load_image(img_path: Path) -> Image.Image:
    """
    Load image from various formats including .npy.
    
    Args:
        img_path: Path to image file
        
    Returns:
        PIL Image in RGB format
    """
    if img_path.suffix.lower() == '.npy':
        arr = np.load(img_path)
        # Handle .npy saved as [0,1] float or [0,255] uint8
        if arr.dtype in [np.float32, np.float64]:
            return Image.fromarray((arr * 255).astype(np.uint8))
        else:
            return Image.fromarray(arr.astype(np.uint8))
    else:
        return Image.open(img_path).convert('RGB')


def find_hq_match(lq_path: Path, hq_dir: Path) -> Optional[Path]:
    """
    Find corresponding HQ image for a given LQ image.
    
    Args:
        lq_path: Path to LQ image
        hq_dir: Directory containing HQ images
        
    Returns:
        Path to HQ image if found, None otherwise
    """
    # Extract base name (remove _crfXX_pY suffix)
    base_name = lq_path.stem.split('_crf')[0]
    
    # Search recursively in hq_dir
    matches = list(hq_dir.glob(f"**/{base_name}.*"))
    return matches[0] if matches else None


def align_images_for_metrics(
    lq_img: Image.Image,
    restored_img: Image.Image,
    hq_img: Image.Image,
    center_crop: bool
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Align LQ, restored, and HQ images to same size for fair metrics.
    
    Args:
        lq_img: Low quality PIL Image
        restored_img: Restored PIL Image
        hq_img: High quality PIL Image
        center_crop: Whether center crop was used
        
    Returns:
        Tuple of (lq_np, restored_np, hq_np) as numpy arrays
    """
    # Ensure HQ matches restored size
    if hq_img.size != restored_img.size:
        if center_crop:
            # Crop HQ to match restored center region
            crop_w, crop_h = restored_img.size
            start_w = (hq_img.width - crop_w) // 2
            start_h = (hq_img.height - crop_h) // 2
            hq_img = hq_img.crop((
                start_w, start_h,
                start_w + crop_w, start_h + crop_h
            ))
        else:
            # Resize HQ to match restored
            hq_img = hq_img.resize(restored_img.size, Image.LANCZOS)
    
    # Ensure LQ matches restored size
    if lq_img.size != restored_img.size:
        if center_crop:
            # Crop LQ to match restored center region
            crop_w, crop_h = restored_img.size
            start_w = (lq_img.width - crop_w) // 2
            start_h = (lq_img.height - crop_h) // 2
            lq_img = lq_img.crop((
                start_w, start_h,
                start_w + crop_w, start_h + crop_h
            ))
        else:
            # Resize LQ to match restored
            lq_img = lq_img.resize(restored_img.size, Image.LANCZOS)
    
    # Convert to numpy arrays
    return np.array(lq_img), np.array(restored_img), np.array(hq_img)


# ==============================================================================
# SECTION 4: Processing Functions
# ==============================================================================

def process_single_image(
    model: AV1RestorerInference,
    img_path: Path,
    output_path: Path,
    crf: Optional[int],
    preset: Optional[int],
    auto_detect: bool,
    dry_run: bool,
    overwrite: bool,
    center_crop: bool,
    use_tta: bool,
    hq_path: Optional[Path] = None
) -> Optional[ImageMetrics]:
    """
    Process a single image file with comprehensive error handling.
    
    Args:
        model: Configured AV1RestorerInference instance
        img_path: Path to input image
        output_path: Path to save restored image
        crf: Manual CRF override
        preset: Manual preset override
        auto_detect: Enable auto-detection from filename
        dry_run: If True, only log actions without processing
        overwrite: If True, overwrite existing output files
        center_crop: If True, only process center tile
        use_tta: If True, enable Test-Time Augmentation
        hq_path: Optional path to HQ image for metrics
        
    Returns:
        ImageMetrics if HQ image provided, None otherwise
    """
    # Skip if output exists and not overwriting
    if not overwrite and output_path.exists():
        logger.debug(f"Skip (exists): {output_path.name}")
        return None
    
    # Auto-detect parameters from filename if enabled
    if auto_detect:
        params = extract_params_from_filename(img_path.name)
        if params:
            crf = crf or params[0]
            preset = preset or params[1]
    
    # Validate CRF
    if crf is None:
        logger.warning(f"Skip: {img_path.name} (no CRF specified/detected)")
        return None
    
    # Default preset
    preset = preset or 4
    
    # Dry run mode
    if dry_run:
        logger.info(
            f"[DRY RUN] {img_path.name} -> {output_path.name} "
            f"(CRF={crf}, P={preset}, TTA={use_tta})"
        )
        return None
    
    # Process image
    try:
        # Load input
        lq_img = load_image(img_path)
        
        # Restore
        restored_img = model.restore(lq_img, crf, preset, center_crop, use_tta)
        
        # Save output
        output_path.parent.mkdir(parents=True, exist_ok=True)
        restored_img.save(output_path, quality=95, optimize=True)
        
        # Compute metrics if HQ available
        if hq_path and hq_path.exists():
            hq_img = load_image(hq_path)
            
            # Align images for fair comparison
            lq_np, restored_np, hq_np = align_images_for_metrics(
                lq_img, restored_img, hq_img, center_crop
            )
            
            # Compute MSE metrics
            lq_mse = compute_mse(lq_np, hq_np)
            restored_mse = compute_mse(restored_np, hq_np)
            improvement = lq_mse - restored_mse
            improvement_pct = (improvement / lq_mse * 100.0) if lq_mse > 1e-9 else 0.0
            
            return ImageMetrics(lq_mse, restored_mse, improvement, improvement_pct)
        
        return None
    
    except Exception as e:
        logger.error(f"Failed: {img_path.name} - {e}")
        return None


def process_directory(
    model: AV1RestorerInference,
    input_dir: Path,
    output_dir: Path,
    crf: Optional[int],
    preset: Optional[int],
    auto_detect: bool,
    test_mode: bool,
    dry_run: bool,
    overwrite: bool,
    center_crop: bool,
    use_tta: bool,
    hq_dir: Optional[Path] = None
) -> None:
    """
    Process all images in a directory with optional test mode metrics.
    
    Test mode expects structure:
        input_dir/
        ├── crf_23/
        │   ├── preset_4/
        │   │   ├── img_001_crf23_p4.avif
        │   │   └── ...
        │   └── ...
        └── ...
    
    Args:
        model: Configured AV1RestorerInference instance
        input_dir: Root directory containing images
        output_dir: Directory to save restored images
        crf: Manual CRF (acts as filter in test mode)
        preset: Manual preset (acts as filter in test mode)
        auto_detect: Enable auto-detection from filenames
        test_mode: Enable structured directory processing
        dry_run: If True, only log actions
        overwrite: If True, overwrite existing files
        center_crop: If True, only process center tiles
        use_tta: If True, enable TTA
        hq_dir: Optional directory with HQ images for metrics
    """
    # Build list of images to process
    if test_mode:
        logger.info(f"Test mode: Scanning {input_dir}")
        search_paths = []
        
        # Apply filters based on CRF and Preset
        if crf is not None and preset is not None:
            # Specific CRF and Preset
            path = input_dir / f"crf_{crf}" / f"preset_{preset}"
            if path.exists():
                search_paths.append(path)
                logger.info(f"  Filter: crf_{crf}/preset_{preset}")
            else:
                logger.warning(f"  Path not found: {path}")
        
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
                logger.warning(f"  CRF dir not found: {crf_dir}")
        
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
            logger.info("  No filter: processing all preset directories")
        
        # Collect all images from search paths
        image_files = []
        for path in search_paths:
            if path.exists():
                found = find_images(path)
                image_files.extend(found)
    
    else:
        # Standard mode: recursive search
        logger.info(f"Standard mode: Scanning {input_dir}")
        image_files = find_images(input_dir)
    
    if not image_files:
        logger.warning("No images found")
        return
    
    logger.info(f"Found {len(image_files)} images to process")
    
    # Process all images
    metrics_list: List[ImageMetrics] = []
    per_crf_metrics = defaultdict(list)
    start_time = time.perf_counter()
    
    for img_path in tqdm(image_files, desc="Restoring", unit="img"):
        # Preserve directory structure
        try:
            rel_path = img_path.relative_to(input_dir)
        except ValueError:
            rel_path = Path(img_path.name)
        
        output_path = (output_dir / rel_path).with_suffix('.png')
        
        # Determine CRF/Preset for this image
        img_crf, img_preset = crf, preset
        if auto_detect:
            params = extract_params_from_filename(img_path.name)
            if params:
                img_crf = img_crf or params[0]
                img_preset = img_preset or params[1]
        
        # Find HQ match if in test mode
        hq_path = None
        if test_mode and hq_dir:
            hq_path = find_hq_match(img_path, hq_dir)
            if not hq_path:
                logger.debug(f"No HQ match for {img_path.name}")
        
        # Process image
        metrics = process_single_image(
            model, img_path, output_path, img_crf, img_preset,
            auto_detect, dry_run, overwrite, center_crop, use_tta, hq_path
        )
        
        # Collect metrics
        if metrics:
            tqdm.write(
                f"  {rel_path.name}: "
                f"L2(LQ)={metrics.lq_loss:.4f}, "
                f"L2(Restored)={metrics.restored_loss:.4f}, "
                f"Δ={metrics.improvement_percent:+.2f}%"
            )
            metrics_list.append(metrics)
            if img_crf:
                per_crf_metrics[img_crf].append(metrics)
    
    # Print summary
    end_time = time.perf_counter()
    total_time = end_time - start_time
    total_images = len(image_files)
    avg_fps = total_images / total_time if total_time > 0 else 0
    
    logger.info("\n" + "="*70)
    logger.info("INFERENCE SUMMARY")
    logger.info("="*70)
    logger.info(f"Total images:  {total_images}")
    logger.info(f"Total time:    {total_time:.2f}s")
    logger.info(f"Average FPS:   {avg_fps:.2f} img/s")
    
    if metrics_list:
        avg_lq = np.mean([m.lq_loss for m in metrics_list])
        avg_restored = np.mean([m.restored_loss for m in metrics_list])
        avg_improve = np.mean([m.improvement for m in metrics_list])
        avg_improve_pct = (avg_improve / avg_lq * 100.0) if avg_lq > 1e-9 else 0.0
        
        logger.info(f"\n--- Overall Metrics (n={len(metrics_list)}) ---")
        logger.info(f"Avg LQ Loss (MSE):       {avg_lq:.6f}")
        logger.info(f"Avg Restored Loss (MSE): {avg_restored:.6f}")
        logger.info(f"Avg Improvement (MSE):   {avg_improve:.6f}")
        logger.info(f"Improvement %:           {avg_improve_pct:+.2f}%")
        
        if per_crf_metrics:
            logger.info("\n--- Per-CRF Metrics ---")
            for crf_val in sorted(per_crf_metrics.keys()):
                crf_m = per_crf_metrics[crf_val]
                n = len(crf_m)
                lq = np.mean([m.lq_loss for m in crf_m])
                restored = np.mean([m.restored_loss for m in crf_m])
                improve = np.mean([m.improvement_percent for m in crf_m])
                
                logger.info(f"  CRF {crf_val:2d} (n={n:3d}): "
                          f"LQ={lq:.4f}, Restored={restored:.4f}, Δ={improve:+.2f}%")
    
    logger.info("="*70 + "\n")


# ==============================================================================
# SECTION 5: CLI
# ==============================================================================

def main():
    """Parse arguments and orchestrate inference."""
    parser = argparse.ArgumentParser(
        description="AV1 Image Restoration (v5.0)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Required
    parser.add_argument('-c', '--checkpoint', required=True,
                       help='Path to model checkpoint (.pth)')
    
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('-i', '--input',
                            help='Single input image')
    input_group.add_argument('-d', '--input_dir',
                            help='Input directory for batch processing')
    
    # Output
    parser.add_argument('-o', '--output',
                       help='Output path for single image')
    parser.add_argument('-od', '--output_dir',
                       help='Output directory (required for batch mode)')
    
    # Parameters
    parser.add_argument('--crf', type=int,
                       help='CRF value (filter in test mode)')
    parser.add_argument('--preset', type=int,
                       help='Preset value (filter in test mode, default: 4)')
    parser.add_argument('--auto', action='store_true',
                       help='Auto-detect CRF/preset from filenames (_crfXX_pY)')
    
    # Test mode
    parser.add_argument('--test', action='store_true',
                       help='Test mode (structured dirs: lq/crf_*/preset_*/)')
    parser.add_argument('--hq_dir',
                       help='HQ directory for computing metrics')
    
    # Options
    parser.add_argument('--device', default='auto',
                       choices=['auto', 'cuda', 'mps', 'cpu'],
                       help='Compute device')
    parser.add_argument('--tile', type=int, default=512,
                       help='Tile size for large images')
    parser.add_argument('--overlap', type=int, default=64,
                       help='Tile overlap for seamless stitching')
    parser.add_argument('--crop', action='store_true',
                       help='Only process center tile (faster)')
    parser.add_argument('--tta', action='store_true',
                       help='Test-Time Augmentation (8x slower, +quality)')
    parser.add_argument('--dry_run', action='store_true',
                       help='Print actions without processing')
    parser.add_argument('--overwrite', action='store_true',
                       help='Overwrite existing output files')
    
    args = parser.parse_args()
    
    # Validation
    if args.input_dir and not args.output_dir:
        parser.error("--output_dir required with --input_dir")
    if args.input and not args.output:
        parser.error("--output required with --input")
    
    if not args.auto and not args.test and args.crf is None:
        parser.error("Must provide --crf in manual mode (or use --auto/--test)")
    
    if args.tta:
        logger.info("🔥 TTA enabled (8x slower, higher quality)")
    
    # Execute
    try:
        model = AV1RestorerInference(args.checkpoint, args.device, args.tile, args.overlap)
        
        if args.input:
            # Single image
            hq_path = None
            if args.hq_dir:
                hq_path = find_hq_match(Path(args.input), Path(args.hq_dir))
            
            metrics = process_single_image(
                model,
                Path(args.input).expanduser(),
                Path(args.output).expanduser(),
                args.crf, args.preset, args.auto,
                args.dry_run, args.overwrite,
                args.crop, args.tta, hq_path
            )
            
            if metrics:
                logger.info(f"\nMetrics:")
                logger.info(f"  LQ Loss:       {metrics.lq_loss:.6f}")
                logger.info(f"  Restored Loss: {metrics.restored_loss:.6f}")
                logger.info(f"  Improvement:   {metrics.improvement_percent:+.2f}%")
        
        else:
            # Directory
            hq_dir_path = Path(args.hq_dir).expanduser() if args.hq_dir else None
            process_directory(
                model,
                Path(args.input_dir).expanduser(),
                Path(args.output_dir).expanduser(),
                args.crf, args.preset, args.auto, args.test,
                args.dry_run, args.overwrite,
                args.crop, args.tta, hq_dir_path
            )
        
        logger.info("✓ Complete!")
    
    except Exception as e:
        logger.error(f"✗ Error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    main()