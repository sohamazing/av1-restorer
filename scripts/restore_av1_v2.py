#!/usr/bin/env python3
"""
restore_av1.py — Unified Inference Script for AV1 U-Net Restorer (v2.1)

Features:
- Single-image and directory batch modes
- Auto-detect CRF/preset from filenames (supports --auto)
- Test-mode for structured test directories: lq/crf_{N}/preset_{M}/...
- Manual overrides: apply a single CRF/preset to all images in a directory
- Preserves relative directory structure when saving
- Safe device selection (auto -> cuda, mps, cpu)
- Tiling support for large images
- Handles checkpoint loading by reading config from the checkpoint file
- Test mode computes L2 loss metrics (restored vs HQ, LQ vs HQ)
- Includes --dry-run and --overwrite flags for safe testing

Author: Soham Mukherjee
"""

import argparse
import logging
import re
import sys
from pathlib import Path
from typing import Optional, Tuple, List, Dict
from tqdm import tqdm
import numpy as np
from PIL import Image
import torch
import torchvision.transforms.v2 as T

# --- Project Setup ---
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

# --- SOTA Model Imports ---
try:
    # Use the V2 factory function, which contains the correct architectures
    from av1_restorer.models.av1_conditional_unet_restorer_v2 import create_av1_restorer
except ImportError:
    logging.error("Failed to import 'create_av1_restorer' from 'av1_restorer.models.av1_conditional_unet_restorer_v2'.")
    logging.error("Please ensure the file exists and 'project_root' is correct.")
    sys.exit(1)
# ------------------------

# --- Logging Configuration ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger("AV1Restorer")

# --- Utility Functions ---
FILENAME_PATTERN = re.compile(r"_crf(\d+)_p(\d+)\.")

def extract_params_from_filename(name: str) -> Optional[Tuple[int, int]]:
    """Parses a filename (e.g., 'img_crf35_p4.avif') to find CRF and preset values."""
    m = FILENAME_PATTERN.search(name)
    return (int(m.group(1)), int(m.group(2))) if m else None

def find_images(path: Path, exts=('.avif', '.png', '.jpg', '.jpeg')) -> List[Path]:
    """Recursively finds all image files with given extensions in a directory."""
    return sorted([p for p in path.rglob('*') if p.suffix.lower() in exts])

def setup_device(dev_str: str) -> torch.device:
    """Auto-selects the best available hardware device (CUDA > MPS > CPU)."""
    if dev_str == 'auto':
        if torch.cuda.is_available(): 
            logger.info("Auto-selected device: CUDA")
            return torch.device('cuda')
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            logger.info("Auto-selected device: MPS")
            return torch.device('mps')
        logger.info("Auto-selected device: CPU")
        return torch.device('cpu')
    logger.info(f"User-selected device: {dev_str}")
    return torch.device(dev_str)

def compute_l2_loss(img1: np.ndarray, img2: np.ndarray) -> float:
    """Compute L2 (MSE) loss between two images."""
    return np.mean((img1.astype(float) - img2.astype(float)) ** 2)

# --- Main Inference Engine ---
class AV1Restorer:
    """A wrapper class to load the model and handle all inference logic."""

    def __init__(self, checkpoint: str, device='auto', tile=512, overlap=64):
        self.device = setup_device(device)
        self.tile_size = tile
        self.tile_overlap = overlap
        self.to_tensor = T.Compose([T.ToImage(), T.ToDtype(torch.float32, scale=True)])
        
        # Load the model and its training configuration from the checkpoint file.
        # This now uses the SOTA factory function.
        self.model, self.config = self._load_model_from_factory(checkpoint)
        self.model.eval().to(self.device)
        
        # Configure normalization based on how the loaded model was trained.
        norm_range = self.config.get('dataset', {}).get('norm_range', [-1, 1])
        self.norm_range = tuple(norm_range) if isinstance(norm_range, list) else norm_range
        
        if self.norm_range == (-1, 1):
            self.normalize = T.Normalize(mean=[0.5]*3, std=[0.5]*3)
            self.denormalize = lambda x: (x + 1) / 2
        else:
            self.normalize = lambda x: x
            self.denormalize = lambda x: x
        
        logger.info(f"✓ Model loaded from: {checkpoint}")
        logger.info(f"  Device: {self.device}")
        logger.info(f"  Norm range: {self.norm_range}")
        logger.info(f"  Tile size: {tile}×{tile} (overlap: {overlap})")

    def _load_model_from_factory(self, ckpt_path: str) -> Tuple[torch.nn.Module, dict]:
        """
        SOTA FIX: Loads a checkpoint, reconstructs the model using the 
        create_av1_restorer factory, and loads the weights.
        """
        path = Path(ckpt_path).expanduser()
        
        # Handle checkpoint path resolution
        if not path.is_file():
            # Check relative to project root
            potential_path = project_root / ckpt_path
            if potential_path.is_file():
                path = potential_path
            else:
                 raise FileNotFoundError(f"Checkpoint not found at '{ckpt_path}' or '{potential_path}'")
        
        logger.info(f"Loading checkpoint: {path}")
        data = torch.load(path, map_location='cpu')
        config = data.get('config')
        if not config:
            raise ValueError("Checkpoint is missing the 'config' dictionary, cannot reconstruct model.")

        # --- SOTA FIX: Reconstruct model using the *same factory* as training ---
        # This ensures the architecture is 100% correct
        logger.info("Reconstructing model from saved config...")
        model = create_av1_restorer(
            size=config['model']['size'],
            crf_range=tuple(config['dataset']['crf_range']),
            preset_range=tuple(config['dataset']['preset_range']),
            norm_range=tuple(config['dataset'].get('norm_range', [-1, 1]))
        )
        # ---------------------------------------------------------------------
        
        # Prioritize loading EMA weights if available, as they often give better results.
        weights = None
        if 'ema_state_dict' in data:
            logger.info("  Using EMA weights (from 'ema_state_dict')")
            weights = data['ema_state_dict']
        elif 'ema_shadow' in data: # Fallback for older/different save formats
            logger.info("  Using EMA weights (from 'ema_shadow')")
            weights = data['ema_shadow']
        elif 'model_state_dict' in data:
            logger.info("  Using standard model weights (from 'model_state_dict')")
            weights = data['model_state_dict']
        else:
            raise ValueError("Could not find 'ema_state_dict' or 'model_state_dict' in checkpoint.")
        
        # Load with strict=False to handle any minor mismatches gracefully
        missing_keys, unexpected_keys = model.load_state_dict(weights, strict=False)
        
        if missing_keys:
            logger.warning(f"  Model is missing keys: {missing_keys}")
        if unexpected_keys:
            logger.warning(f"  Ignored unexpected keys in checkpoint: {unexpected_keys}")
        
        total_params = sum(p.numel() for p in model.parameters())
        logger.info(f"  Loaded Model: {config['model']['size']} ({total_params/1e6:.2f}M parameters)")
        
        return model, config

    @torch.no_grad()
    def restore(self, img: Image.Image, crf: int, preset: int, crop: bool = False) -> Image.Image:
        """
        Restores a single PIL Image with optional center cropping.
        
        Args:
            img: Input PIL Image
            crf: CRF value for conditioning
            preset: Preset value for conditioning
            crop: If True, only restore center tile (faster for large images)
        """
        # Prepare image: convert to tensor, normalize, and move to device
        x = self.normalize(self.to_tensor(img)).unsqueeze(0).to(self.device)
        
        # Prepare conditioning inputs
        crf_t = torch.tensor([[float(crf)]], device=self.device)
        pre_t = torch.tensor([[float(preset)]], device=self.device)
        
        # Get the correct inference function from the model
        inference_fn = getattr(self.model, 'inference', self.model.forward)
        
        # Run inference with tiling support
        # The model's 'inference' function intelligently handles CRF-only vs CRF+Preset
        # by checking its own cond_dim. We can safely pass both.
        out = inference_fn(
            x, crf_t, pre_t,
            tile_size=self.tile_size,
            tile_overlap=self.tile_overlap,
            center_crop=crop
        )
        
        # Convert back to PIL Image
        out = torch.clamp(self.denormalize(out), 0, 1).squeeze(0).cpu().permute(1, 2, 0).numpy()
        return Image.fromarray((out * 255).astype(np.uint8))


# --- Main Processing Logic ---
def process_single_image(
    model: AV1Restorer, 
    img_path: Path, 
    out_path: Path,
    crf: Optional[int], 
    preset: Optional[int],
    auto: bool, 
    dry_run: bool = False, 
    overwrite: bool = False,
    crop: bool = False,
    hq_path: Optional[Path] = None
) -> Optional[Dict[str, float]]:
    """
    Orchestrates the restoration of a single image file.
    Handles parameter detection, dry-run, execution, and metrics.
    
    Returns:
        Dict with metrics if hq_path provided, else None
    """
    if not overwrite and out_path.exists():
        logger.debug(f"Skip (exists): {out_path}")
        return None

    # Auto-detect parameters from filename if needed.
    if auto and (crf is None or preset is None):
        params = extract_params_from_filename(img_path.name)
        if params: 
            crf, preset = params
    
    # Final check for parameters
    if crf is None:
        logger.warning(f"Skipping (CRF not specified or detected): {img_path}")
        return None
    if preset is None:
        logger.warning(f"No preset specified or detected for {img_path.name}, defaulting to 4.")
        preset = 4 # Default preset

    # In dry-run mode, just log what would happen.
    if dry_run:
        logger.info(f"[DRY RUN] {img_path.name} -> {out_path.name} (CRF={crf}, Preset={preset})")
        return None

    # Execute the restoration.
    try:
        lq_img = Image.open(img_path).convert('RGB')
        restored = model.restore(lq_img, crf, preset, crop=crop)
        
        out_path.parent.mkdir(parents=True, exist_ok=True)
        restored.save(out_path, quality=95)
        
        # Compute metrics if HQ image provided (test mode)
        if hq_path and hq_path.exists():
            hq_img = Image.open(hq_path).convert('RGB')
            
            # Ensure same size
            if hq_img.size != restored.size:
                # If center cropping, crop HQ to match
                if crop:
                    crop_h, crop_w = restored.size[1], restored.size[0]
                    start_h = (hq_img.height - crop_h) // 2
                    start_w = (hq_img.width - crop_w) // 2
                    hq_img = hq_img.crop((start_w, start_h, start_w + crop_w, start_h + crop_h))
                else: # Otherwise, resize
                    logger.warning(f"Restored size {restored.size} != HQ size {hq_img.size}. Resizing HQ for metrics.")
                    hq_img = hq_img.resize(restored.size, Image.LANCZOS)
            
            # Convert to numpy arrays
            if crop: # If cropped, we need to crop LQ img too for fair comparison
                crop_h, crop_w = restored.size[1], restored.size[0]
                start_h = (lq_img.height - crop_h) // 2
                start_w = (lq_img.width - crop_w) // 2
                lq_img = lq_img.crop((start_w, start_h, start_w + crop_w, start_h + crop_h))
            
            lq_np = np.array(lq_img)
            restored_np = np.array(restored)
            hq_np = np.array(hq_img)
            
            # Compute L2 losses
            lq_loss = compute_l2_loss(lq_np, hq_np)
            restored_loss = compute_l2_loss(restored_np, hq_np)
            improvement = lq_loss - restored_loss
            
            return {
                'lq_loss': lq_loss,
                'restored_loss': restored_loss,
                'improvement': improvement
            }
        
        return None
        
    except Exception as e:
        logger.error(f"Failed to process {img_path.name}: {e}", exc_info=True)
        return None


def process_directory(
    model: AV1Restorer, 
    in_dir: Path, 
    out_dir: Path,
    crf: Optional[int], 
    preset: Optional[int], 
    auto: bool, 
    test_mode: bool, 
    dry_run: bool, 
    overwrite: bool,
    crop: bool = False,
    hq_dir: Optional[Path] = None
):
    """Finds images based on the mode and processes them."""
    
    # In test mode, filter the subdirectories based on --crf and --preset args.
    if test_mode:
        # Test mode expects: lq/crf_XX/preset_Y/*.avif structure
        logger.info(f"Test mode enabled: Searching in {in_dir}...")
        search_paths = []
        
        if crf is not None and preset is not None:
            # Specific CRF and preset
            logger.info(f"  Filtering for: crf_{crf}/preset_{preset}")
            search_paths.append(in_dir / f"crf_{crf}" / f"preset_{preset}")
        elif crf is not None:
            # All presets for specific CRF
            logger.info(f"  Filtering for: crf_{crf}/preset_*")
            crf_dir = in_dir / f"crf_{crf}"
            if crf_dir.exists():
                search_paths.extend([p for p in crf_dir.iterdir() if p.is_dir() and p.name.startswith('preset_')])
        elif preset is not None:
            # All CRFs for specific preset
            logger.info(f"  Filtering for: crf_*/preset_{preset}")
            for crf_dir in in_dir.glob("crf_*"):
                preset_dir = crf_dir / f"preset_{preset}"
                if preset_dir.exists():
                    search_paths.append(preset_dir)
        else:
            # All CRFs and presets (auto mode in test)
            logger.info("  No filter: processing all found 'preset_*' subdirectories.")
            search_paths.extend(in_dir.rglob("preset_*"))
        
        image_files = []
        for search_path in search_paths:
            if search_path.exists():
                image_files.extend(find_images(search_path))
            else:
                logger.warning(f"  Search path not found: {search_path}")
    else:
        # In normal mode, search the entire input directory.
        logger.info(f"Standard mode: Searching recursively in {in_dir}...")
        image_files = find_images(in_dir)

    if not image_files:
        logger.warning(f"No images found for the specified criteria in {in_dir}")
        return

    logger.info(f"Found {len(image_files)} images to process.")
    
    # Metrics tracking for test mode
    metrics_list = []
    
    for img_path in tqdm(image_files, desc="Restoring images"):
        # Preserve the relative path from the input base to the output.
        try:
            rel_path = img_path.relative_to(in_dir)
        except ValueError:
             # Handle case where img_path is not in in_dir (e.g., test mode search)
             # Fallback: use the 'crf_*/preset_*' structure relative to in_dir
             try:
                 rel_path = img_path.relative_to(in_dir.parent.parent)
             except ValueError:
                 rel_path = img_path.name # Last resort
                 
        out_path = (out_dir / rel_path).with_suffix('.png')
        
        # Determine CRF/Preset for this image
        img_crf, img_preset = crf, preset
        if auto: # Auto-detect from filename (overrides manual args if found)
            params = extract_params_from_filename(img_path.name)
            if params:
                img_crf, img_preset = params
        
        # Find corresponding HQ image if in test mode
        hq_path = None
        if test_mode and hq_dir:
            # HQ path logic: hq_dir / (img_name - suffixes) .png
            # e.g., lq/crf_45/preset_4/img_001_crf45_p4.avif -> hq/img_001.png
            base_name = img_path.name.split('_crf')[0]
            hq_base_name = img_path.stem.split('_crf')[0] # Get 'image_001'
            
            # Look for HQ image with same base name in root or subdirs
            potential_hq = list(hq_dir.glob(f"**/{hq_base_name}.*"))
            if potential_hq:
                hq_path = potential_hq[0] # Take first match
            else:
                logger.warning(f"No HQ match found for {img_path.name} (looked for {hq_base_name}.*)")
        
        metrics = process_single_image(
            model, img_path, out_path, img_crf, img_preset, 
            auto, dry_run, overwrite, crop, hq_path
        )
        
        if metrics:
            # Log metrics per image in test mode
            tqdm.write(f"Metrics for {rel_path}:")
            tqdm.write(f"  LQ Loss (MSE):       {metrics['lq_loss']:.6f}")
            tqdm.write(f"  Restored Loss (MSE): {metrics['restored_loss']:.6f}")
            tqdm.write(f"  Improvement:         {metrics['improvement']:.6f}")
            metrics_list.append(metrics)
    
    # Print summary metrics if in test mode
    if metrics_list:
        avg_lq_loss = np.mean([m['lq_loss'] for m in metrics_list])
        avg_restored_loss = np.mean([m['restored_loss'] for m in metrics_list])
        avg_improvement = np.mean([m['improvement'] for m in metrics_list])
        
        logger.info("\n" + "="*60)
        logger.info("TEST MODE METRICS SUMMARY")
        logger.info("="*60)
        logger.info(f"Number of images: {len(metrics_list)}")
        logger.info(f"Average LQ Loss (MSE):       {avg_lq_loss:.6f}")
        logger.info(f"Average Restored Loss (MSE): {avg_restored_loss:.6f}")
        logger.info(f"Average Improvement (MSE):   {avg_improvement:.6f}")
        if avg_lq_loss > 1e-9: # Avoid division by zero
            logger.info(f"Improvement % (vs MSE):      {(avg_improvement/avg_lq_loss)*100:.2f}%")
        logger.info("="*60 + "\n")


# --- Command-Line Interface ---
def main():
    """Parses arguments and orchestrates the inference process."""
    parser = argparse.ArgumentParser(
        description="Unified inference script for AV1 U-Net Restorer (v2)",
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
    parser.add_argument('--output_dir', type=str, 
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
    
    args = parser.parse_args()

    # --- Argument Validation ---
    if args.input_dir and not args.output_dir:
        parser.error("--output_dir is required when using --input_dir.")
    if args.input and not args.output:
        parser.error("--output is required when using --input.")
    
    # Logic for manual mode (not auto or test)
    if not args.auto and not args.test:
        if args.crf is None:
            parser.error("Must provide --crf for manual mode (when not using --auto or --test).")
        if args.preset is None:
            logger.warning("No --preset specified in manual mode, defaulting to 4.")
            args.preset = 4

    # --- Execution ---
    try:
        model = AV1Restorer(args.checkpoint, args.device, args.tile, args.overlap)

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
                model, Path(args.input), Path(args.output), 
                args.crf, args.preset, args.auto, args.dry_run, args.overwrite, args.crop, hq_path
            )
            
            if metrics:
                logger.info(f"\nMetrics for {Path(args.input).name}:")
                logger.info(f"  LQ Loss (MSE):       {metrics['lq_loss']:.6f}")
                logger.info(f"  Restored Loss (MSE): {metrics['restored_loss']:.6f}")
                logger.info(f"  Improvement (MSE):   {metrics['improvement']:.6f}")
        else:
            # Directory processing
            hq_dir = Path(args.hq_dir).expanduser() if args.hq_dir else None
            process_directory(
                model, Path(args.input_dir).expanduser(), Path(args.output_dir).expanduser(), 
                args.crf, args.preset, args.auto, args.test, 
                args.dry_run, args.overwrite, args.crop, hq_dir
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

4. Dry Run (Safe Test)
Use --dry_run with any command to see what the script would do without actually processing any images or writing any files. This is perfect for verifying your paths and filters are correct.

python scripts/restore_av1.py \
    --checkpoint checkpoints/conditional_unet_tiny_M4/best.pth \
    --input_dir /Users/soham/Documents/aura/av1_data/test/lq \
    --output_dir /Users/soham/Documents/aura/results/tiny_test_full_metrics \
    --hq_dir /Users/soham/Documents/aura/av1_data/test/hq \
    --test \
    --auto \
    --device mps \
    --dry_run
"""