#!/usr/bin/env python3
"""
restore_av1.py — Unified Inference Script for AV1 U-Net Restorer

Features:
- Single-image and directory batch modes
- Auto-detect CRF/preset from filenames (supports --auto)
- Test-mode for structured test directories: lq/crf_{N}/preset_{M}/...
- Manual overrides: apply a single CRF/preset to all images in a directory
- Preserves relative directory structure when saving
- Safe device selection (auto -> cuda, mps, cpu)
- Tiling support for large images
- Handles common checkpoint name shortcuts and EMA/state dict variants
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

# --- Model Imports ---
try:
    from av1_restorer.models.av1_unet_restorer import AV1UNetRestorer
except ImportError:
    logging.error("Failed to import model from 'av1_restorer.models.av1_unet_restorer'. Ensure the path is correct.")
    sys.exit(1)

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
            return torch.device('cuda')
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return torch.device('mps')
        return torch.device('cpu')
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
        self.model, self.config = self._load_model(checkpoint)
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

    def _load_model(self, ckpt_path: str) -> Tuple[torch.nn.Module, dict]:
        """Loads a checkpoint, reconstructs the model, and loads the weights."""
        path = Path(ckpt_path).expanduser()
        
        # Handle shortcuts like 'best' by looking in the default checkpoint directory.
        if not path.is_file():
            path = Path('checkpoints/av1-restorer-v1') / ckpt_path
            if not path.exists():
                path = Path('checkpoints/av1-restorer-v1') / f"{ckpt_path}.pth"
        
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
            
        logger.info(f"Loading checkpoint: {path}")
        data = torch.load(path, map_location='cpu')
        config = data.get('config')
        if not config:
            raise ValueError("Checkpoint is missing the 'config' dictionary, cannot reconstruct model.")

        # Re-create the model architecture using the config stored in the checkpoint.
        model_cfg = config['model']
        dset_cfg = config['dataset']
        
        # Get model size configs
        size = model_cfg['size']
        size_configs = {
            'tiny': {'channels': [24, 48, 64, 128, 224], 'blocks': [2, 2, 2, 2, 4]},
            'small': {'channels': [20, 40, 80, 160, 320], 'blocks': [2, 2, 3, 3, 6]},
            'base': {'channels': [48, 64, 128, 224, 384], 'blocks': [2, 2, 3, 3, 6]},
            'large': {'channels': [64, 96, 128, 256, 512], 'blocks': [2, 3, 4, 4, 8]},
        }
        
        model_config = size_configs[size]
        model_config['crf_range'] = tuple(dset_cfg['crf_range'])
        model_config['preset_range'] = tuple(dset_cfg['preset_range'])
        model_config['norm_range'] = tuple(dset_cfg.get('norm_range', [-1, 1]))
        
        model = AV1UNetRestorer(model_config)
        
        # Prioritize loading EMA weights if available, as they often give better results.
        if 'ema_shadow' in data:
            logger.info("  Using EMA weights")
            # Reconstruct EMA state dict
            state_dict = {}
            for name, param in model.named_parameters():
                if name in data['ema_shadow']:
                    state_dict[name] = data['ema_shadow'][name]
            model.load_state_dict(state_dict)
        else:
            weights = data.get('model_state_dict') or data.get('model_state')
            if not weights:
                raise ValueError("Could not find model weights in checkpoint.")
            model.load_state_dict(weights)
        
        total_params = sum(p.numel() for p in model.parameters())
        logger.info(f"  Model: {size} ({total_params/1e6:.2f}M parameters)")
        
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
        
        # Run inference with tiling support
        out = self.model.inference(
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
    crop: bool = False
    hq_path: Optional[Path] = None
) -> Optional[Dict[str, float]]:
    """
    Orchestrates the restoration of a single image file.
    
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
    
    if crf is None or preset is None:
        logger.warning(f"Skipping (no params available): {img_path}")
        return None

    # In dry-run mode, just log what would happen.
    if dry_run:
        logger.info(f"[DRY RUN] {img_path.name} -> {out_path.name} (CRF={crf}, Preset={preset})")
        return None

    # Execute the restoration.
    try:
        lq_img = Image.open(img_path).convert('RGB')
        restored = model.restore(lq_img, crf, preset, crop=args.crop)
        
        out_path.parent.mkdir(parents=True, exist_ok=True)
        restored.save(out_path, quality=95)
        
        # Compute metrics if HQ image provided (test mode)
        if hq_path and hq_path.exists():
            hq_img = Image.open(hq_path).convert('RGB')
            
            # Ensure same size
            if hq_img.size != restored.size:
                hq_img = hq_img.resize(restored.size, Image.LANCZOS)
            
            # Convert to numpy arrays
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
        logger.error(f"Failed to process {img_path.name}: {e}")
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
        search_paths = []
        
        if crf is not None and preset is not None:
            # Specific CRF and preset
            search_paths.append(in_dir / f"crf_{crf}" / f"preset_{preset}")
        elif crf is not None:
            # All presets for specific CRF
            crf_dir = in_dir / f"crf_{crf}"
            if crf_dir.exists():
                search_paths.extend([p for p in crf_dir.iterdir() if p.is_dir() and p.name.startswith('preset_')])
        elif preset is not None:
            # All CRFs for specific preset
            for crf_dir in in_dir.glob("crf_*"):
                preset_dir = crf_dir / f"preset_{preset}"
                if preset_dir.exists():
                    search_paths.append(preset_dir)
        else:
            # All CRFs and presets (auto mode in test)
            search_paths.extend(in_dir.rglob("preset_*"))
        
        image_files = []
        for search_path in search_paths:
            image_files.extend(find_images(search_path))
    else:
        # In normal mode, search the entire input directory.
        image_files = find_images(in_dir)

    if not image_files:
        logger.warning(f"No images found for the specified criteria in {in_dir}")
        return

    logger.info(f"Found {len(image_files)} images to process.")
    
    # Metrics tracking for test mode
    metrics_list = []
    
    for img_path in tqdm(image_files, desc="Restoring images"):
        # Preserve the relative path from the input base to the output.
        rel_path = img_path.relative_to(in_dir)
        out_path = (out_dir / rel_path).with_suffix('.png')
        
        # Extract CRF/preset from path if in test mode and auto
        img_crf, img_preset = crf, preset
        if test_mode and auto:
            params = extract_params_from_filename(img_path.name)
            if params:
                img_crf, img_preset = params
        
        # Find corresponding HQ image if in test mode
        hq_path = None
        if test_mode and hq_dir:
            # Extract base name without CRF/preset suffix
            base_name = img_path.name.split('_crf')[0]
            # Look for HQ image with same base name
            potential_hq = list(hq_dir.glob(f"{base_name}.*"))
            if potential_hq:
                hq_path = potential_hq[0]
        
        metrics = process_single_image(
            model, img_path, out_path, img_crf, img_preset, 
            auto, dry_run, overwrite, hq_path, crop
        )
        
        if metrics:
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
        logger.info(f"Average Improvement:         {avg_improvement:.6f}")
        logger.info(f"Improvement %:               {(avg_improvement/avg_lq_loss)*100:.2f}%")
        logger.info("="*60 + "\n")


# --- Command-Line Interface ---
def main():
    """Parses arguments and orchestrates the inference process."""
    parser = argparse.ArgumentParser(
        description="Unified inference script for AV1 U-Net Restorer",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # --- Required Arguments ---
    parser.add_argument('--checkpoint', '-c', required=True, 
                       help='Path to model checkpoint (e.g., "best" or full path).')
    
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
                       help='Manual preset value. In --test mode, acts as a filter.')
    parser.add_argument('--auto', action='store_true', 
                       help='Auto-detect CRF/preset from filenames.')
    
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
    if not args.auto and not args.test and (args.crf is None or args.preset is None):
        parser.error("Must provide --crf and --preset for manual mode (when not using --auto or --test).")

    # --- Execution ---
    try:
        model = AV1Restorer(args.checkpoint, args.device, args.tile, args.overlap)

        if args.input:
            # Single image processing
            hq_path = Path(args.hq_dir) / Path(args.input).name if args.hq_dir else None
            metrics = process_single_image(
                model, Path(args.input), Path(args.output), 
                args.crf, args.preset, args.auto, args.dry_run, args.overwrite, hq_path
            )
            
            if metrics:
                logger.info(f"\nMetrics:")
                logger.info(f"  LQ Loss (MSE):       {metrics['lq_loss']:.6f}")
                logger.info(f"  Restored Loss (MSE): {metrics['restored_loss']:.6f}")
                logger.info(f"  Improvement:         {metrics['improvement']:.6f}")
        else:
            # Directory processing
            hq_dir = Path(args.hq_dir).expanduser() if args.hq_dir else None
            process_directory(
                model, Path(args.input_dir), Path(args.output_dir), 
                args.crf, args.preset, args.auto, args.test, 
                args.dry_run, args.overwrite, hq_dir
            )
        
        logger.info("✓ Inference complete!")

    except Exception as e:
        logger.error(f"❌ An unexpected error occurred: {e}", exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    main()