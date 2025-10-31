#!/usr/bin/env python3
"""
Preprocesses an image dataset (AVIF, PNG, etc.) into NumPy (.npy) files.

This significantly speeds up training and inference by moving the slow decoding
step to a one-time offline process.

- Saves as float32 [0, 1] for direct use by the model.
- Preserves the exact directory structure.
- Parallelized with multiprocessing to be extremely fast.
- Skips files that are already processed.
- Logs all errors to a file.

Usage:
    # Preprocess the entire LQ test set
    python scripts/preprocess_dataset.py \
        --input_dir ./av1_data/test/lq \
        --output_dir ./av1_data_npy/test/lq

    # Preprocess the HQ test set
    python scripts/preprocess_dataset.py \
        --input_dir ./av1_data/test/hq \
        --output_dir ./av1_data_npy/test/hq \
        --extensions .png .jpg

    # Preprocess the entire training set (run for both lq and hq)
    python scripts/preprocess_dataset.py \
        --input_dir ./av1_data/train/lq \
        --output_dir ./av1_data_npy/train/lq

    python scripts/preprocess_dataset.py \
        --input_dir ./av1_data/train/hq \
        --output_dir ./av1_data_npy/train/hq \
        --extensions .png .jpg
"""

import argparse
import logging
from pathlib import Path
import numpy as np
from PIL import Image
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from typing import List, Tuple, Optional

# --- Configuration ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger("NPYPreprocessor")

# Default extensions to look for
IMAGE_EXTENSIONS = ('.avif', '.png', '.jpg', '.jpeg', '.webp', '.bmp', '.tiff')

def find_images_and_tasks(
    in_dir: Path, 
    out_dir: Path, 
    extensions: Tuple[str, ...]
) -> List[Tuple[Path, Path]]:
    """Finds all images and creates a list of tasks, skipping existing."""
    logger.info(f"Scanning for {extensions} files in {in_dir}...")
    image_files = sorted([
        p for p in in_dir.rglob('*') 
        if p.is_file() and p.suffix.lower() in extensions
    ])
    
    logger.info(f"Found {len(image_files):,} images. Checking for existing .npy files...")
    
    tasks = []
    for img_path in tqdm(image_files, desc="Generating tasks"):
        rel_path = img_path.relative_to(in_dir)
        output_path = (out_dir / rel_path).with_suffix('.npy')
        
        if not output_path.exists():
            tasks.append((img_path, output_path))
            
    return tasks

def process_image_worker(args_tuple: Tuple[Path, Path]) -> Optional[str]:
    """
    Worker function: Loads one image, converts to float32 [0, 1], and saves as .npy.
    
    Args:
        args_tuple (Tuple[Path, Path]): (img_path, output_path)
    
    Returns:
        None on success, or an error string on failure.
    """
    img_path, output_path = args_tuple
    try:
        # Create parent directory
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Load image
        with Image.open(img_path) as img:
            img_rgb = img.convert('RGB')
        
        # Convert to np.float32 [0, 1]
        img_np = np.array(img_rgb).astype(np.float32) / 255.0
        
        # Save as .npy
        np.save(output_path, img_np)
        
        return None # Success
    except Exception as e:
        error_msg = f"Failed to process {img_path.name}: {e}"
        logger.debug(error_msg) # Log debug for immediate feedback
        return error_msg # Return error for final summary

def main():
    parser = argparse.ArgumentParser(
        description="Preprocess image dataset to .npy files.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '--input_dir', type=str, required=True,
        help='Input directory containing image files.'
    )
    parser.add_argument(
        '--output_dir', type=str, required=True,
        help='Output directory to save .npy files.'
    )
    parser.add_argument(
        '--extensions', nargs='+', default=IMAGE_EXTENSIONS,
        help='Which image extensions to process.'
    )
    parser.add_argument(
        '--workers', type=int, default=max(1, cpu_count() - 2),
        help='Number of parallel workers. Default is (CPUs - 2).'
    )
    args = parser.parse_args()

    in_dir = Path(args.input_dir).expanduser().resolve()
    out_dir = Path(args.output_dir).expanduser().resolve()
    
    if not in_dir.exists():
        logger.error(f"Input directory not found: {in_dir}")
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Input:    {in_dir}")
    logger.info(f"Output:   {out_dir}")
    logger.info(f"Workers:  {args.workers}")

    tasks = find_images_and_tasks(in_dir, out_dir, tuple(args.extensions))
    
    if not tasks:
        logger.info("✅ All files already pre-processed. Nothing to do.")
        return

    logger.info(f"Found {len(tasks):,} new images to process.")
    
    errors = []
    with Pool(processes=args.workers) as pool:
        results = list(tqdm(
            pool.imap_unordered(process_image_worker, tasks), 
            total=len(tasks), 
            desc="Processing images"
        ))
        
        errors = [r for r in results if r is not None]

    logger.info(f"✓ Preprocessing complete.")
    logger.info(f"  Successfully processed: {len(tasks) - len(errors):,}/{len(tasks):,}")

    if errors:
        error_log = out_dir / f"preprocess_error_log_{in_dir.name}.txt"
        logger.error(f"Encountered {len(errors)} errors. See log: {error_log}")
        with open(error_log, "w") as f:
            for err in errors:
                f.write(err + "\n")

if __name__ == "__main__":
    main()