# create_split.py
import argparse
import random
import shutil
from pathlib import Path
from tqdm import tqdm
import logging

# --- Setup basic logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

def create_deterministic_split(
    hq_source_dir: Path,
    lq_source_dir: Path,
    output_root_dir: Path,
    val_ratio: float,
    seed: int
):
    """
    Scans a source HQ directory, creates a deterministic train/val split,
    and moves both HQ and all corresponding LQ files to the new structure.
    """
    # --- 1. Identify all unique HQ images and create a reproducible shuffle ---
    logging.info(f"Scanning for HQ images in: {hq_source_dir}")
    
    # Find all common image types
    image_extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.tiff']
    all_hq_files = [p for p in hq_source_dir.glob('**/*') if p.suffix.lower() in image_extensions]

    if not all_hq_files:
        logging.error(f"No HQ images found in {hq_source_dir}. Aborting.")
        return

    # Sort for initial consistency, then shuffle for randomness
    all_hq_files.sort()
    random.seed(seed)
    random.shuffle(all_hq_files)
    
    # --- 2. Determine the split based on the validation ratio ---
    split_index = int(len(all_hq_files) * val_ratio)
    val_hq_paths = all_hq_files[:split_index]
    train_hq_paths = all_hq_files[split_index:]

    val_basenames = {p.stem for p in val_hq_paths}
    train_basenames = {p.stem for p in train_hq_paths}

    logging.info(f"Split created: {len(train_basenames)} train files, {len(val_basenames)} validation files.")

    # --- 3. Create destination directories ---
    train_hq_dest = output_root_dir / "train" / "hq"
    val_hq_dest = output_root_dir / "val" / "hq"
    train_lq_dest = output_root_dir / "train" / "lq"
    val_lq_dest = output_root_dir / "val" / "lq"

    train_hq_dest.mkdir(parents=True, exist_ok=True)
    val_hq_dest.mkdir(parents=True, exist_ok=True)
    # LQ destination subdirectories will be created on-the-fly

    # --- 4. Move HQ files ---
    for hq_path in tqdm(train_hq_paths, desc="Moving Train HQ files"):
        shutil.move(str(hq_path), train_hq_dest / hq_path.name)
        
    for hq_path in tqdm(val_hq_paths, desc="Moving Val HQ files"):
        shutil.move(str(hq_path), val_hq_dest / hq_path.name)

    # --- 5. Find and move all corresponding LQ files ---
    logging.info(f"Scanning for all LQ files in {lq_source_dir}. This may take a moment...")
    all_lq_files = list(lq_source_dir.rglob("*.avif"))
    
    logging.info(f"Found {len(all_lq_files)} total LQ files to process.")

    for lq_path in tqdm(all_lq_files, desc="Moving LQ files"):
        # Extract the base name (e.g., '0001' from '0001_crf23_p4.avif')
        base_name = lq_path.name.split('_crf')[0]
        
        # Determine if it belongs to train or val set
        if base_name in train_basenames:
            dest_root = train_lq_dest
        elif base_name in val_basenames:
            dest_root = val_lq_dest
        else:
            continue # Skip if it doesn't match any HQ file (should not happen)
            
        # Recreate the subfolder structure (e.g., crf_XX/preset_Y/)
        relative_path = lq_path.relative_to(lq_source_dir)
        final_dest = dest_root / relative_path
        
        # Ensure the final destination directory exists
        final_dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(lq_path), final_dest)
        
    logging.info("Splitting process complete!")
    logging.info(f"Final data is located in: {output_root_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Deterministically split combined HQ and LQ datasets into train and validation sets."
    )
    parser.add_argument(
        "--hq_source", 
        type=str, 
        required=True,
        help="Path to the combined folder of high-quality images for training and validation."
    )
    parser.add_argument(
        "--lq_source", 
        type=str, 
        required=True,
        help="Path to the root of the low-quality AV1 dataset (containing crf_XX folders)."
    )
    parser.add_argument(
        "--output_root", 
        type=str, 
        required=True,
        help="Path to the root directory where the final 'train' and 'val' folders will be created."
    )
    parser.add_argument(
        "--val_ratio", 
        type=float, 
        default=0.1,
        help="The proportion of the dataset to use for validation (e.g., 0.1 for 10%)."
    )
    parser.add_argument(
        "--seed", 
        type=int, 
        default=42,
        help="A random seed to ensure the split is reproducible."
    )
    args = parser.parse_args()

    create_deterministic_split(
        hq_source_dir=Path(args.hq_source).expanduser(),
        lq_source_dir=Path(args.lq_source).expanduser(),
        output_root_dir=Path(args.output_root).expanduser(),
        val_ratio=args.val_ratio,
        seed=args.seed
    )

"""
python create_split.py \
    --hq_source ~/Documents/aura/data/train_hq_combined \
    --lq_source dummy_lq \
    --output_root ~/Documents/aura/data_final \
    --val_ratio 0.1 \
    --seed 8


--hq_source: Your folder containing the combined DIV2K_train_HR and Flickr_HR_train images.
--lq_source: The root folder containing all the crf_XX/preset_Y subdirectories.
--output_root: A new folder where the final split will be created.
--val_ratio: The train, valid, 
--test_ratio: default None, (and test split? optional)


data_final/
├── train/
│   ├── hq/
│   │   ├── 0001.png
│   │   └── 0003.png
│   │   └── ... (90% of the images)
│   └── lq/
│       ├── crf_23/
│       │   └── preset_4/
│       │       ├── 0001_crf23_p4.avif
│       │       └── 0003_crf23_p4.avif
│       │       └── ...
│       └── ... (all other crf/preset folders)
└── val/
    ├── hq/
    │   ├── 0002.png
    │   └── 0004.png
    │   └── ... (10% of the images)
    └── lq/
        ├── crf_23/
        │   └── preset_4/
        │       ├── 0002_crf23_p4.avif
        │       └── 0004_crf23_p4.avif
        │       └── ...
        └── ... (all other crf/preset folders)
"""


