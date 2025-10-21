# scripts/preprocess_av1_dataset.py
"""
Pre-processes AVIF/PNG datasets to full-resolution NumPy (.npy) files for fast loading.

- Eliminates per-epoch AVIF decoding overhead.
- Preserves full resolution for flexible random cropping during training.
- Parallelized for efficiency.

Usage:
python scripts/preprocess_av1_dataset.py \
    --lq_dir ~/av1_data/train/lq \
    --hq_dir ~/av1_data/train/hq \
    --out_dir ~/av1_data_npy/train
"""
import argparse
import re
import logging
from pathlib import Path
from multiprocessing import Pool, cpu_count
from PIL import Image
import numpy as np
from tqdm import tqdm

# --- Configuration ---
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(message)s")
FILENAME_PATTERN = re.compile(r"^(.+?)_crf(\d+)_p(\d+)\.avif$")

def process_pair(args_tuple):
    """Worker function: converts one LQ/HQ pair to full-resolution .npy files."""
    lq_path, hq_root, hq_ext, lq_out_path, hq_out_path = args_tuple
    try:
        # Ensure output directories exist
        lq_out_path.parent.mkdir(parents=True, exist_ok=True)
        hq_out_path.parent.mkdir(parents=True, exist_ok=True)

        # --- LQ processing ---
        lq_img = Image.open(lq_path).convert("RGB")
        np.save(lq_out_path, np.array(lq_img, dtype=np.uint8))

        # --- HQ processing ---
        match = FILENAME_PATTERN.match(lq_path.name)
        if not match:
            return f"Invalid LQ filename format: {lq_path.name}"

        base_name = match.groups()[0]
        hq_path = hq_root / f"{base_name}{hq_ext}"
        if not hq_path.exists():
            return f"Missing HQ pair for {lq_path.name}"

        hq_img = Image.open(hq_path).convert("RGB")
        np.save(hq_out_path, np.array(hq_img, dtype=np.uint8))

        return None # Success
    except Exception as e:
        return f"Error processing {lq_path}: {e}"

def main():
    parser = argparse.ArgumentParser(
        description="Pre-decode AVIF/PNG dataset to full-resolution .npy files.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--lq_dir", type=str, required=True, help="Input LQ directory (AVIF).")
    parser.add_argument("--hq_dir", type=str, required=True, help="Input HQ directory (PNG).")
    parser.add_argument("--out_dir", type=str, required=True, help="Output directory for .npy files.")
    parser.add_argument("--hq_ext", type=str, default=".png", help="HQ file extension.")
    parser.add_argument("--workers", type=int, default=max(1, cpu_count() - 2), help="Number of parallel workers.")
    args = parser.parse_args()

    lq_root = Path(args.lq_dir).expanduser()
    hq_root = Path(args.hq_dir).expanduser()
    out_root = Path(args.out_dir).expanduser()

    logging.info(f"Scanning for AVIF files in {lq_root}...")
    lq_files = list(lq_root.rglob("*.avif"))
    if not lq_files:
        logging.error(f"No .avif files found in {lq_root}. Check your path.")
        return

    logging.info(f"Found {len(lq_files):,} files. Generating tasks to process...")
    tasks = []
    for lq_path in tqdm(lq_files, desc="Generating tasks"):
        relative_path = lq_path.relative_to(lq_root)
        match = FILENAME_PATTERN.match(lq_path.name)
        if not match: continue
        base_name = match.groups()[0]

        lq_out_path = (out_root / "lq" / relative_path).with_suffix(".npy")
        hq_out_path = (out_root / "hq" / f"{base_name}.npy")

        if not (lq_out_path.exists() and hq_out_path.exists()):
            tasks.append((lq_path, hq_root, args.hq_ext, lq_out_path, hq_out_path))

    if not tasks:
        logging.info("✅ All files already pre-processed. Nothing to do.")
        return

    logging.info(f"Processing {len(tasks):,} new image pairs with {args.workers} workers...")
    with Pool(processes=args.workers) as pool:
        results = list(tqdm(pool.imap_unordered(process_pair, tasks), total=len(tasks), desc="Processing images"))

    errors = [r for r in results if r is not None]
    if errors:
        error_log = out_root / "preprocess_error_log.txt"
        logging.error(f"Encountered {len(errors)} errors. See {error_log}")
        with open(error_log, "w") as f:
            for err in errors: f.write(err + "\n")

    logging.info(f"✅ Pre-processing complete! Full-resolution data ready at {out_root}")

if __name__ == "__main__":
    main()


