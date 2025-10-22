# utils/av1_dataset_fast.py

"""
Fast AV1 Dataset - Loads pre-processed .npy files for maximum throughput.
Drop-in replacement for AV1Dataset that loads from .npy files

Eliminates AVIF decoding overhead by loading pre-decoded NumPy arrays.
Maintains identical API to AV1Dataset for seamless integration.

Usage:
    1. Pre-process dataset: python scripts/preprocess_av1_dataset.py ...
    2. Update config: lq_ext: ".npy"
    3. Training automatically uses FastAV1Dataset
"""

import re
import logging
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any
from collections import Counter

import torch
from torch.utils.data import Dataset
from torchvision.transforms import v2 as T
from PIL import Image
import numpy as np
from tqdm import tqdm

logger = logging.getLogger(__name__)


class AV1DatasetFast(Dataset):
    """
    High-performance dataset for pre-processed .npy files. (NumPy)
    
    ~10x faster than decoding AVIF files on-the-fly.
    Identical API to AV1Dataset for drop-in replacement. 
    """
    
    FILENAME_PATTERN = re.compile(r"^(.+?)_crf(\d+)_p(\d+)\.npy$")
    VALID_CRF_RANGE = (0, 63)
    VALID_PRESET_RANGE = (0, 13)

    def __init__(
        self,
        lq_root_dir: str,
        hq_root_dir: str,
        hq_ext: str, # Keep hq_ext even if forced later, for signature consistency
        patch_size: int,
        crf_range: Optional[Tuple[int, int]] = None,
        preset_range: Optional[Tuple[int, int]] = None,
        augment: bool = True,
        norm_range: Tuple[int, int] = (0, 1),
        return_metadata: bool = False,
        cached_image_pairs: Optional[list] = None
    ):
        # Validate and resolve paths
        self.lq_root = Path(lq_root_dir).expanduser().resolve()
        self.hq_root = Path(hq_root_dir).expanduser().resolve()

        if not self.lq_root.exists():
            raise FileNotFoundError(f"LQ directory not found: {self.lq_root}")
        if not self.hq_root.exists():
            raise FileNotFoundError(f"HQ directory not found: {self.hq_root}")

        # Force .npy extension for HQ as FastDataset expects preprocessed HQ too
        self.hq_ext = '.npy'
        self.patch_size = patch_size
        self.return_metadata = return_metadata

        # Validate filtering ranges
        self.crf_range = self._validate_range(crf_range, self.VALID_CRF_RANGE, "CRF")
        self.preset_range = self._validate_range(preset_range, self.VALID_PRESET_RANGE, "Preset")

        # Setup transforms
        self.augment_tf = self._get_augment_transforms() if augment else None
        self.final_tf = self._get_final_transform(norm_range)

        # --- Caching Logic ---
        if cached_image_pairs is not None:
            logger.debug(f"Using cached file list with {len(cached_image_pairs)} pairs for FastDataset.")
            self.image_pairs = cached_image_pairs
        else:
            # --- Original Scanning/Filtering Logic ---
            # Log configuration
            logger.info("="*60)
            logger.info("Initializing FastAV1Dataset (NumPy format)") # Keep specific log
            logger.info(f"  LQ Root:       {self.lq_root}")
            logger.info(f"  HQ Root:       {self.hq_root}")
            logger.info(f"  CRF Range:     {self.crf_range}")
            logger.info(f"  Preset Range:  {self.preset_range}")
            logger.info(f"  Patch Size:    {patch_size}×{patch_size}")
            logger.info(f"  Augmentation:  {'Enabled' if augment else 'Disabled'}")
            logger.info(f"  Normalization: {norm_range}")
            logger.info("="*60)

            # Build dataset index (scans for .npy files now)
            self.image_pairs = self._build_index()

            logger.info(f"✓ FastDataset ready with {len(self.image_pairs):,} pairs")
            logger.info("="*60)
            # --- End of Original Logic ---

        # Add a check here in case the cache was empty or scanning failed
        if not self.image_pairs:
            raise FileNotFoundError(
                "No valid image pairs found for FastDataset. Check paths, ranges, naming, or cache."
            )
    
    @staticmethod
    def _validate_range(rng: Optional[Tuple[int, int]], valid_rng: Tuple[int, int], name: str) -> Tuple[int, int]:
        """Validate parameter range."""
        if rng is None:
            return valid_rng
        
        min_val, max_val = rng
        valid_min, valid_max = valid_rng
        
        if not (valid_min <= min_val <= max_val <= valid_max):
            raise ValueError(f"{name} range {rng} invalid. Must be within {valid_rng}.")
        
        return (min_val, max_val)
    
    def _get_augment_transforms(self):
        """Random horizontal and vertical flips."""
        return T.Compose([
            T.RandomHorizontalFlip(p=0.5),
            T.RandomVerticalFlip(p=0.5),
        ])
    
    def _get_final_transform(self, norm_range: Tuple[int, int]):
        """Convert PIL to tensor with normalization."""
        if norm_range == (0, 1):
            return T.Compose([
                T.ToImage(),
                T.ToDtype(torch.float32, scale=True),
            ])
        elif norm_range == (-1, 1):
            return T.Compose([
                T.ToImage(),
                T.ToDtype(torch.float32, scale=True),
                T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ])
        else:
            raise ValueError(f"Invalid norm_range: {norm_range}")
    
    def _build_index(self) -> List[Dict[str, Any]]:
        """Scan for .npy files and build index."""
        pairs = []
        lq_files = list(self.lq_root.rglob("*.npy"))
        
        if not lq_files:
            raise FileNotFoundError(
                f"No .npy files found in {self.lq_root}. "
                f"Did you run scripts/preprocess_av1_dataset.py?"
            )
        
        logger.info(f"Found {len(lq_files):,} NPY files, filtering...")
        
        skipped_parse = 0
        skipped_crf = 0
        skipped_preset = 0
        missing_hq = 0
        
        for lq_path in tqdm(lq_files, desc="Indexing", unit="file"):
            # Parse filename
            match = self.FILENAME_PATTERN.match(lq_path.name)
            if not match:
                skipped_parse += 1
                continue
            
            base, crf_str, preset_str = match.groups()
            crf = int(crf_str)
            preset = int(preset_str)
            
            # Filter by ranges
            if not (self.crf_range[0] <= crf <= self.crf_range[1]):
                skipped_crf += 1
                continue
            
            if not (self.preset_range[0] <= preset <= self.preset_range[1]):
                skipped_preset += 1
                continue
            
            # Find matching HQ file
            hq_path = self.hq_root / f"{base}.npy"
            if not hq_path.exists():
                missing_hq += 1
                continue
            
            pairs.append({
                "lq": lq_path,
                "hq": hq_path,
                "crf": crf,
                "preset": preset,
                "base": base
            })
        
        # Log filtering stats
        if skipped_parse > 0:
            logger.warning(f"Skipped {skipped_parse:,} files (invalid naming)")
        if skipped_crf > 0:
            logger.info(f"Filtered {skipped_crf:,} files (CRF out of range)")
        if skipped_preset > 0:
            logger.info(f"Filtered {skipped_preset:,} files (preset out of range)")
        if missing_hq > 0:
            logger.warning(f"Skipped {missing_hq:,} files (missing HQ pair)")
        
        if not pairs:
            raise FileNotFoundError("No valid pairs found after filtering.")
        
        return pairs
    
    def __len__(self) -> int:
        return len(self.image_pairs)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Load sample from .npy files (fast!)."""
        meta = self.image_pairs[idx]
        
        try:
            # Load NumPy arrays directly (10x faster than AVIF decode)
            lq_array = np.load(meta["lq"])  # [H, W, 3] uint8
            hq_array = np.load(meta["hq"])  # [H, W, 3] uint8
            
            # Convert to PIL for transforms
            lq_img = Image.fromarray(lq_array)
            hq_img = Image.fromarray(hq_array)
            
            # Apply identical random crop
            i, j, h, w = T.RandomCrop.get_params(lq_img, (self.patch_size, self.patch_size))
            lq_patch = T.functional.crop(lq_img, i, j, h, w)
            hq_patch = T.functional.crop(hq_img, i, j, h, w)
            
            # Apply identical augmentations
            if self.augment_tf:
                seed = torch.randint(0, 2**32, (1,)).item()
                torch.manual_seed(seed)
                lq_patch = self.augment_tf(lq_patch)
                torch.manual_seed(seed)
                hq_patch = self.augment_tf(hq_patch)
            
            # Convert to tensors
            lq_tensor = self.final_tf(lq_patch)
            hq_tensor = self.final_tf(hq_patch)
            
            result = {
                "lq": lq_tensor,
                "hq": hq_tensor,
                "crf": torch.tensor([meta["crf"]], dtype=torch.float32),
                "preset": torch.tensor([meta["preset"]], dtype=torch.float32),
            }
            
            if self.return_metadata:
                result["lq_path"] = str(meta["lq"])
                result["base_name"] = meta["base"]
            
            return result
            
        except Exception as e:
            logger.warning(f"Error loading at index {idx}: {e}. Resampling.")
            return self.__getitem__(torch.randint(0, len(self), (1,)).item())
    
    # Utility methods (same as AV1Dataset)
    def get_crf_distribution(self) -> Dict[int, int]:
        return dict(Counter(pair["crf"] for pair in self.image_pairs))
    
    def get_preset_distribution(self) -> Dict[int, int]:
        return dict(Counter(pair["preset"] for pair in self.image_pairs))


"""
STEP 1: Pre-process your dataset (ONE TIME, run overnight)
============================================================

# Training data
python scripts/preprocess_av1_dataset.py \
    --lq_dir ~/av1_data/train/lq \
    --hq_dir ~/av1_data/train/hq \
    --out_dir ~/av1_data_npy/train \
    --workers 16

# Validation data
python scripts/preprocess_av1_dataset.py \
    --lq_dir ~/av1_data/val/lq \
    --hq_dir ~/av1_data/val/hq \
    --out_dir ~/av1_data_npy/val \
    --workers 16

# Test data
python scripts/preprocess_av1_dataset.py \
    --lq_dir ~/av1_data/test/lq \
    --hq_dir ~/av1_data/test/hq \
    --out_dir ~/av1_data_npy/test \
    --workers 16


STEP 2: Update your config YAML
================================

data:
  train_lq_root: "~/gcs_data/av1_data_npy/train/lq"  # Point to .npy files
  train_hq_root: "~/gcs_data/av1_data_npy/train/hq"
  val_lq_root: "~/gcs_data/av1_data_npy/val/lq"
  val_hq_root: "~/gcs_data/av1_data_npy/val/hq"
  hq_ext: ".npy"  # Changed from .png
  lq_ext: ".npy"  # Changed from .avif - THIS TRIGGERS FAST MODE

system:
  num_workers: 12  # Can use more workers now (less CPU overhead)


STEP 3: Train normally
=======================

python -m av1_restorer.train_av1_restorer --config configs/train_tiny_restorer.yaml

# Training will automatically detect .npy format and use FastAV1Dataset!

"""
