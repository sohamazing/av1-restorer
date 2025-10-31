# utils/av1_dataset.py
"""
AV1 Dataset Module - Production Ready

Loads paired LQ/HQ images for AV1 artifact removal training.
Supports flexible CRF and preset filtering with robust error handling.

Directory Structure Expected:
    av1_dataset/
    ├── train/
    │   ├── crf_23/
    │   │   └── preset_4/
    │   │       ├── 0001_crf23_p4.avif
    │   │       └── 0002_crf23_p4.avif
    │   ├── crf_24/
    │   ...
    └── val/ (same structure)

Author: Soham Mukherjee
Version: 1.0
"""

import re
import logging
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any
from collections import Counter

import torch
from torch.utils.data import Dataset
from torchvision.transforms import v2 as T
from PIL import Image, UnidentifiedImageError
from tqdm import tqdm

logger = logging.getLogger(__name__)


class AV1Dataset(Dataset):
    """
    Professional dataset for AV1 artifact removal.
    
    Features:
      - Regex-based metadata extraction from filenames
      - Dynamic CRF and preset range filtering
      - Identical random cropping for LQ/HQ pairs
      - Graceful handling of corrupted files
      - Configurable normalization ([0,1] or [-1,1])
      - Data distribution analysis utilities
    
    Args:
        lq_root_dir (str): Root directory with compressed AVIF files
        hq_root_dir (str): Directory with original HQ images
        hq_ext (str): HQ file extension (e.g., '.png', '.jpg')
        patch_size (int): Square patch size for training
        crf_range (tuple): (min_crf, max_crf) to include, None for all
        preset_range (tuple): (min_preset, max_preset), None for all
        augment (bool): Enable random flips
        norm_range (str): '[0,1]' or '[-1,1]'
        crop (str): 'random' or 'center' or 'none'
        return_metadata (bool): Include file paths in output
    
    Example:
        >>> dataset = AV1Dataset(
        ...     lq_root_dir='./av1_dataset/train',
        ...     hq_root_dir='~/Desktop/Photos/Div2K/DIV2K_train_HR',
        ...     hq_ext='.png',
        ...     patch_size=256,
        ...     crf_range=(23, 63),
        ...     preset_range=(4, 4),
        ...     norm_range='[0,1]'
        ...     augment=True,
        ...     crop='random' 
        ... )
    """
    
    # Filename pattern: {base}_crf{CRF}_p{PRESET}.avif
    FILENAME_PATTERN = re.compile(r"^(.+?)_crf(\d+)_p(\d+)\.avif$")
    
    # Valid AV1 parameter ranges
    VALID_CRF_RANGE = (0, 63)
    VALID_PRESET_RANGE = (0, 8)  # libaom-av1 cpu_used: [0, 8], svt-av1 preset: [0, 13]

    def __init__(
        self,
        lq_root_dir: str,
        hq_root_dir: str,
        hq_ext: str,
        patch_size: int,
        crf_range: Optional[Tuple[int, int]] = None,
        preset_range: Optional[Tuple[int, int]] = None,
        norm_range: Tuple[int, int] = (0, 1),
        augment: bool = True,
        crop_mode: str = 'random',
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
        
        self.hq_ext = f'.{hq_ext.lstrip(".")}'
        self.patch_size = patch_size
        self.return_metadata = return_metadata
        
        # Validate and store filtering ranges
        self.crf_range = self._validate_range(
            crf_range, self.VALID_CRF_RANGE, "CRF"
        )
        self.preset_range = self._validate_range(
            preset_range, self.VALID_PRESET_RANGE, "Preset"
        )

        valid_crop_modes = ['random', 'center', 'none']
        if crop_mode not in valid_crop_modes:
            raise ValueError(f"Invalid crop_mode: '{crop_mode}'. Choose from {valid_crop_modes}")
        self.crop_mode = crop_mode
        
        # Setup transforms
        self.augment_tf = self._get_augment_transforms() if augment else None
        self.final_tf = self._get_final_transform(norm_range)

        if cached_image_pairs is not None:
            # If a file list is provided, use it and skip scanning
            logger.debug(f"Using cached file list with {len(cached_image_pairs)} pairs.")
            self.image_pairs = cached_image_pairs
        else:
            # Log configuration
            logger.info("="*60)
            logger.info("Initializing AV1Dataset")
            logger.info(f"  LQ Root:       {self.lq_root}")
            logger.info(f"  HQ Root:       {self.hq_root}")
            logger.info(f"  CRF Range:     {self.crf_range}")
            logger.info(f"  Preset Range:  {self.preset_range}")
            logger.info(f"  Patch Size:    {patch_size}×{patch_size}")
            logger.info(f"  Crop Mode:     {self.crop_mode}") # <-- Log crop mode
            logger.info(f"  Augmentation:  {'Enabled' if augment else 'Disabled'}")
            logger.info(f"  Normalization: {norm_range}")
            logger.info("="*60)
            
            # Build dataset index
            self.image_pairs = self._build_index()
            
            logger.info(f"✓ Dataset ready with {len(self.image_pairs):,} pairs")
            logger.info("="*60)

        # Add a check here in case the cache was empty or scanning failed
        if not self.image_pairs:
            raise FileNotFoundError(
                "No valid image pairs found. Check paths, ranges, naming, or cache."
            )
        
    @staticmethod
    def _validate_range(
        rng: Optional[Tuple[int, int]], 
        valid_rng: Tuple[int, int],
        name: str
    ) -> Tuple[int, int]:
        """Validate parameter range is within valid bounds."""
        if rng is None:
            return valid_rng
        
        min_val, max_val = rng
        valid_min, valid_max = valid_rng
        
        if not (valid_min <= min_val <= max_val <= valid_max):
            raise ValueError(
                f"{name} range {rng} invalid. "
                f"Must be within {valid_rng} and min ≤ max."
            )
        
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
                T.ToDtype(torch.float32, scale=True),  # [0, 255] -> [0, 1]
            ])
        elif norm_range == (-1, 1):
            return T.Compose([
                T.ToImage(),
                T.ToDtype(torch.float32, scale=True),
                T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]), # [0, 1] -> [-1, 1]
            ])
        else:
            raise ValueError(
                f"Invalid norm_range: {norm_range}. "
                f"Must be (0, 1) or (-1, 1)"
            )
    
    def _build_index(self) -> List[Dict[str, Any]]:
        """
        Scan LQ directory, parse filenames, filter by CRF/preset,
        and match with HQ images.
        """
        pairs = []
        lq_files = list(self.lq_root.rglob("*.avif"))
        
        if not lq_files:
            raise FileNotFoundError(f"No AVIF files found in {self.lq_root}")
        
        logger.info(f"Found {len(lq_files):,} AVIF files, filtering...")
        
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
            
            # Filter by CRF range
            if not (self.crf_range[0] <= crf <= self.crf_range[1]):
                skipped_crf += 1
                continue
            
            # Filter by preset range
            if not (self.preset_range[0] <= preset <= self.preset_range[1]):
                skipped_preset += 1
                continue
            
            # Find matching HQ image
            hq_path = self.hq_root / f"{base}{self.hq_ext}"
            if not hq_path.exists():
                missing_hq += 1
                logger.debug(f"Missing HQ for {lq_path.name}: {hq_path}")
                continue
            
            pairs.append({
                "lq": lq_path,
                "hq": hq_path,
                "crf": crf,
                "preset": preset,
                "base": base
            })
        
        # Log filtering statistics
        if skipped_parse > 0:
            logger.warning(f"Skipped {skipped_parse:,} files (invalid naming)")
        if skipped_crf > 0:
            logger.info(f"Filtered {skipped_crf:,} files (CRF out of range)")
        if skipped_preset > 0:
            logger.info(f"Filtered {skipped_preset:,} files (preset out of range)")
        if missing_hq > 0:
            logger.warning(f"Skipped {missing_hq:,} files (missing HQ pair)")
        
        if not pairs:
            raise FileNotFoundError(
                "No valid image pairs found after filtering. "
                "Check paths, ranges, and naming convention."
            )
        
        return pairs
    
    def __len__(self) -> int:
        return len(self.image_pairs)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Load and process a single training sample.
        
        Returns:
            dict with keys: 'lq', 'hq', 'crf', 'preset'
            Optional: 'lq_path', 'base_name' if return_metadata=True
        """
        meta = self.image_pairs[idx]

        try:
            lq_img = Image.open(meta["lq"]).convert("RGB")
            hq_img = Image.open(meta["hq"]).convert("RGB")

            lq_patch, hq_patch = lq_img, hq_img # Start with full images

            # --- APPLY CROPPING BASED ON self.crop_mode ---
            if self.patch_size > 0: # Only crop if patch_size is valid
                img_w, img_h = lq_img.size
                # Ensure images are large enough for the crop
                if img_w >= self.patch_size and img_h >= self.patch_size:
                    if self.crop_mode == 'random':
                        i, j, h, w = T.RandomCrop.get_params(
                            lq_img, (self.patch_size, self.patch_size)
                        )
                        lq_patch = T.functional.crop(lq_img, i, j, h, w)
                        hq_patch = T.functional.crop(hq_img, i, j, h, w)
                    elif self.crop_mode == 'center':
                        i = (img_h - self.patch_size) // 2
                        j = (img_w - self.patch_size) // 2
                        h = w = self.patch_size
                        lq_patch = T.functional.crop(lq_img, i, j, h, w)
                        hq_patch = T.functional.crop(hq_img, i, j, h, w)
                    # elif self.crop_mode == 'none':
                    #     pass # Keep full image if mode is 'none'
                else:
                    logger.warning(
                        f"Image {meta['base']} ({img_w}x{img_h}) is smaller than "
                        f"patch size {self.patch_size}. Skipping crop for this item."
                    )
            # --- END CROPPING LOGIC ---


            # --- APPLY AUGMENTATION (FLIPS) INDEPENDENTLY ---
            if self.augment_tf: # Check self.augment here
                seed = torch.randint(0, 2**32, (1,)).item()
                torch.manual_seed(seed)
                lq_patch = self.augment_tf(lq_patch)
                torch.manual_seed(seed)
                hq_patch = self.augment_tf(hq_patch)
            # --- END AUGMENTATION ---

            # Convert to tensors
            lq_tensor = self.final_tf(lq_patch)
            hq_tensor = self.final_tf(hq_patch)
            
            # Build output dictionary
            result = {
                "lq": lq_tensor,
                "hq": hq_tensor,
                "crf": torch.tensor([meta["crf"]], dtype=torch.float32),
                "preset": torch.tensor([meta["preset"]], dtype=torch.float32),
            }
            
            # Add metadata if requested
            if self.return_metadata:
                result["lq_path"] = str(meta["lq"])
                result["base_name"] = meta["base"]
            
            return result
            
        except (IOError, UnidentifiedImageError, OSError) as e:
            logger.warning(
                f"Corrupted file at index {idx}: {meta['lq']}. "
                f"Randomly sampling another. Error: {e}"
            )
            # Load random sample to prevent training crash
            random_idx = torch.randint(0, len(self), (1,)).item()
            return self.__getitem__(random_idx)
    
    # ========== Utility Methods ==========
    
    def get_crf_distribution(self) -> Dict[int, int]:
        """Return distribution of CRF values in dataset."""
        return dict(Counter(pair["crf"] for pair in self.image_pairs))
    
    def get_preset_distribution(self) -> Dict[int, int]:
        """Return distribution of preset values in dataset."""
        return dict(Counter(pair["preset"] for pair in self.image_pairs))
    
    def get_statistics(self) -> Dict[str, Any]:
        """Return comprehensive dataset statistics."""
        crf_dist = self.get_crf_distribution()
        preset_dist = self.get_preset_distribution()
        
        return {
            "total_pairs": len(self.image_pairs),
            "crf_range": self.crf_range,
            "preset_range": self.preset_range,
            "crf_distribution": crf_dist,
            "preset_distribution": preset_dist,
            "crf_unique_values": len(crf_dist),
            "preset_unique_values": len(preset_dist),
            "patch_size": self.patch_size,
        }
    
    def print_statistics(self):
        """Print formatted dataset statistics."""
        stats = self.get_statistics()
        
        print("\n" + "="*60)
        print("DATASET STATISTICS")
        print("="*60)
        print(f"Total Pairs:       {stats['total_pairs']:,}")
        print(f"CRF Range:         {stats['crf_range']}")
        print(f"Preset Range:      {stats['preset_range']}")
        print(f"Patch Size:        {stats['patch_size']}×{stats['patch_size']}")
        print(f"\nCRF Distribution ({stats['crf_unique_values']} unique values):")
        for crf, count in sorted(stats['crf_distribution'].items()):
            print(f"  CRF {crf:2d}: {count:5,} images")
        print(f"\nPreset Distribution ({stats['preset_unique_values']} unique values):")
        for preset, count in sorted(stats['preset_distribution'].items()):
            print(f"  Preset {preset}: {count:5,} images")
        print("="*60 + "\n")


# ============================================================================
# Unit Tests / Example Usage
# ============================================================================

if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)
    
    # Example configuration
    dataset = AV1Dataset(
        lq_root_dir="/Users/soham/Documents/aura/av1_dataset/train",
        hq_root_dir="/Users/soham/Desktop/Photos/Div2K/DIV2K_train_HR",
        hq_ext=".png",
        patch_size=256,
        crf_range=(23, 63),
        preset_range=(4, 4),
        norm_range=(-1, 1),
        crop_mode='center',
        augment=True,
        return_metadata=False
    )
    
    # Print statistics
    dataset.print_statistics()
    
    # Test loading a sample
    sample = dataset[0]
    print(f"Sample keys: {sample.keys()}")
    print(f"LQ shape: {sample['lq'].shape}, range: [{sample['lq'].min():.3f}, {sample['lq'].max():.3f}]")
    print(f"HQ shape: {sample['hq'].shape}, range: [{sample['hq'].min():.3f}, {sample['hq'].max():.3f}]")
    print(f"CRF: {sample['crf'].item()}, Preset: {sample['preset'].item()}")
