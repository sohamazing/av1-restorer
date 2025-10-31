"""
===============================================================================
AV1 DATASET MODULE (Unified SOTA Factory Edition)
===============================================================================

Author: Soham Mukherjee
Version: 5.4 (SOTA, Pickle-Safe, Fast Grid-Init, Optimal GetItem)

This is the definitive, SOTA dataset file. It provides all dataset
variants under a single `create_dataset` factory.

It is built on a "mixin" architecture to be DRY (Don't Repeat Yourself)
and pre-defines all class combinations at the top level. This ensures
compatibility with PyTorch's multiprocessing DataLoader.

AV1 Dataset Module - Production Ready

Loads paired LQ/HQ images for AV1 artifact removal training.
Supports flexible CRF and preset filtering with robust error handling.

Directory Structure Expected:
    av1_data/
    ├── train/
    │   ├── crf_23/
    │   │   └── preset_4/
    │   │       ├── 0001_crf23_p4.avif
    │   │       └── 0002_crf23_p4.avif
    │   ├── crf_24/
    │   ...
    └── val/ (same structure)

==============================================================================
CLASSES
==============================================================================
  - AV1Dataset:       The core class for file indexing and PIL-based loading.
  - AV1DatasetFast:   High-performance subclass. Loads .npy files.

  - _VirtualAugment: "Mixin" for virtual training augmentation (N-crops).
  - _DeterministicGrid: "Mixin" for deterministic grid validation.

  - Pre-defined combinations of the above (e.g., AV1FastTrainAugmented, AV1ValGrid) for the factory to use.

==============================================================================
FACTORY
==============================================================================
  - create_dataset(): The single public function that intelligently
    selects the correct, pre-defined, pickle-safe class.
"""

import re
import math
import logging
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any
from collections import Counter

import torch
from torch.utils.data import Dataset
from torchvision.transforms import v2 as T
from PIL import Image, UnidentifiedImageError
from tqdm import tqdm
import numpy as np

# --- Top-level logger ---
logger = logging.getLogger(__name__)

# ============================================================================
# SECTION 1: BASE DATASET (PIL-based)
# ============================================================================

class AV1Dataset(Dataset):
    """
    Base dataset for AV1 artifact removal (PIL-based).
    
    Handles core logic: file indexing, PIL loading, and a split transform
    pipeline for maximum performance.

    Features:
      - Regex-based metadata extraction from filenames
      - Dynamic CRF and preset range filtering
      - Identical random cropping for LQ/HQ pairs
      - Graceful handling of corrupted files
      - Configurable normalization ([0,1] or [-1,1])
    
    Args:
        lq_root_dir (str): Path to the Low-Quality (LQ) image directory.
        hq_root_dir (str): Path to the High-Quality (HQ) image directory.
        hq_ext (str): File extension for HQ images (e.g., '.png').
        patch_size (int): The square patch size for cropping.
        crf_range (Tuple[int, int], optional): (min, max) CRF values to include.
        preset_range (Tuple[int, int], optional): (min, max) preset values to include.
        norm_range (Tuple[int, int], optional): Normalization range, e.g., (0, 1) or (-1, 1).
        augment (bool, optional): Whether to apply flip augmentations.
        crop_mode (str, optional): 'random', 'center', or 'none'.
        return_metadata (bool, optional): If True, return file paths and base names.
        cached_image_pairs (list, optional): Pre-scanned file list to accelerate init.
    """
    
    # This pattern is for .avif files, subclasses can override it 
    FILENAME_PATTERN = re.compile(r"^(.+?)_crf(\d+)_p(\d+)\.avif$") # {base}_crf{XX}_p{Y}.avif
    VALID_CRF_RANGE = (0, 63)
    VALID_PRESET_RANGE = (0, 8) # libaom-av1 cpu_used: [0, 8], svt-av1 preset: [0, 13]

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
        self.lq_root = Path(lq_root_dir).expanduser().resolve()
        self.hq_root = Path(hq_root_dir).expanduser().resolve()
        
        if not self.lq_root.exists():
            raise FileNotFoundError(f"LQ directory not found: {self.lq_root}")
        if not self.hq_root.exists():
            raise FileNotFoundError(f"HQ directory not found: {self.hq_root}")
        
        self.hq_ext = f'.{hq_ext.lstrip(".")}'
        self.patch_size = patch_size
        self.return_metadata = return_metadata
        self.crop_mode = crop_mode
        self.augment = augment
        
        self.crf_range = self._validate_range(crf_range, self.VALID_CRF_RANGE, "CRF")
        self.preset_range = self._validate_range(preset_range, self.VALID_PRESET_RANGE, "Preset")

        # --- SOTA v5.3 Performance Fix: Split Transform Pipeline ---
        # 1. Geometric transforms (cheap, on PIL images)
        self.geometric_transform = self._create_geometric_pipeline(
            patch_size, crop_mode, augment
        )
        # 2. Tensor transforms (expensive, on small patches)
        self.tensor_transform = self._create_tensor_pipeline(norm_range)
        # ---
        
        if cached_image_pairs is not None:
            logger.debug(f"Using cached file list with {len(cached_image_pairs)} pairs.")
            self.image_pairs = cached_image_pairs
        else:
            self._log_init(norm_range)
            self.image_pairs = self._build_index()
            logger.info(f"✓ {self.__class__.__name__} ready with {len(self.image_pairs):,} pairs")
            logger.info("="*60)

        if not self.image_pairs:
            raise FileNotFoundError("No valid image pairs found.")

    def _log_init(self, norm_range):
        """Helper to log the initialization parameters."""
        logger.info("="*60)
        logger.info(f"Initializing {self.__class__.__name__} (PIL Loader)")
        logger.info(f"  LQ Root:       {self.lq_root}")
        logger.info(f"  HQ Root:       {self.hq_root}")
        logger.info(f"  CRF Range:     {self.crf_range}")
        logger.info(f"  Preset Range:  {self.preset_range}")
        logger.info(f"  Patch Size:    {self.patch_size}x{self.patch_size}")
        logger.info(f"  Crop Mode:     {self.crop_mode}")
        logger.info(f"  Augmentation:  {self.augment}")
        logger.info(f"  Normalization: {norm_range}")
        logger.info("="*60)
    
    def _create_geometric_pipeline(
        self, 
        patch_size: int, 
        crop_mode: str, 
        augment: bool
    ) -> T.Compose:
        """Builds a v2 transform pipeline for *geometric* ops (on PIL)."""
        transforms_list = []

        if patch_size > 0:
            if crop_mode == 'random':
                transforms_list.append(T.RandomCrop(
                    (patch_size, patch_size),
                    pad_if_needed=True,
                    padding_mode='reflect'
                ))
            elif crop_mode == 'center':
                transforms_list.append(T.CenterCrop((patch_size, patch_size)))
            elif crop_mode != 'none':
                raise ValueError(f"Invalid crop_mode: '{crop_mode}'. Must be 'random', 'center', or 'none'.")
        
        if augment:
            transforms_list.extend([
                T.RandomHorizontalFlip(p=0.5),
                T.RandomVerticalFlip(p=0.5),
            ])
        
        if not transforms_list:
            return T.Identity()
            
        return T.Compose(transforms_list)

    def _create_tensor_pipeline(
        self, 
        norm_range: Tuple[int, int]
    ) -> T.Compose:
        """Builds a v2 transform pipeline for *tensor* ops."""
        transforms_list = [
            T.ToImage(), # PIL -> Tensor [C, H, W], uint8
            T.ToDtype(torch.float32, scale=True) # [0, 255] -> [0, 1]
        ]
        
        if norm_range == (-1, 1):
            transforms_list.append(T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]))
        elif norm_range != (0, 1):
            raise ValueError(f"Invalid norm_range: {norm_range}.")
            
        return T.Compose(transforms_list)

    @staticmethod
    def _validate_range(
        rng: Optional[Tuple[int, int]], 
        valid_rng: Tuple[int, int],
        name: str
    ) -> Tuple[int, int]:
        """Validate parameter range is within valid bounds."""
        if rng is None: return valid_rng
        min_val, max_val = rng
        valid_min, valid_max = valid_rng
        if not (valid_min <= min_val <= max_val <= valid_max):
            raise ValueError(f"{name} range {rng} invalid.")
        return (min_val, max_val)
    
    def _build_index(self) -> List[Dict[str, Any]]:
        """Scan LQ directory for .avif files, filter, and match with HQ."""
        pairs = []
        lq_files = list(self.lq_root.rglob("*.avif"))
        if not lq_files:
            raise FileNotFoundError(f"No AVIF files found in {self.lq_root}")
        
        logger.info(f"Found {len(lq_files):,} AVIF files, filtering...")
        skipped_crf, skipped_preset, missing_hq = 0, 0, 0
        
        for lq_path in tqdm(lq_files, desc="Indexing", unit="file"):
            match = self.FILENAME_PATTERN.match(lq_path.name)
            if not match: continue
            
            base, crf_str, preset_str = match.groups()
            crf, preset = int(crf_str), int(preset_str)
            
            if not (self.crf_range[0] <= crf <= self.crf_range[1]):
                skipped_crf += 1; continue
            if not (self.preset_range[0] <= preset <= self.preset_range[1]):
                skipped_preset += 1; continue
            
            hq_path = self.hq_root / f"{base}{self.hq_ext}"
            if not hq_path.exists():
                missing_hq += 1; continue
            
            pairs.append({
                "lq": lq_path, "hq": hq_path, "crf": crf,
                "preset": preset, "base": base
            })
        
        if skipped_crf > 0: logger.info(f"Filtered {skipped_crf:,} files (CRF out of range)")
        if skipped_preset > 0: logger.info(f"Filtered {skipped_preset:,} files (preset out of range)")
        if missing_hq > 0: logger.warning(f"Skipped {missing_hq:,} files (missing HQ pair)")
        
        return pairs
    
    def __len__(self) -> int:
        return len(self.image_pairs)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Loads PIL images, applies cheap PIL-space geometric transforms,
        then applies expensive tensor conversion/normalization.
        
        Args:
            idx (int): The index of the item to fetch.
            
        Returns:
            Dict[str, Any]: A dictionary containing 'lq', 'hq', 'crf', 
                            and 'preset' tensors.
        """
        meta = self.image_pairs[idx]
        try:
            lq_img = Image.open(meta["lq"]).convert("RGB")
            hq_img = Image.open(meta["hq"]).convert("RGB")

            # Apply geometric pipeline (crop/flip) with a shared seed
            seed = torch.randint(0, 2**32, (1,)).item()
            torch.manual_seed(seed); lq_patch = self.geometric_transform(lq_img)
            torch.manual_seed(seed); hq_patch = self.geometric_transform(hq_img)
            
            # Apply tensor pipeline (ToImage, ToDtype, Normalize)
            lq_tensor = self.tensor_transform(lq_patch)
            hq_tensor = self.tensor_transform(hq_patch)
            
            result = {
                "lq": lq_tensor, "hq": hq_tensor,
                "crf": torch.tensor([meta["crf"]], dtype=torch.float32),
                "preset": torch.tensor([meta["preset"]], dtype=torch.float32),
            }
            
            if self.return_metadata:
                result["lq_path"] = str(meta["lq"])
                result["base_name"] = meta["base"]
            
            return result
            
        except (IOError, UnidentifiedImageError, OSError) as e:
            logger.warning(
                f"Corrupted file at index {idx}: {meta['lq']}. "
                f"Sampling another. Error: {e}"
            )
            random_idx = torch.randint(0, len(self), (1,)).item()
            return self.__getitem__(random_idx)

# ============================================================================
# SECTION 2: HIGH-PERFORMANCE DATASET (NumPy-based)
# ============================================================================

class AV1DatasetFast(AV1Dataset):
    """
    High-performance dataset for pre-processed .npy files.
    
    Inherits from AV1Dataset and overrides file indexing and loading
    to use .npy files and an all-tensor transform pipeline.
    """
    
    FILENAME_PATTERN = re.compile(r"^(.+?)_crf(\d+)_p(\d+)\.npy$")

    def __init__(self, **kwargs):
        """
        Initializes the Fast Dataset.
        Forces hq_ext to '.npy' as it expects preprocessed HQ files too.
        """
        kwargs['hq_ext'] = '.npy'
        super().__init__(**kwargs)

    def _log_init(self, norm_range):
        """Override logging to specify Fast loader."""
        logger.info("="*60)
        logger.info(f"Initializing {self.__class__.__name__} (NumPy Loader)")
        logger.info(f"  LQ Root:       {self.lq_root}")
        logger.info(f"  HQ Root:       {self.hq_root} (forced .npy)")
        logger.info(f"  CRF Range:     {self.crf_range}")
        logger.info(f"  Preset Range:  {self.preset_range}")
        logger.info(f"  Patch Size:    {self.patch_size}x{self.patch_size}")
        logger.info(f"  Crop Mode:     {self.crop_mode}")
        logger.info(f"  Augmentation:  {self.augment}")
        logger.info(f"  Normalization: {norm_range}")
        logger.info("="*60)
        
    def _create_tensor_pipeline(
        self, 
        norm_range: Tuple[int, int]
    ) -> T.Compose:
        """Override: Builds a tensor-to-tensor pipeline (no ToImage)."""
        transforms_list = [
            T.ToDtype(torch.float32, scale=True) # uint8 -> [0, 1]
        ]
        
        if norm_range == (-1, 1):
            transforms_list.append(T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]))
        elif norm_range != (0, 1):
            raise ValueError(f"Invalid norm_range: {norm_range}.")
            
        return T.Compose(transforms_list)

    def _build_index(self) -> List[Dict[str, Any]]:
        """Override: Scan for .npy files."""
        pairs = []
        lq_files = list(self.lq_root.rglob("*.npy"))
        if not lq_files:
            raise FileNotFoundError(
                f"No .npy files found in {self.lq_root}. "
                f"Did you run the preprocessing script?"
            )
        
        logger.info(f"Found {len(lq_files):,} NPY files, filtering...")
        skipped_crf, skipped_preset, missing_hq = 0, 0, 0
        
        for lq_path in tqdm(lq_files, desc="Indexing .npy", unit="file"):
            match = self.FILENAME_PATTERN.match(lq_path.name)
            if not match: continue
            
            base, crf_str, preset_str = match.groups()
            crf, preset = int(crf_str), int(preset_str)
            
            if not (self.crf_range[0] <= crf <= self.crf_range[1]):
                skipped_crf += 1; continue
            if not (self.preset_range[0] <= preset <= self.preset_range[1]):
                skipped_preset += 1; continue
            
            hq_path = self.hq_root / f"{base}.npy"
            if not hq_path.exists():
                missing_hq += 1; continue
            
            pairs.append({
                "lq": lq_path, "hq": hq_path, "crf": crf,
                "preset": preset, "base": base
            })
        
        if skipped_crf > 0: logger.info(f"Filtered {skipped_crf:,} files (CRF out of range)")
        if skipped_preset > 0: logger.info(f"Filtered {skipped_preset:,} files (preset out of range)")
        if missing_hq > 0: logger.warning(f"Skipped {missing_hq:,} files (missing HQ .npy pair)")
        
        return pairs

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Override: Load .npy arrays and use an all-tensor transform pipeline.
        This is the SOTA implementation that avoids the PIL bottleneck.
        """
        meta = self.image_pairs[idx]
        try:
            # --- SOTA: Load .npy and convert to tensor ONCE ---
            lq_array = np.load(meta["lq"]) # [H, W, 3], uint8
            hq_array = np.load(meta["hq"])
            
            lq_tensor_uint8 = torch.from_numpy(lq_array).permute(2, 0, 1) # [3, H, W]
            hq_tensor_uint8 = torch.from_numpy(hq_array).permute(2, 0, 1)
            
            # Apply geometric pipeline (crop/flip) with a shared seed
            seed = torch.randint(0, 2**32, (1,)).item()
            torch.manual_seed(seed); lq_patch = self.geometric_transform(lq_tensor_uint8)
            torch.manual_seed(seed); hq_patch = self.geometric_transform(hq_tensor_uint8)

            # Apply tensor pipeline (ToDtype, Normalize)
            lq_tensor = self.tensor_transform(lq_patch)
            hq_tensor = self.tensor_transform(hq_patch)
            
            result = {
                "lq": lq_tensor, "hq": hq_tensor,
                "crf": torch.tensor([meta["crf"]], dtype=torch.float32),
                "preset": torch.tensor([meta["preset"]], dtype=torch.float32),
            }
            
            if self.return_metadata:
                result["lq_path"] = str(meta["lq"])
                result["base_name"] = meta["base"]
            
            return result
            
        except Exception as e:
            logger.warning(
                f"Corrupted .npy file at index {idx}: {meta['lq']}. "
                f"Sampling another. Error: {e}"
            )
            random_idx = torch.randint(0, len(self), (1,)).item()
            return self.__getitem__(random_idx)

# ============================================================================
# SECTION 3: STRATEGY "MIXIN" CLASSES
# ============================================================================

class _VirtualAugment(object):
    """
    Mixin Class for Virtual Training Augmentation.
    
    This class is not a standalone Dataset. It's designed to be
    combined with a 'Loader' class (like AV1Dataset or AV1DatasetFast)
    using multiple inheritance.
    
    It overrides __len__ and __getitem__ to implement N-crops-per-image,
    solving the CPU/IO bottleneck of loading the same file repeatedly.
    
    e.g., class FinalDataset(_VirtualAugment, AV1DatasetFast): pass
    """
    
    def __init__(self, augment_factor: int = 1, **kwargs):
        """
        Initializes the virtual augmentation strategy.
        
        Args:
            augment_factor (int): The number of virtual crops per image.
            **kwargs: Arguments passed to the next class in the MRO (the Loader).
        """
        if augment_factor < 1:
            raise ValueError(f"augment_factor must be >= 1, got {augment_factor}")
        
        # Force training-specific settings on the base loader
        kwargs['crop_mode'] = 'random'
        kwargs['augment'] = True # 'augment' here means flip_augment
        
        # Call the next class in the Method Resolution Order (MRO)
        # This will be AV1Dataset or AV1DatasetFast
        super().__init__(**kwargs) 
        
        self.augment_factor = augment_factor
        self.base_length = super().__len__()
        
        logger.info("=" * 60)
        logger.info(f"Mixing in _VirtualAugment")
        logger.info(f"  Base images: {self.base_length:,}")
        logger.info(f"  Augment factor: {augment_factor}")
        logger.info(f"  Effective size: {len(self):,} samples/epoch")
        logger.info("=" * 60)
    
    def __len__(self) -> int:
        """Return augmented dataset size."""
        return self.base_length * self.augment_factor
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Maps the augmented index back to a base image index.
        
        This is the core of the SOTA strategy. The DataLoader asks for
        items 0, 1, 2, 3... (up to N * base_length). This function maps
        them back, e.g., (0, 1, 2, 3) all map to base_idx 0.
        
        It then calls the parent's (Base/Fast) __getitem__(0), which
        loads the file *once* and applies a *new* random crop *every time*.
        This distributes the load and solves the CPU bottleneck.
        """
        base_idx = idx // self.augment_factor
        return super().__getitem__(base_idx)


class _DeterministicGrid(object):
    """
    Mixin Class for Deterministic Grid Validation.
    
    Overrides __len__ and __getitem__ to provide a deterministic,
    centered grid of patches for stable validation metrics.
    
    e.g., class FinalDataset(_DeterministicGrid, AV1Dataset): pass
    """
    
    def __init__(
        self,
        augment_factor: int = 1,
        **kwargs
    ):
        """
        Initializes the grid strategy.
        
        Args:
            augment_factor (int): The number of grid patches. Must be a
                                  perfect square (1, 4, 9, 16...).
            **kwargs: Arguments passed to the next class in the MRO (the Loader).
        """
        is_grid = (augment_factor > 0) and \
                  (math.sqrt(augment_factor) == int(math.sqrt(augment_factor)))
        
        if not is_grid:
            raise ValueError(
                f"val_augment_factor ({augment_factor}) must be a perfect square."
            )
        self.grid_side = int(math.sqrt(augment_factor))
        
        # Force validation-specific settings on the base loader
        kwargs['crop_mode'] = 'none' # We handle all cropping manually
        kwargs['augment'] = False    # No random flips for validation
        
        super().__init__(**kwargs)
        
        self.augment_factor = augment_factor
        
        # SOTA FIX v5.2: We must build the patch map *after* the parent
        # __init__ has run and self.image_pairs has been populated.
        self._build_patch_map()
        
        logger.info("=" * 60)
        logger.info(f"Mixing in _DeterministicGrid")
        logger.info(f"  Base images: {len(self.image_pairs):,}")
        logger.info(f"  Grid layout: {self.grid_side}x{self.grid_side}")
        logger.info(f"  Patch size: {self.patch_size}x{self.patch_size}")
        logger.info(f"  Total patches: {len(self.patch_map):,}")
        if self.skipped_images > 0:
            logger.warning(f"  Skipped images (too small): {self.skipped_images}")
        logger.info("=" * 60)
    
    def _get_image_size(self, meta: Dict[str, Any]) -> Optional[Tuple[int, int]]:
        """
        SOTA FIX (v5.2): Read image dimensions super-fast.
        For .npy, loads the array (unavoidable but done once).
        For .avif, reads only the PIL header (extremely fast).
        """
        try:
            if isinstance(self, AV1DatasetFast):
                # .npy: Must load the array to get its shape.
                arr = np.load(meta['lq'])
                return arr.shape[1], arr.shape[0] # W, H
            else:
                # PIL: This is EXTREMELY fast.
                with Image.open(meta['lq']) as img:
                    return img.size # W, H
        except Exception as e:
            logger.warning(f"Cannot read dimensions for {meta['lq']}: {e}")
            return None

    def _build_patch_map(self):
        """Pre-calculate centered grid coordinates for all images."""
        self.patch_map = []
        self.skipped_images = 0
        grid_total_size = self.grid_side * self.patch_size
        
        logger.info("Pre-calculating centered grid patch coordinates...")
        # Use disable=None to auto-disable tqdm for non-TTY (like logs)
        for img_idx, meta in enumerate(tqdm(self.image_pairs, 
                                            desc="Mapping grid", 
                                            disable=None)):
            
            size = self._get_image_size(meta)
            if size is None:
                self.skipped_images += 1
                continue
            
            img_w, img_h = size
            
            if img_w < grid_total_size or img_h < grid_total_size:
                self.skipped_images += 1
                continue
            
            # Calculate top-left offset for the *entire* grid
            offset_x = (img_w - grid_total_size) // 2
            offset_y = (img_h - grid_total_size) // 2
            
            # Generate patch coordinates
            for row in range(self.grid_side):
                for col in range(self.grid_side):
                    patch_y = offset_y + (row * self.patch_size)
                    patch_x = offset_x + (col * self.patch_size)
                    self.patch_map.append({
                        'img_idx': img_idx,
                        'crop_y': patch_y,
                        'crop_x': patch_x
                    })
        
        if not self.patch_map and self.skipped_images > 0:
            raise ValueError(
                f"No valid images for grid sampling. All {len(self.image_pairs)} "
                f"images are smaller than required {grid_total_size}x{grid_total_size}."
            )
    
    def __len__(self) -> int:
        """Return total number of grid patches."""
        return len(self.patch_map)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Load a specific, pre-calculated grid patch.
        
        SOTA Strategy (v5.3 - Performance Fix):
        1. Loads the *full* PIL/NPY image (fast).
        2. Applies a *deterministic PIL/Tensor crop* (fast).
        3. Applies the *tensor conversion/normalization* (fast, on small patch).
        This avoids normalizing the full-size image.
        """
        patch_info = self.patch_map[idx]
        img_idx = patch_info['img_idx']
        crop_y, crop_x = patch_info['crop_y'], patch_info['crop_x']
        
        meta = self.image_pairs[img_idx]
        
        try:
            # --- 1. Load Full Image (PIL or NPY) ---
            # We bypass the parent __getitem__ to load the raw image,
            # which is necessary for our manual cropping pipeline.
            if isinstance(self, AV1DatasetFast):
                lq_array = np.load(meta["lq"])
                hq_array = np.load(meta["hq"])
                lq_full = torch.from_numpy(lq_array).permute(2, 0, 1) # [3, H, W] uint8
                hq_full = torch.from_numpy(hq_array).permute(2, 0, 1) # [3, H, W] uint8
            else:
                lq_full = Image.open(meta["lq"]).convert("RGB") # PIL Image
                hq_full = Image.open(meta["hq"]).convert("RGB") # PIL Image

            # --- 2. Apply Deterministic Crop (Fast) ---
            lq_patch = T.functional.crop(
                lq_full, crop_y, crop_x, self.patch_size, self.patch_size
            )
            hq_patch = T.functional.crop(
                hq_full, crop_y, crop_x, self.patch_size, self.patch_size
            )
            
            # --- 3. Apply Tensor/Normalization Pipeline (Fast) ---
            # We use the tensor_transform from the base class.
            # For .npy, we must use a separate pipeline that skips ToImage().
            if isinstance(self, AV1DatasetFast):
                # Lazily create and cache the fast tensor transform
                if not hasattr(self, '_tensor_transform_fast'):
                    # This re-uses the *base* class's method
                    self._tensor_transform_fast = super()._create_tensor_pipeline(self.norm_range)
                
                lq_tensor = self._tensor_transform_fast(lq_patch)
                hq_tensor = self._tensor_transform_fast(hq_patch)
            else:
                # Use the standard PIL->Tensor pipeline
                lq_tensor = self.tensor_transform(lq_patch)
                hq_tensor = self.tensor_transform(hq_patch)
            
            result = {
                "lq": lq_tensor, "hq": hq_tensor,
                "crf": torch.tensor([meta["crf"]], dtype=torch.float32),
                "preset": torch.tensor([meta["preset"]], dtype=torch.float32),
            }
            
            if self.return_metadata:
                result["lq_path"] = str(meta["lq"])
                result["base_name"] = f"{meta['base']}_grid{idx % self.augment_factor}"
            
            return result
            
        except Exception as e:
            logger.error(f"Error loading grid patch {idx} from {meta['lq']}: {e}")
            raise

# ============================================================================
# SECTION 4: PICKLE-SAFE SOTA COMBINATIONS (MODULE LEVEL)
# ============================================================================

# --- These are the final, pickle-safe classes the factory will use ---
# They are defined at the top level of the module so pickle can find them.
# This fixes the "Can't pickle local object" error from multiprocessing.

class AV1Train(AV1Dataset):
    """Standard 1-to-1 PIL-based training dataset."""
    pass

class AV1TrainAugmented(_VirtualAugment, AV1Dataset):
    """PIL-based N-crop training dataset."""
    pass

class AV1Val(AV1Dataset):
    """Standard 1-to-1 PIL-based validation dataset."""
    pass
    
class AV1ValGrid(_DeterministicGrid, AV1Dataset):
    """PIL-based grid validation dataset."""
    pass

# --- Fast .npy versions ---
class AV1FastTrain(AV1DatasetFast):
    """Standard 1-to-1 .npy-based training dataset."""
    pass

class AV1FastTrainAugmented(_VirtualAugment, AV1DatasetFast):
    """High-performance .npy-based N-crop training dataset."""
    pass
    
class AV1FastVal(AV1DatasetFast):
    """Standard 1-to-1 .npy-based validation dataset."""
    pass

class AV1FastValGrid(_DeterministicGrid, AV1DatasetFast):
    """High-performance .npy-based grid validation dataset."""
    pass

# Mark as available for the factory
FAST_DATASET_AVAILABLE = True


# ============================================================================
# SECTION 5: PUBLIC DATASET FACTORY
# ============================================================================

def create_dataset(
    lq_root: str,
    hq_root: str,
    hq_ext: str,
    lq_ext: str,
    patch_size: int,
    crop_mode: str,
    augment_factor: int,
    crf_range: Tuple[int, int],
    preset_range: Tuple[int, int],
    norm_range: Tuple[int, int],
    # [NEW] Added flip_augment as a top-level parameter
    flip_augment: Optional[bool] = None, 
    cached_image_pairs: Optional[list] = None,
    return_metadata: bool = False
) -> Dataset:
    """
    Intelligently creates and returns the correct dataset instance
    by selecting a pre-defined, pickle-safe class.
    
    This is the single public function to be imported by the trainer.
    
    Args:
        lq_root (str): Path to Low-Quality (LQ) images.
        hq_root (str): Path to High-Quality (HQ) images.
        hq_ext (str): File extension for HQ images (e.g., '.png').
        lq_ext (str): File extension for LQ images (e.g., '.avif', '.npy').
        patch_size (int): The square crop size.
        crop_mode (str): 'random' or 'center'. This field
                         drives the logic for train/val selection.
        augment_factor (int): 
            For 'random' crop: Number of virtual crops per image (e.g., 16).
            For 'center' crop: Number of grid patches (e.g., 4, 9, 16).
        crf_range (tuple): (min, max) CRF values to include.
        preset_range (tuple): (min, max) preset values to include.
        norm_range (tuple): (min, max) normalization range, e.g., (-1, 1).
        flip_augment (bool, optional): Explicitly enable/disable flips.
            If None, defaults to True for 'random' crop and False for 'center'.
        cached_image_pairs (list, optional): Pre-scanned file list.
        return_metadata (bool, optional): If True, items will include metadata.
        
    Returns:
        torch.utils.data.Dataset: The configured dataset instance.
        
    Raises:
        ValueError: If crop_mode is invalid or augment_factor is invalid
                    for grid validation.
    """
    
    # --- 1. Aggregate Common Parameters ---
    common_args = {
        'lq_root_dir': lq_root,
        'hq_root_dir': hq_root,
        'hq_ext': hq_ext,
        'patch_size': patch_size,
        'crf_range': crf_range,
        'preset_range': preset_range,
        'norm_range': norm_range,
        'cached_image_pairs': cached_image_pairs,
        'return_metadata': return_metadata
    }
    
    use_fast_loader = (lq_ext == '.npy' and FAST_DATASET_AVAILABLE)

    # --- 2. S.O.T.A. Logic Branching (driven by crop_mode) ---

    if crop_mode == 'random':
        # --- TRAINING MODE ---
        # Default to True for flips if not specified
        common_args['augment'] = True if flip_augment is None else flip_augment
        common_args['crop_mode'] = 'random'
        
        if augment_factor > 1:
            # --- Strategy: Virtual Augmentation ---
            LoaderClass = AV1FastTrainAugmented if use_fast_loader else AV1TrainAugmented
            logger.info(f"Factory: Using {LoaderClass.__name__} (x{augment_factor} virtual crops)")
            return LoaderClass(augment_factor=augment_factor, **common_args)
        else:
            # --- Strategy: Standard 1-to-1 Training ---
            LoaderClass = AV1FastTrain if use_fast_loader else AV1Train
            logger.info(f"Factory: Using {LoaderClass.__name__} for training.")
            return LoaderClass(**common_args)

    elif crop_mode == 'center':
        # --- VALIDATION / TEST MODE ---
        # Default to False for flips if not specified
        common_args['augment'] = False if flip_augment is None else flip_augment
        
        is_grid = (augment_factor > 1) and \
                  (math.sqrt(augment_factor) == int(math.sqrt(augment_factor)))
        
        if is_grid:
            # --- Strategy: Deterministic Grid ---
            LoaderClass = AV1FastValGrid if use_fast_loader else AV1ValGrid
            logger.info(f"Factory: Using {LoaderClass.__name__} ({augment_factor}-patch grid)")
            return LoaderClass(augment_factor=augment_factor, **common_args)
        else:
            # --- Strategy: Standard Center Crop ---
            if augment_factor > 1: # User passed 5, 6, etc.
                logger.warning(
                    f"Factory: augment_factor ({augment_factor}) is not a "
                    f"perfect square. Falling back to single center-crop."
                )
            LoaderClass = AV1FastVal if use_fast_loader else AV1Val
            logger.info(f"Factory: Using {LoaderClass.__name__} (single center-crop).")
            return LoaderClass(crop_mode='center', **common_args)
            
    else:
        raise ValueError(
            f"Unknown crop_mode for factory: '{crop_mode}'. "
            f"Must be 'random' (for training) or 'center' (for validation)."
        )

# --- 5. Public API Control ---
# This ensures that `from utils.av1_dataset import *`
# only imports the factory function, hiding the internal classes.
__all__ = ['create_dataset']
