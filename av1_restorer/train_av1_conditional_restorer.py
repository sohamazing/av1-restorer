# train_av1_conditional_restorer.py
"""
Training Script for Conditional AV1 U-Net Restorers

Supports both CRF-only and CRF+Preset conditioning modes.
Automatically selects based on preset_range in config.

==============================================================================
FEATURES
==============================================================================
- Automatic model selection: CRF-only if preset_range is single value
- Pre-processed (.npy) and on-the-fly (.avif) dataset loading
- Full curriculum learning (progressive patch sizes, CRF ranges)
- SOTA training: Mixed Precision (AMP), EMA, Warmup + Cosine Annealing
- W&B integration for experiment tracking
- Robust checkpointing with resume support
- Cross-platform (CUDA, MPS, CPU) via torch.amp

==============================================================================
USAGE
==============================================================================
Basic:
    python train_conditional_unet.py --config configs/unet_base.yaml

Resume:
    python train_conditional_unet.py --config configs/unet_base.yaml --resume latest

Resume W&B:
    python train_conditional_unet.py --config configs/unet_base.yaml \
        --resume best --wandb_id abc123def

==============================================================================
Author: Soham Mukherjee
Version: 2.0 (Conditional U-Net with Auto CRF-only Detection)
License: MIT
==============================================================================
"""

import os
import sys
import yaml
import argparse
import logging
from pathlib import Path
from typing import Dict, Any, Tuple, Optional
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from tqdm import tqdm

# Cross-platform AMP support
import torch.amp

# Project imports
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

# from av1_restorer.models.av1_conditional_unet_restorer import create_av1_restorer # V1
from av1_restorer.models.av1_conditional_unet_restorer_v2 import create_av1_restorer # V2

try:
    from utils.loss import CombinedLoss
    from utils.av1_dataset import AV1Dataset
    from utils.av1_dataset_fast import AV1DatasetFast
except ImportError as e:
    print(f"Failed to import utilities: {e}")
    sys.exit(1)

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)-8s | %(message)s')
logger = logging.getLogger("ConditionalUNetTrainer")


# ==============================================================================
# SECTION 1: Exponential Moving Average (EMA)
# ==============================================================================

class EMA:
    """
    Exponential Moving Average for stable model evaluation.
    
    Maintains shadow parameters updated as:
        shadow = decay × shadow + (1 - decay) × model_params
    
    Args:
        model: Neural network model
        decay: EMA decay rate (0.9999 recommended, ~10K step average)
    """
    def __init__(self, model: nn.Module, decay: float = 0.9999):
        self.model = model
        self.decay = decay
        self.shadow = {
            name: param.data.clone() 
            for name, param in model.named_parameters() 
            if param.requires_grad
        }
        self.backup = {}
    
    def update(self):
        """Update shadow parameters with current model parameters."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name].data.mul_(self.decay).add_(
                    param.data, alpha=1.0 - self.decay
                )
    
    def apply_shadow(self):
        """Temporarily replace model parameters with EMA shadow for validation."""
        self.backup = {
            name: param.data.clone()
            for name, param in self.model.named_parameters()
            if param.requires_grad
        }
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data.copy_(self.shadow[name])
    
    def restore(self):
        """Restore original model parameters after validation."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data.copy_(self.backup[name])
        self.backup = {}


# ==============================================================================
# SECTION 2: Conditional U-Net Trainer
# ==============================================================================

class ConditionalUNetTrainer:
    """
    Config-driven trainer for Conditional AV1 U-Net Restorers.
    
    Automatically detects CRF-only vs CRF+Preset based on preset_range:
        - If preset_range is single value (e.g., [4,4]): CRF-only mode
        - Otherwise: CRF+Preset mode
    """
    
    def __init__(
        self,
        config_path: str,
        resume_from: Optional[str] = None,
        wandb_id: Optional[str] = None
    ):
        self.config = self._load_config(config_path)
        self.resume_from = resume_from
        self.wandb_id = wandb_id

        # Dataset file list caching (avoid rescanning directories per stage)
        self.train_image_pairs_cache = {}
        self.val_image_pairs_cache = {}
        
        # System setup (device, seeds, AMP)
        self._setup_system()
        
        # Model and loss
        self.model = self._setup_model()
        self.loss_fn = self._setup_loss()
        
        # EMA if enabled
        ema_cfg = self.config['optimizer']
        self.ema = None
        if ema_cfg.get('use_ema', False):
            self.ema = EMA(self.model, decay=ema_cfg.get('ema_decay', 0.9999))
            logger.info(f"EMA enabled (decay={ema_cfg.get('ema_decay', 0.9999)})")
        
        # W&B logging
        self._setup_logging()
        
        # Summary and validation samples
        self._print_summary()
        self.val_samples = self._get_fixed_val_samples()
        
        # Training state
        self.start_stage = 0
        self.start_epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')
        self.saved_optimizer_state = None
        self.saved_scheduler_state = None
        
        # Load checkpoint if resuming
        if self.resume_from:
            self._load_checkpoint()
    
    # --------------------------------------------------------------------------
    # Setup Methods
    # --------------------------------------------------------------------------
    
    def _load_config(self, path: str) -> Dict[str, Any]:
        """Load and validate YAML configuration."""
        path = Path(path).expanduser()
        if not path.exists():
            raise FileNotFoundError(f"Config not found: {path}")
        
        with open(path, 'r') as f:
            config = yaml.safe_load(f)
        
        logger.info(f"Config loaded: {path}")
        return config
    
    def _setup_system(self):
        """Setup device, random seeds, checkpointing, and AMP."""
        sys_cfg = self.config['system']
        
        # Device selection (auto-detect: cuda > mps > cpu)
        device_str = sys_cfg.get('device', 'auto')
        if device_str == 'auto':
            if torch.cuda.is_available():
                self.device = torch.device('cuda')
                logger.info(f"Device: CUDA ({torch.cuda.get_device_name(0)})")
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                self.device = torch.device('mps')
                logger.info("Device: Apple MPS")
            else:
                self.device = torch.device('cpu')
                logger.warning("Device: CPU (training will be slow)")
        else:
            self.device = torch.device(device_str)
            logger.info(f"Device: {self.device}")
        
        # Random seed for reproducibility
        seed = sys_cfg.get('seed', 42)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        logger.info(f"Random seed: {seed}")
        
        # Checkpoint directory
        self.checkpoint_dir = Path(self.config['checkpoint']['dir']).expanduser()
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Checkpoints: {self.checkpoint_dir}")
        
        # Mixed precision (AMP) - use torch.amp for cross-platform support
        self.use_amp = sys_cfg.get('mixed_precision', False)
        if self.use_amp and self.device.type == 'cuda':
            # Only CUDA supports GradScaler
            self.scaler = torch.amp.GradScaler('cuda', enabled=self.use_amp)
            logger.info("AMP: Enabled with GradScaler (CUDA)")
        elif self.use_amp:
            # MPS/CPU: AMP without GradScaler
            self.scaler = None
            logger.info(f"AMP: Enabled without GradScaler ({self.device.type})")
        else:
            self.scaler = None
            logger.info("AMP: Disabled")
    
    def _setup_model(self) -> nn.Module:
        """
        Create U-Net model with automatic CRF-only detection.
        
        Detection logic:
            - If preset_range[0] == preset_range[1]: CRF-only mode
            - Otherwise: CRF+Preset mode
        
        Sets self.model_needs_preset flag for forward pass logic.
        """
        model_cfg = self.config['model']
        dset_cfg = self.config['dataset']
        
        # Ensure model type is 'unet'
        model_type = model_cfg.get('type', 'unet')
        if model_type != 'unet':
            raise ValueError(
                f"This script only supports model type 'unet', "
                f"found '{model_type}' in config."
            )
        
        size = model_cfg['size']
        crf_range = tuple(dset_cfg['crf_range'])
        preset_range = tuple(dset_cfg.get('preset_range', [0, 8]))
        norm_range = tuple(dset_cfg.get('norm_range', [-1, 1]))
        
        logger.info(f"Creating model: {model_type} (size={size})")
        
        # Automatic CRF-only detection
        if preset_range[0] == preset_range[1]:
            logger.info(f"Preset range is single value ({preset_range[0]}). Using CRF-Only U-Net.")
            self.model_needs_preset = False
        else:
            logger.info("Using Standard U-Net (CRF + Preset).")
            self.model_needs_preset = True
        
        # Create model (factory handles internal conditioning setup)
        model = create_av1_restorer(
            size=size,
            crf_range=crf_range,
            preset_range=preset_range,
            norm_range=norm_range
        )
        
        return model.to(self.device)
    
    def _setup_loss(self) -> nn.Module:
        """Initialize loss function from config."""
        norm_range = tuple(self.config['dataset'].get('norm_range', [-1, 1]))
        loss_fn = CombinedLoss(self.config['loss'], norm_range=norm_range)
        logger.info("Loss function initialized")
        return loss_fn.to(self.device)
    
    def _setup_logging(self):
        """Initialize W&B if enabled and available."""
        if not (self.config['project']['log_to_wandb'] and WANDB_AVAILABLE):
            if self.config['project']['log_to_wandb']:
                logger.warning("W&B requested but not installed")
            return
        
        # Try to recover run ID from checkpoint
        run_id = self.wandb_id
        if self.resume_from and not run_id:
            try:
                ckpt_path = self._get_checkpoint_path(self.resume_from)
                if ckpt_path and ckpt_path.exists():
                    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
                    run_id = ckpt.get('wandb_id')
            except Exception as e:
                logger.warning(f"Could not read W&B ID from checkpoint: {e}")
        
        # Initialize W&B
        wandb.init(
            project=self.config['project']['name'],
            name=self.config['project']['experiment_name'],
            config=self.config,
            id=run_id,
            resume="allow"
        )
        self.wandb_id = wandb.run.id
        logger.info(f"W&B: {wandb.run.url}")
    
    def _print_summary(self):
        """Print training configuration summary."""
        logger.info("=" * 80)
        logger.info(f"{'AV1 Conditional U-Net - Training Configuration':^80}")
        logger.info("=" * 80)
        
        # Project info
        proj = self.config['project']
        logger.info(f"Project: {proj['name']}")
        logger.info(f"Experiment: {proj['experiment_name']}")
        
        # Model info
        model_cfg = self.config['model']
        params = sum(p.numel() for p in self.model.parameters())
        logger.info(f"Model: {model_cfg['type']} ({model_cfg['size']})")
        logger.info(f"Parameters: {params:,} ({params/1e6:.2f}M)")
        logger.info(f"Conditioning: CRF{' + Preset' if self.model_needs_preset else '-Only'}")
        
        # Dataset info
        dset_cfg = self.config['dataset']
        logger.info(f"CRF Range: {dset_cfg['crf_range']}")
        logger.info(f"Preset Range: {dset_cfg['preset_range']}")
        
        # Curriculum info
        logger.info("Curriculum:")
        for i, stage in enumerate(self.config['curriculum']):
            patch = stage['patch_size']
            batch = stage['batch_size']
            epochs = stage['epochs']
            crf = stage.get('crf_range', dset_cfg['crf_range'])
            
            if isinstance(patch, list):
                logger.info(
                    f"  Stage {i+1}: {epochs[0]}+{epochs[1]} epochs, "
                    f"{patch[0]}/{patch[1]}px, batch {batch[0]}/{batch[1]}, CRF {crf}"
                )
            else:
                logger.info(
                    f"  Stage {i+1}: {epochs} epochs, {patch}px, batch {batch}, CRF {crf}"
                )
        
        # Training config
        logger.info(f"Device: {self.device} | AMP: {self.use_amp} | EMA: {self.ema is not None}")
        logger.info("=" * 80)
    
    def _get_fixed_val_samples(self) -> Optional[Dict[str, Any]]:
        """Sample fixed validation images for W&B visualization."""
        try:
            data_cfg = self.config['data']
            dset_cfg = self.config['dataset']
            num_samples = self.config['training'].get('num_val_samples_to_log', 4)
            
            # Get patch size from last curriculum stage
            last_stage = self.config['curriculum'][-1]
            patch_size = last_stage['patch_size']
            if isinstance(patch_size, list):
                patch_size = patch_size[-1]

            # Check cache for file list
            val_cache_key = f"{data_cfg['val_lq_root']}_{tuple(dset_cfg['crf_range'])}"
            cached_val_image_pairs = self.val_image_pairs_cache.get(val_cache_key)
            
            # Create validation dataset
            val_dataset = AV1Dataset(
                lq_root_dir=data_cfg['val_lq_root'],
                hq_root_dir=data_cfg['val_hq_root'],
                hq_ext=data_cfg.get('hq_ext', '.png'),
                patch_size=patch_size,
                crf_range=tuple(dset_cfg['crf_range']),
                preset_range=tuple(dset_cfg['preset_range']),
                norm_range=tuple(dset_cfg.get('norm_range', [-1, 1])),
                crop_mode='center',
                augment=False,
                return_metadata=True,
                cached_image_pairs=cached_val_image_pairs
            )

            # Cache file list for future use
            if cached_val_image_pairs is None and hasattr(val_dataset, 'image_pairs'):
                self.val_image_pairs_cache[val_cache_key] = val_dataset.image_pairs
            
            loader = DataLoader(val_dataset, batch_size=num_samples, shuffle=False)
            batch = next(iter(loader))
            
            # Move to device
            result = {
                k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }
            
            logger.info(f"Sampled {num_samples} validation images for visualization")
            return result
        
        except Exception as e:
            logger.warning(f"Could not sample validation images: {e}")
            return None
    
    # --------------------------------------------------------------------------
    # Dataset & Optimizer Setup
    # --------------------------------------------------------------------------
    
    def _create_dataloaders(
        self,
        stage_config: dict
    ) -> Tuple[DataLoader, DataLoader]:
        """Create train and validation dataloaders for a curriculum stage."""
        data_cfg = self.config['data']
        dset_cfg = self.config['dataset']
        sys_cfg = self.config['system']
        
        # Extract stage parameters
        patch_size = stage_config['patch_size']
        batch_size = stage_config['batch_size']
        crf_range = stage_config.get('crf_range', dset_cfg['crf_range'])

        # Check cache for file lists
        train_cache_key = f"{data_cfg['train_lq_root']}_{tuple(crf_range)}"
        val_cache_key = f"{data_cfg['val_lq_root']}_{tuple(crf_range)}"
        cached_train_image_pairs = self.train_image_pairs_cache.get(train_cache_key)
        cached_val_image_pairs = self.val_image_pairs_cache.get(val_cache_key)
        
        # Handle progressive training (use first sub-stage for dataloader)
        if isinstance(patch_size, list):
            patch_size = patch_size[0]
            batch_size = batch_size[0]
        
        # Select dataset class based on file extension
        lq_ext = data_cfg.get('lq_ext', '.avif').lower()
        if lq_ext == '.npy' and 'AV1DatasetFast' in globals():
            DatasetClass = AV1DatasetFast
            logger.info("Using pre-processed .npy dataset (fast)")
        else:
            DatasetClass = AV1Dataset
            if lq_ext != '.avif':
                logger.warning(f"Using on-the-fly {lq_ext} loading (slow)")
        
        # Common dataset arguments
        common_args = {
            'patch_size': patch_size,
            'crf_range': tuple(crf_range),
            'preset_range': tuple(dset_cfg['preset_range']),
            'norm_range': tuple(dset_cfg.get('norm_range', [-1, 1])),
            'hq_ext': data_cfg.get('hq_ext', '.png')
        }
        
        # Create datasets
        train_dset = DatasetClass(
            lq_root_dir=data_cfg['train_lq_root'],
            hq_root_dir=data_cfg['train_hq_root'],
            cached_image_pairs=cached_train_image_pairs,
            crop_mode='random',
            augment=True,
            **common_args
        )
        
        val_dset = DatasetClass(
            lq_root_dir=data_cfg['val_lq_root'],
            hq_root_dir=data_cfg['val_hq_root'],
            cached_image_pairs=cached_val_image_pairs,
            crop_mode='center',
            augment=False,
            **common_args
        )

        # Cache file lists for future use
        if cached_train_image_pairs is None and hasattr(train_dset, 'image_pairs'):
            self.train_image_pairs_cache[train_cache_key] = train_dset.image_pairs
        if cached_val_image_pairs is None and hasattr(val_dset, 'image_pairs'):
            self.val_image_pairs_cache[val_cache_key] = val_dset.image_pairs
        
        # Dataloader config
        num_workers = sys_cfg.get('num_workers', 8)
        pin_memory = (self.device.type == 'cuda')
        
        train_loader = DataLoader(
            train_dset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=True,
            persistent_workers=(num_workers > 0),
            prefetch_factor=4 if num_workers > 0 else None
        )
        
        val_loader = DataLoader(
            val_dset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=max(2, num_workers // 2),
            pin_memory=pin_memory,
            drop_last=False,
            persistent_workers=(num_workers > 0),
            prefetch_factor=2 if num_workers > 0 else None
        )
        
        logger.info(
            f"Dataloaders: {len(train_dset):,} train, {len(val_dset):,} val samples"
        )
        
        return train_loader, val_loader
    
    def _setup_optimizer(
        self,
        total_steps: int
    ) -> Tuple[torch.optim.Optimizer, Optional[Any]]:
        """Create optimizer and LR scheduler with warmup."""
        opt_cfg = self.config['optimizer']
        sched_cfg = self.config['scheduler']
        
        # Optimizer
        lr = opt_cfg['lr']
        if opt_cfg['type'].lower() == 'adamw':
            optimizer = AdamW(
                self.model.parameters(),
                lr=lr,
                betas=tuple(opt_cfg.get('betas', [0.9, 0.999])),
                weight_decay=opt_cfg.get('weight_decay', 1e-4)
            )
        else:
            optimizer = Adam(
                self.model.parameters(),
                lr=lr,
                betas=tuple(opt_cfg.get('betas', [0.9, 0.999]))
            )
        
        logger.info(f"Optimizer: {opt_cfg['type'].upper()}, lr={lr}")
        
        # LR Scheduler with warmup
        scheduler_type = sched_cfg.get('type', 'cosine').lower()
        warmup_steps = sched_cfg.get('warmup_steps', 500)
        
        if scheduler_type == 'cosine':
            main_scheduler = CosineAnnealingLR(
                optimizer,
                T_max=max(1, total_steps - warmup_steps),
                eta_min=sched_cfg.get('min_lr', 1e-6)
            )
            warmup_scheduler = LinearLR(
                optimizer,
                start_factor=1e-5,
                end_factor=1.0,
                total_iters=warmup_steps
            )
            scheduler = SequentialLR(
                optimizer,
                schedulers=[warmup_scheduler, main_scheduler],
                milestones=[warmup_steps]
            )
            logger.info(f"Scheduler: Cosine with {warmup_steps}-step warmup")
        else:
            scheduler = None
            logger.info("Scheduler: None")
        
        return optimizer, scheduler
    
    # --------------------------------------------------------------------------
    # Training Loop Logic
    # --------------------------------------------------------------------------
    
    def train(self):
        """Main training loop orchestrating curriculum stages."""
        logger.info("\n" + "=" * 80)
        logger.info("STARTING TRAINING")
        logger.info("=" * 80 + "\n")

        curriculum = self.config['curriculum']
        if not isinstance(curriculum, list):
            raise ValueError("Config 'curriculum' must be a list of stages")

        # Estimate total steps across all stages for scheduler
        estimated_total_steps = 0
        stage_loader_lengths = []
        try:
            logger.info("Estimating total steps for scheduler...")
            for stage_cfg in curriculum:
                temp_train_loader, _ = self._create_dataloaders(stage_cfg)
                current_loader_len = len(temp_train_loader)
                stage_loader_lengths.append(current_loader_len)

                stage_epochs = stage_cfg['epochs']
                if isinstance(stage_epochs, list):
                    estimated_total_steps += sum(stage_epochs) * current_loader_len
                else:
                    estimated_total_steps += stage_epochs * current_loader_len
                del temp_train_loader
            logger.info(f"Estimated total steps: {estimated_total_steps:,}")
        except Exception as e:
            logger.warning(f"Could not estimate total steps: {e}. Using fallback.")
            estimated_total_steps = sum(
                sum(s['epochs']) if isinstance(s['epochs'], list) else s['epochs']
                for s in curriculum
            ) * 500
            stage_loader_lengths = [500] * len(curriculum)

        # Setup optimizer and scheduler once
        optimizer, scheduler = self._setup_optimizer(estimated_total_steps)

        # Restore optimizer/scheduler state if resuming from start
        if self.start_stage == 0 and self.start_epoch == 0 and self.saved_optimizer_state:
            logger.info("Restoring optimizer/scheduler state...")
            try:
                optimizer.load_state_dict(self.saved_optimizer_state)
                logger.info("Optimizer state restored")
                if scheduler and self.saved_scheduler_state:
                    scheduler.load_state_dict(self.saved_scheduler_state)
                    logger.info("Scheduler state restored")
            except Exception as e:
                logger.error(f"Failed to restore state: {e}")
            finally:
                self.saved_optimizer_state = None
                self.saved_scheduler_state = None

        # Curriculum loop
        for stage_idx in range(self.start_stage, len(curriculum)):
            stage_cfg = curriculum[stage_idx]
            logger.info(f"\n▶ Starting Stage {stage_idx + 1}/{len(curriculum)}")

            current_stage_loader_len = stage_loader_lengths[stage_idx]
            self._run_stage(stage_idx, stage_cfg, optimizer, scheduler, current_stage_loader_len)

            # Save stage completion checkpoint
            epochs_in_stage = stage_cfg['epochs']
            last_epoch_idx = (sum(epochs_in_stage) - 1) if isinstance(epochs_in_stage, list) else (epochs_in_stage - 1)
            self._save_checkpoint(
                last_epoch_idx, stage_idx, optimizer, scheduler, is_stage_complete=True
            )

            self.start_epoch = 0  # Reset for next stage

        # Training complete
        logger.info("\n" + "=" * 80)
        logger.info("🎉 TRAINING COMPLETE!")
        logger.info(f"Best validation loss: {self.best_val_loss:.6f}")
        logger.info("=" * 80 + "\n")
        
        if WANDB_AVAILABLE and self.config['project']['log_to_wandb'] and wandb.run:
            try:
                wandb.finish()
                logger.info("W&B run finished")
            except Exception as e:
                logger.error(f"Error finishing W&B: {e}")

    def _run_stage(
        self,
        stage_idx: int,
        stage_cfg: dict,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[Any],
        loader_len_estimate: int
    ):
        """Run a single curriculum stage with optional progressive sub-stages."""
        is_progressive = isinstance(stage_cfg.get('patch_size'), list)

        # Restore optimizer/scheduler state if resuming mid-stage
        if stage_idx == self.start_stage and self.start_epoch > 0 and self.saved_optimizer_state:
            logger.info(f"Restoring state for Stage {stage_idx+1} resume...")
            try:
                optimizer.load_state_dict(self.saved_optimizer_state)
                logger.info("Optimizer state restored")
                if scheduler and self.saved_scheduler_state:
                    scheduler.load_state_dict(self.saved_scheduler_state)
                    logger.info("Scheduler state restored")
            except Exception as e:
                logger.error(f"Failed to restore state: {e}")
            finally:
                self.saved_optimizer_state = None
                self.saved_scheduler_state = None

        if is_progressive:
            # Progressive stage with multiple sub-stages
            patch_sizes = stage_cfg['patch_size']
            batch_sizes = stage_cfg['batch_size']
            epochs_per_sub = stage_cfg['epochs']
            total_epochs = sum(epochs_per_sub)

            if not (len(patch_sizes) == len(batch_sizes) == len(epochs_per_sub)):
                raise ValueError(f"Stage {stage_idx+1}: Mismatched progressive lists")

            logger.info(f"Progressive stage: {len(patch_sizes)} sub-stages")
            cumulative_epochs = 0

            for sub_idx in range(len(patch_sizes)):
                sub_patch = patch_sizes[sub_idx]
                sub_batch = batch_sizes[sub_idx]
                sub_epochs = epochs_per_sub[sub_idx]

                # Skip completed sub-stages on resume
                if self.start_epoch >= cumulative_epochs + sub_epochs:
                    logger.info(f"  Skipping Sub-stage {stage_idx+1}.{sub_idx+1} (completed)")
                    cumulative_epochs += sub_epochs
                    continue

                logger.info(
                    f"  → Sub-stage {stage_idx+1}.{sub_idx+1}: "
                    f"{sub_epochs} epochs @ {sub_patch}px, batch {sub_batch}"
                )

                # Create dataloaders for this sub-stage
                sub_cfg = stage_cfg.copy()
                sub_cfg['patch_size'] = sub_patch
                sub_cfg['batch_size'] = sub_batch
                train_loader, val_loader = self._create_dataloaders(sub_cfg)

                # Determine start epoch within this sub-stage
                start_epoch_in_sub = max(0, self.start_epoch - cumulative_epochs)

                # Run epochs for this sub-stage
                self._run_epochs(
                    stage_idx, start_epoch_in_sub, sub_epochs,
                    cumulative_epochs, total_epochs,
                    train_loader, val_loader, optimizer, scheduler, sub_idx
                )

                cumulative_epochs += sub_epochs
                self.start_epoch = 0  # Reset for sequential runs

        else:
            # Simple stage with fixed parameters
            epochs = stage_cfg['epochs']
            logger.info(f"Simple stage: {epochs} epochs")
            train_loader, val_loader = self._create_dataloaders(stage_cfg)

            self._run_epochs(
                stage_idx, self.start_epoch, epochs,
                0, epochs,
                train_loader, val_loader, optimizer, scheduler, -1
            )

    def _run_epochs(
        self,
        stage_idx: int,
        start_epoch: int,
        num_epochs: int,
        epoch_offset: int,
        total_epochs: int,
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[Any],
        sub_stage_idx: int = -1
    ):
        """Run training and validation loop for specified epochs."""
        for epoch_in_segment in range(start_epoch, num_epochs):
            current_epoch = epoch_offset + epoch_in_segment

            # Train epoch
            self._train_epoch(
                current_epoch, total_epochs, stage_idx,
                train_loader, optimizer, scheduler, sub_stage_idx
            )

            # Validation and checkpointing
            self._validate_and_checkpoint(
                current_epoch, total_epochs, stage_idx,
                val_loader, optimizer, scheduler
            )

    def _train_epoch(
        self,
        epoch: int,
        total_epochs: int,
        stage_idx: int,
        train_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[Any],
        sub_stage_idx: int = -1
    ):
        """Train the model for one epoch."""
        self.model.train()

        # Setup progress bar
        stage_desc = f"Stage {stage_idx+1}"
        if sub_stage_idx != -1:
            stage_desc += f".{sub_stage_idx+1}"
        pbar_desc = f"{stage_desc} | Epoch {epoch+1}/{total_epochs}"
        pbar = tqdm(train_loader, desc=pbar_desc, leave=True, dynamic_ncols=True)

        # Accumulators
        epoch_total_loss = 0.0
        epoch_loss_components = defaultdict(float)

        for batch_idx, batch in enumerate(pbar):
            # Move data to device
            try:
                lq = batch['lq'].to(self.device, non_blocking=True)
                hq = batch['hq'].to(self.device, non_blocking=True)
                crf = batch['crf'].to(self.device, non_blocking=True)
                
                # Only fetch preset if model needs it
                preset = None
                if self.model_needs_preset:
                    preset = batch['preset'].to(self.device, non_blocking=True)
            except KeyError as e:
                logger.error(f"Missing key in batch: {e}")
                raise
            except Exception as e:
                logger.error(f"Error moving batch to device: {e}")
                raise

            # Forward pass with AMP
            with torch.amp.autocast(device_type=str(self.device.type), enabled=self.use_amp):
                try:
                    # Conditional forward pass based on model type
                    if self.model_needs_preset:
                        restored = self.model(lq, crf, preset)
                    else:
                        restored = self.model(lq, crf)

                    loss, loss_dict = self.loss_fn(restored, hq)

                    # Check for NaN/Inf
                    if not torch.isfinite(loss):
                        logger.error(f"Loss is NaN/Inf at step {self.global_step}")
                        logger.error(f"Loss components: {loss_dict}")
                        self._save_checkpoint(
                            epoch, stage_idx, optimizer, scheduler, is_emergency=True
                        )
                        raise RuntimeError(f"Loss became NaN/Inf at step {self.global_step}")

                except Exception as e:
                    logger.error(f"Error in forward pass: {e}", exc_info=True)
                    self._save_checkpoint(
                        epoch, stage_idx, optimizer, scheduler, is_emergency=True
                    )
                    raise

            # Backward pass and optimization
            optimizer.zero_grad(set_to_none=True)

            try:
                if self.scaler:
                    # CUDA AMP with GradScaler
                    self.scaler.scale(loss).backward()
                    
                    # Gradient clipping
                    grad_clip = self.config['training'].get('grad_clip_norm', 0.0)
                    if grad_clip > 0:
                        self.scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip)
                    
                    self.scaler.step(optimizer)
                    self.scaler.update()
                else:
                    # MPS/CPU AMP or no AMP
                    loss.backward()
                    
                    # Gradient clipping
                    grad_clip = self.config['training'].get('grad_clip_norm', 0.0)
                    if grad_clip > 0:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip)
                    
                    optimizer.step()
            except Exception as e:
                logger.error(f"Error in backward pass: {e}", exc_info=True)
                self._save_checkpoint(
                    epoch, stage_idx, optimizer, scheduler, is_emergency=True
                )
                raise

            # Scheduler and EMA updates
            if scheduler:
                scheduler.step()
            if self.ema:
                self.ema.update()

            # Accumulate metrics
            batch_loss = loss.item()
            epoch_total_loss += batch_loss
            for k, v in loss_dict.items():
                epoch_loss_components[k] += v.item()

            # Update progress bar
            lr = optimizer.param_groups[0]['lr']
            pbar.set_postfix(
                Loss=f"{batch_loss:.4f}",
                LR=f"{lr:.2e}",
                refresh=(batch_idx % 10 == 0)
            )

            # W&B step logging
            if WANDB_AVAILABLE and self.config['project']['log_to_wandb'] and wandb.run:
                log_freq = self.config['training'].get('log_every_n_steps', 50)
                if self.global_step % log_freq == 0:
                    wandb_log = {
                        f'train/{k.replace("loss_", "")}': v.item()
                        for k, v in loss_dict.items()
                    }
                    wandb_log['train/learning_rate'] = lr
                    wandb.log(wandb_log, step=self.global_step)

            self.global_step += 1

        # Epoch end logging
        if len(train_loader) > 0:
            avg_loss = epoch_total_loss / len(train_loader)
            avg_components = {
                k: v / len(train_loader)
                for k, v in epoch_loss_components.items()
            }

            logger.info(f"Epoch {epoch+1}/{total_epochs} | Avg Train Loss: {avg_loss:.6f}")

            if WANDB_AVAILABLE and self.config['project']['log_to_wandb'] and wandb.run:
                wandb_log = {
                    f'epoch/avg_{k.replace("loss_", "")}': v
                    for k, v in avg_components.items()
                }
                wandb_log['epoch/avg_total_loss'] = avg_loss
                wandb.log(wandb_log, step=self.global_step)
        else:
            logger.warning("Train loader empty")

    def _validate_and_checkpoint(
        self,
        epoch: int,
        total_epochs: int,
        stage_idx: int,
        val_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[Any]
    ):
        """Run validation and save checkpoints based on config."""
        val_freq = self.config['training'].get('validate_every_n_epochs', 1)
        is_last_epoch = (epoch == total_epochs - 1)
        is_best = False

        # Run validation if frequency met or last epoch
        if val_freq > 0 and ((epoch + 1) % val_freq == 0 or is_last_epoch):
            val_loss = self._validate(epoch, total_epochs, stage_idx, val_loader)

            # Check if new best
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                is_best = True
                logger.info(f"✨ New best validation loss: {val_loss:.6f}")
                self._save_checkpoint(epoch, stage_idx, optimizer, scheduler, is_best=True)

            # Save latest checkpoint
            if not is_best:
                self._save_checkpoint(epoch, stage_idx, optimizer, scheduler)

        # Periodic checkpoint (independent of validation)
        save_freq = self.config['checkpoint'].get('save_every_n_epochs', 0)
        if save_freq > 0 and (epoch + 1) % save_freq == 0:
            validation_ran = (val_freq > 0 and ((epoch + 1) % val_freq == 0 or is_last_epoch))
            if not validation_ran:
                self._save_checkpoint(epoch, stage_idx, optimizer, scheduler)

    @torch.no_grad()
    def _validate(
        self,
        epoch: int,
        total_epochs: int,
        stage_idx: int,
        val_loader: DataLoader
    ) -> float:
        """Run validation and return average loss."""
        self.model.eval()
        if self.ema:
            self.ema.apply_shadow()

        # Accumulators
        total_loss = 0.0
        total_loss_components = defaultdict(float)
        baseline_l1, baseline_l2 = 0.0, 0.0
        restored_l1, restored_l2 = 0.0, 0.0

        pbar = tqdm(
            val_loader,
            desc=f"Validating Epoch {epoch+1}/{total_epochs}",
            leave=False,
            dynamic_ncols=True
        )

        for batch_idx, batch in enumerate(pbar):
            try:
                lq = batch['lq'].to(self.device, non_blocking=True)
                hq = batch['hq'].to(self.device, non_blocking=True)
                crf = batch['crf'].to(self.device, non_blocking=True)
                
                preset = None
                if self.model_needs_preset:
                    preset = batch['preset'].to(self.device, non_blocking=True)

                # Inference
                inference_fn = getattr(self.model, 'inference', self.model.forward)

                with torch.amp.autocast(device_type=str(self.device.type), enabled=self.use_amp):
                    if self.model_needs_preset:
                        restored = inference_fn(lq, crf, preset)
                    else:
                        restored = inference_fn(lq, crf)

                    loss, loss_dict = self.loss_fn(restored, hq)

                # Accumulate metrics
                total_loss += loss.item()
                for k, v in loss_dict.items():
                    total_loss_components[k] += v.item()

                baseline_l1 += F.l1_loss(lq, hq).item()
                baseline_l2 += F.mse_loss(lq, hq).item()
                restored_l1 += F.l1_loss(restored, hq).item()
                restored_l2 += F.mse_loss(restored, hq).item()

            except Exception as e:
                logger.error(f"Validation error batch {batch_idx}: {e}", exc_info=True)
                continue

        # Calculate averages
        num_batches = len(val_loader)
        if num_batches == 0:
            logger.warning("Validation loader empty")
            if self.ema:
                self.ema.restore()
            return float('inf')

        avg_loss = total_loss / num_batches
        avg_components = {k: v / num_batches for k, v in total_loss_components.items()}
        avg_baseline_l1 = baseline_l1 / num_batches
        avg_baseline_l2 = baseline_l2 / num_batches
        avg_restored_l1 = restored_l1 / num_batches
        avg_restored_l2 = restored_l2 / num_batches

        # Calculate improvements
        epsilon = 1e-9
        l1_improvement = ((avg_baseline_l1 - avg_restored_l1) / (avg_baseline_l1 + epsilon)) * 100
        l2_improvement = ((avg_baseline_l2 - avg_restored_l2) / (avg_baseline_l2 + epsilon)) * 100

        # Console logging
        logger.info(f"📊 Validation Epoch {epoch+1}/{total_epochs}:")
        logger.info(f"    Avg Loss: {avg_loss:.6f}")
        logger.info(f"    L1: {avg_restored_l1:.6f} (Baseline: {avg_baseline_l1:.6f}, ↓ {l1_improvement:.2f}%)")
        logger.info(f"    L2: {avg_restored_l2:.6f} (Baseline: {avg_baseline_l2:.6f}, ↓ {l2_improvement:.2f}%)")

        # W&B logging
        if WANDB_AVAILABLE and self.config['project']['log_to_wandb'] and wandb.run:
            wandb_dict = {
                'val/epoch': epoch + 1,
                'val/stage': stage_idx + 1,
                'val/avg_total_loss': avg_loss,
                **{f'val/avg_{k.replace("loss_", "")}': v for k, v in avg_components.items()},
                'val/baseline_l1': avg_baseline_l1,
                'val/baseline_l2': avg_baseline_l2,
                'val/restored_l1': avg_restored_l1,
                'val/restored_l2': avg_restored_l2,
                'val/improvement_l1': l1_improvement,
                'val/improvement_l2': l2_improvement,
            }

            if self.val_samples:
                try:
                    self._log_visual_samples(epoch + 1, stage_idx + 1)
                except Exception as e:
                    logger.warning(f"Failed to log visual samples: {e}")

            wandb.log(wandb_dict, step=self.global_step)

        if self.ema:
            self.ema.restore()

        return avg_loss

    @torch.no_grad()
    def _log_visual_samples(self, epoch_num: int, stage_num: int):
        """Generate and log visual comparison samples to W&B."""
        if not self.val_samples:
            return

        self.model.eval()

        lq = self.val_samples['lq']
        hq = self.val_samples['hq']
        num_samples = lq.size(0)

        inference_fn = getattr(self.model, 'inference', self.model.forward)

        restored_list = []
        for i in range(num_samples):
            lq_i = lq[i:i+1]
            try:
                with torch.amp.autocast(device_type=str(self.device.type), enabled=self.use_amp):
                    crf_i = self.val_samples['crf'][i:i+1]
                    if self.model_needs_preset:
                        preset_i = self.val_samples['preset'][i:i+1]
                        restored_i = inference_fn(lq_i, crf_i, preset_i)
                    else:
                        restored_i = inference_fn(lq_i, crf_i)
                restored_list.append(restored_i)
            except Exception as e:
                logger.warning(f"Error in visual sample {i}: {e}")
                continue

        if not restored_list:
            logger.warning("No visual samples generated")
            return

        restored = torch.cat(restored_list, dim=0)

        # Denormalize for display
        norm_range = tuple(self.config['dataset'].get('norm_range', [-1, 1]))
        if norm_range == (-1, 1):
            denormalize = lambda t: (t.float() + 1.0) / 2.0
            lq_display = denormalize(lq[:restored.size(0)])
            restored_display = denormalize(restored)
            hq_display = denormalize(hq[:restored.size(0)])
        else:
            lq_display = lq[:restored.size(0)].float()
            restored_display = restored.float()
            hq_display = hq[:restored.size(0)].float()

        lq_display.clamp_(0, 1)
        restored_display.clamp_(0, 1)
        hq_display.clamp_(0, 1)

        # Create W&B images
        images_to_log = []
        for i in range(restored_display.size(0)):
            base_name = self.val_samples.get('base_name', [""] * restored_display.size(0))[i]
            crf_str = ""
            try:
                crf_str = f"CRF{int(self.val_samples['crf'][i].item())}"
            except:
                pass
            
            preset_str = ""
            if self.model_needs_preset and 'preset' in self.val_samples:
                try:
                    preset_str = f" P{int(self.val_samples['preset'][i].item())}"
                except:
                    pass

            caption = f"S{stage_num} E{epoch_num} | {base_name} {crf_str}{preset_str} | LQ-Restored-HQ"

            try:
                img_strip = torch.cat(
                    [lq_display[i], restored_display[i], hq_display[i]], dim=2
                )
                images_to_log.append(wandb.Image(img_strip, caption=caption))
            except Exception as e:
                logger.warning(f"Error creating image strip {i}: {e}")
                continue

        if images_to_log:
            wandb.log({"Validation Samples": images_to_log}, step=self.global_step)
        else:
            logger.warning("No visual samples processed")

    # --------------------------------------------------------------------------
    # Checkpointing Logic
    # --------------------------------------------------------------------------

    def _get_checkpoint_path(self, key: Optional[str]) -> Optional[Path]:
        """Resolve checkpoint key to absolute file path."""
        if not key:
            return None

        key_path = Path(key).expanduser()

        # Check if key is an existing file path
        if key_path.is_file():
            return key_path.resolve()

        # Handle symbolic keys
        if self.checkpoint_dir:
            if key.lower() == 'latest':
                return (self.checkpoint_dir / 'latest.pth').resolve()
            if key.lower() == 'best':
                return (self.checkpoint_dir / 'best.pth').resolve()

            # Assume filename within checkpoint directory
            filename = key_path.name
            if not filename.endswith('.pth'):
                filename += '.pth'
            potential_path = (self.checkpoint_dir / filename).resolve()
            if potential_path.is_file():
                return potential_path
            else:
                logger.warning(f"Checkpoint key '{key}' not found: {potential_path}")
                return None
        else:
            logger.error("Checkpoint directory not configured")
            return None

    def _save_checkpoint(
        self,
        epoch: int,
        stage_idx: int,
        optimizer: Optional[torch.optim.Optimizer],
        scheduler: Optional[Any],
        is_best: bool = False,
        is_stage_complete: bool = False,
        is_emergency: bool = False
    ):
        """Save training state to checkpoint."""
        if not self.checkpoint_dir:
            logger.error("Checkpoint directory not set")
            return

        # Prepare state dictionary
        state = {
            'config': self.config,
            'model_state_dict': self.model.state_dict(),
            'stage_idx': stage_idx,
            'epoch': epoch,
            'global_step': self.global_step,
            'best_val_loss': self.best_val_loss,
            'wandb_id': self.wandb_id
        }
        if optimizer:
            state['optimizer_state_dict'] = optimizer.state_dict()
        if scheduler:
            state['scheduler_state_dict'] = scheduler.state_dict()
        if self.ema:
            state['ema_state_dict'] = self.ema.shadow

        # Determine filename
        if is_emergency:
            save_path = self.checkpoint_dir / "emergency_interrupt.pth"
            log_prefix = "🚨 Emergency"
        elif is_best:
            save_path = self.checkpoint_dir / "best.pth"
            log_prefix = "✨ Best"
        elif is_stage_complete:
            save_path = self.checkpoint_dir / f"stage_{stage_idx+1:02d}_complete.pth"
            log_prefix = f"🏁 Stage {stage_idx+1}"
        else:
            save_path = self.checkpoint_dir / "latest.pth"
            log_prefix = "Latest"

        # Save checkpoint
        try:
            save_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(state, save_path)
            if is_best or is_stage_complete or is_emergency:
                logger.info(f"{log_prefix} checkpoint saved: {save_path.name}")
            else:
                logger.debug(f"Saved latest checkpoint")
        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}", exc_info=True)
            return

        # Update latest.pth pointer
        if not is_emergency and save_path != (self.checkpoint_dir / "latest.pth"):
            latest_path = self.checkpoint_dir / "latest.pth"
            try:
                torch.save(state, latest_path)
                logger.debug(f"Updated latest → {save_path.name}")
            except Exception as e:
                logger.error(f"Failed to update latest: {e}")

    def _load_checkpoint(self):
        """Load training state from checkpoint."""
        if not self.resume_from:
            return

        try:
            ckpt_path = self._get_checkpoint_path(self.resume_from)
            if not ckpt_path or not ckpt_path.exists():
                logger.error(f"Checkpoint not found: {self.resume_from}")
                logger.warning("Starting from scratch")
                self.resume_from = None
                return

            logger.info(f"Loading checkpoint: {ckpt_path}")
            ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=False)

            # Restore model
            if 'model_state_dict' not in ckpt:
                raise ValueError("Invalid checkpoint: missing model state")

            missing_keys, unexpected_keys = self.model.load_state_dict(
                ckpt['model_state_dict'], strict=False
            )
            if unexpected_keys:
                logger.warning(f"Unexpected keys: {unexpected_keys}")
            if missing_keys:
                logger.warning(f"Missing keys: {missing_keys}")
            logger.info("Model state restored")

            # Restore training state
            saved_stage = ckpt.get('stage_idx', 0)
            saved_epoch = ckpt.get('epoch', -1)
            self.global_step = ckpt.get('global_step', 0)
            self.best_val_loss = ckpt.get('best_val_loss', float('inf'))

            self.start_stage = saved_stage
            self.start_epoch = saved_epoch + 1

            # Check if stage completed
            try:
                curr_cfg = self.config['curriculum'][self.start_stage]
                epochs_cfg = curr_cfg['epochs']
                total_epochs = sum(epochs_cfg) if isinstance(epochs_cfg, list) else epochs_cfg

                if self.start_epoch >= total_epochs:
                    logger.info(f"Stage {self.start_stage+1} completed")
                    self.start_stage += 1
                    self.start_epoch = 0
                    logger.info(f"Resuming from Stage {self.start_stage+1}")
            except IndexError:
                raise ValueError("Checkpoint stage index out of bounds")

            if self.start_stage >= len(self.config['curriculum']):
                logger.warning("Training already completed")
                sys.exit(0)

            # Restore W&B ID
            ckpt_wandb = ckpt.get('wandb_id')
            if ckpt_wandb and not self.wandb_id:
                self.wandb_id = ckpt_wandb
                logger.info(f"Resuming W&B: {self.wandb_id}")

            # Store optimizer/scheduler state
            self.saved_optimizer_state = ckpt.get('optimizer_state_dict')
            self.saved_scheduler_state = ckpt.get('scheduler_state_dict')

            # Restore EMA
            if self.ema:
                ema_state = ckpt.get('ema_state_dict')
                if ema_state:
                    try:
                        if len(ema_state) == len(self.ema.shadow):
                            self.ema.shadow = ema_state
                            logger.info("EMA state restored")
                        else:
                            logger.warning("EMA key mismatch, resetting")
                            self.ema = EMA(self.model, decay=self.ema.decay)
                    except Exception as e:
                        logger.warning(f"Failed to restore EMA: {e}")

            logger.info(f"Resuming at Stage {self.start_stage+1}, Epoch {self.start_epoch+1}")
            logger.info(f"Global step: {self.global_step}, Best loss: {self.best_val_loss:.6f}")

        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}", exc_info=True)
            logger.warning("Starting from scratch")
            self.start_stage = 0
            self.start_epoch = 0
            self.global_step = 0
            self.best_val_loss = float('inf')
            self.saved_optimizer_state = None
            self.saved_scheduler_state = None
            self.resume_from = None


# ==============================================================================
# SECTION 3: Entry Point
# ==============================================================================

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train Conditional AV1 U-Net Restorer",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '--config', type=str, required=True,
        help="Path to YAML configuration file"
    )
    parser.add_argument(
        '--resume', type=str, default=None,
        help="Resume from checkpoint (latest, best, or path)"
    )
    parser.add_argument(
        '--wandb_id', type=str, default=None,
        help="W&B run ID to resume logging"
    )
    return parser.parse_args()


def main():
    """Main execution with error handling."""
    args = parse_args()
    trainer = None

    try:
        trainer = ConditionalUNetTrainer(
            config_path=args.config,
            resume_from=args.resume,
            wandb_id=args.wandb_id
        )
        trainer.train()

    except KeyboardInterrupt:
        logger.warning("\nTraining interrupted by user")
        if trainer:
            logger.info("Saving emergency checkpoint...")
            try:
                epoch_completed = trainer.start_epoch - 1 if trainer.start_epoch > 0 else -1
                trainer._save_checkpoint(
                    epoch_completed, trainer.start_stage, None, None, is_emergency=True
                )
            except Exception as e:
                logger.error(f"Failed to save emergency checkpoint: {e}")

        if WANDB_AVAILABLE and wandb.run:
            logger.info("Finishing W&B (interrupted)")
            wandb.finish(exit_code=130)
        sys.exit(130)

    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        if trainer and hasattr(trainer, 'model'):
            logger.info("Saving emergency checkpoint...")
            try:
                epoch_completed = trainer.start_epoch - 1 if trainer.start_epoch > 0 else -1
                trainer._save_checkpoint(
                    epoch_completed, trainer.start_stage, None, None, is_emergency=True
                )
            except Exception as save_e:
                logger.error(f"Failed to save emergency checkpoint: {save_e}")

        if WANDB_AVAILABLE and wandb.run:
            logger.info("Finishing W&B (error)")
            wandb.finish(exit_code=1)
        sys.exit(1)


if __name__ == "__main__":
    main()