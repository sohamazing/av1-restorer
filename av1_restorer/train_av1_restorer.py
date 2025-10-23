# train_av1_restorer.py
"""
Universal Training Script for AV1 Artifact Removal Models

==============================================================================
SUPPORTED MODELS
==============================================================================
- AV1UNetRestorer:       Large conditional U-Net (16M params, FiLM conditioning)
- AV1NanoUnetRestorer:   Lightweight U-Net (0.8-2.5M params, specialized per CRF)
- AV1FBCNNRestorer:      FBCNN-inspired lightweight (1.8M params, gated processing)
- AV1NanoResnetRestorer: Single-scale ResNet (1.2-3.3M params, maximum speed)
- AV1NanoMambaRestorer:  Hybrid CNN+SSM (2.0M params, experimental)

==============================================================================
KEY FEATURES
==============================================================================
Automatic model selection based on YAML config
Conditional (crf, preset) vs Non-conditional model handling
Pre-processed (.npy) and on-the-fly (.avif) dataset loading
Full curriculum learning (progressive patch sizes, CRF ranges)
SOTA training: Mixed Precision (AMP), EMA, Warmup + Cosine Annealing
W&B integration for experiment tracking
Robust checkpointing with resume support
Cross-platform (CUDA, MPS, CPU)

==============================================================================
USAGE
==============================================================================
Basic:
    python train_av1_restorer.py --config configs/nano_unet.yaml

Resume:
    python train_av1_restorer.py --config configs/nano_unet.yaml --resume latest

Resume W&B:
    python train_av1_restorer.py --config configs/nano_unet.yaml \
        --resume best --wandb_id abc123def

==============================================================================
Author: Soham Mukherjee
Version: 3.0 (Professional)
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
from torch import autocast
from torch.cuda.amp import GradScaler

# Project imports
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from av1_restorer.models.av1_unet_restorer import create_av1_restorer
from av1_restorer.models.av1_nano_unet_restorer import create_av1_nano_unet_restorer
from av1_restorer.models.av1_fbcnn_restorer import create_av1_fbcnn_restorer
from av1_restorer.models.av1_nano_resnet_restorer import create_av1_nano_resnet_restorer
from av1_restorer.models.av1_nano_mamba_restorer import create_av1_nano_mamba_restorer

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
logger = logging.getLogger("Trainer")


# ==============================================================================
# SECTION 1: Exponential Moving Average (EMA)
# ==============================================================================

class EMA:
    """
    Exponential Moving Average for stable model evaluation.
    
    Maintains shadow parameters that are updated as:
        shadow = decay × shadow + (1 - decay) × model_params
    
    Improves generalization by smoothing weight updates during training.
    Typical decay: 0.9999 (averaging over ~10K updates)
    
    Args:
        model: Neural network model
        decay: EMA decay rate (higher = smoother, slower updates)
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
        """Temporarily replace model parameters with EMA shadow (for validation)."""
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
# SECTION 2: Universal Trainer
# ==============================================================================

class UniversalRestorerTrainer:
    """
    Config-driven trainer for all AV1 restoration models.
    
    Handles:
        • Model creation based on config.model.type
        • Automatic conditional/non-conditional model detection
        • Dataset selection (.npy vs .avif)
        • Curriculum learning with progressive stages
        • Mixed precision training (CUDA/MPS/CPU)
        • EMA for stable evaluation
        • W&B logging and visualization
        • Checkpointing and resume
    
    Args:
        config_path: Path to YAML configuration file
        resume_from: Checkpoint to resume ('latest', 'best', or path)
        wandb_id: W&B run ID for resuming logging
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

        self.train_image_pairs_cache = {}
        self.val_image_pairs_cache = {}
        
        # System setup (device, seeds, AMP)
        self._setup_system()
        
        # Model and loss
        self.model, self.is_conditional_model = self._setup_model()
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
        
        # Device selection
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
        
        # Random seed
        seed = sys_cfg.get('seed', 42)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        logger.info(f"Random seed: {seed}")
        
        # Checkpoint directory
        self.checkpoint_dir = Path(self.config['checkpoint']['dir']).expanduser()
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Checkpoints: {self.checkpoint_dir}")
        
        # Mixed precision (AMP)
        self.use_amp = sys_cfg.get('mixed_precision', False)
        if self.use_amp and self.device.type == 'cuda':
            self.scaler = GradScaler()
            logger.info("AMP: Enabled with GradScaler (CUDA)")
        elif self.use_amp:
            self.scaler = None
            logger.info(f"AMP: Enabled without GradScaler ({self.device.type})")
        else:
            self.scaler = None
            logger.info("  AMP: Disabled")
    
    def _setup_model(self) -> Tuple[nn.Module, bool]:
        """
        Create model based on config.
        
        Returns:
            (model, is_conditional): Model and whether it needs CRF/preset inputs
        """
        model_cfg = self.config['model']
        dset_cfg = self.config['dataset']
        
        model_type = model_cfg.get('type', 'unet')
        size = model_cfg['size']
        crf_range = tuple(dset_cfg['crf_range'])
        preset_range = tuple(dset_cfg.get('preset_range', [0, 8]))
        norm_range = tuple(dset_cfg.get('norm_range', [-1, 1]))
        
        logger.info(f"Creating model: {model_type} (size={size})")
        
        # Model factory mapping
        if model_type == 'unet':
            model = create_av1_restorer(
                size=size, crf_range=crf_range,
                preset_range=preset_range, norm_range=norm_range
            )
            is_conditional = True
        
        elif model_type == 'nano_unet':
            model = create_av1_nano_unet_restorer(
                size=size, crf_min=crf_range[0], crf_max=crf_range[1],
                norm_range=norm_range
            )
            is_conditional = False
        
        elif model_type == 'nano_fbcnn':
            model = create_av1_fbcnn_restorer(
                size=size, crf_min=crf_range[0], crf_max=crf_range[1],
                norm_range=norm_range
            )
            is_conditional = False
        
        elif model_type == 'nano_resnet':
            model = create_av1_nano_resnet_restorer(
                size=size, crf_min=crf_range[0], crf_max=crf_range[1],
                norm_range=norm_range
            )
            is_conditional = False
        
        elif model_type == 'nano_mamba':
            model = create_av1_nano_mamba_restorer(
                size=size, crf_min=crf_range[0], crf_max=crf_range[1],
                norm_range=norm_range
            )
            is_conditional = False
        
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        return model.to(self.device), is_conditional
    
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
                if ckpt_path.exists():
                    ckpt = torch.load(ckpt_path, map_location='cpu')
                    run_id = ckpt.get('wandb_id')
            except Exception as e:
                logger.warning(f"Could not read W&B ID: {e}")
        
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
        logger.info(f"{'AV1 Restorer - Training Configuration':^80}")
        logger.info("=" * 80)
        
        # Project
        proj = self.config['project']
        logger.info(f"Project: {proj['name']}")
        logger.info(f"Experiment: {proj['experiment_name']}")
        
        # Model
        model_cfg = self.config['model']
        params = sum(p.numel() for p in self.model.parameters())
        logger.info(f"Model: {model_cfg['type']} ({model_cfg['size']})")
        logger.info(f"Parameters: {params:,} ({params/1e6:.2f}M)")
        logger.info(f"Conditional: {self.is_conditional_model}")
        
        # Dataset
        dset_cfg = self.config['dataset']
        logger.info(f"CRF Range: {dset_cfg['crf_range']}")
        logger.info(f"Preset Range: {dset_cfg['preset_range']}")
        
        # Curriculum
        logger.info("Curriculum:")
        for i, stage in enumerate(self.config['curriculum']):
            patch = stage['patch_size']
            batch = stage['batch_size']
            epochs = stage['epochs']
            crf = stage.get('crf_range', dset_cfg['crf_range'])
            
            if isinstance(patch, list):
                logger.info(
                    f"  Stage {i+1}: {epochs[0]}+{epochs[1]} epochs, "
                    f"{patch[0]}/{patch[1]}px, batch {batch[0]}/{batch[1]}, "
                    f"CRF {crf}"
                )
            else:
                logger.info(
                    f"  Stage {i+1}: {epochs} epochs, {patch}px, "
                    f"batch {batch}, CRF {crf}"
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

            # --- Caching Logic: Check cache ---
            val_cache_key = f"{data_cfg['val_lq_root']}_{tuple(dset_cfg['crf_range'])}"
            cached_val_image_pairs = self.val_image_pairs_cache.get(val_cache_key)
            # --- End Caching Logic ---
            
            # Create validation dataset
            val_dataset = AV1Dataset(
                lq_root_dir=data_cfg['val_lq_root'],
                hq_root_dir=data_cfg['val_hq_root'],
                hq_ext=data_cfg.get('hq_ext', '.png'),
                patch_size=patch_size,
                crf_range=tuple(dset_cfg['crf_range']),
                preset_range=tuple(dset_cfg['preset_range']),
                augment=False,
                norm_range=tuple(dset_cfg.get('norm_range', [-1, 1])),
                return_metadata=True,
                cached_image_pairs=cached_val_image_pairs
            )

            # --- Caching Logic: Save file list if needed ---
            if cached_val_image_pairs is None and hasattr(val_dataset, 'image_pairs'):
                self.val_image_pairs_cache[val_cache_key] = val_dataset.image_pairs
            # --- End Caching Logic ---
            
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

        # --- Caching Logic: Check cache ---
        # Create unique keys based on path and CRF range
        train_cache_key = f"{data_cfg['train_lq_root']}_{tuple(crf_range)}"
        val_cache_key = f"{data_cfg['val_lq_root']}_{tuple(crf_range)}"

        cached_train_image_pairs = self.train_image_pairs_cache.get(train_cache_key)
        cached_val_image_pairs = self.val_image_pairs_cache.get(val_cache_key)
        # --- End Caching Logic ---
        
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
            'norm_range': tuple(dset_cfg.get('norm_range', [-1, 1]))
        }
        
        if DatasetClass == AV1Dataset:
            common_args['hq_ext'] = data_cfg.get('hq_ext', '.png')
        
        # Create datasets
        train_dset = DatasetClass(
            lq_root_dir=data_cfg['train_lq_root'],
            hq_root_dir=data_cfg['train_hq_root'],
            cached_image_pairs=cached_train_image_pairs,
            augment=True,
            **common_args
        )
        
        val_dset = DatasetClass(
            lq_root_dir=data_cfg['val_lq_root'],
            hq_root_dir=data_cfg['val_hq_root'],
            cached_image_pairs=cached_val_image_pairs,
            augment=False,
            **common_args
        )

        # --- Caching Logic: Save file list if it was just created ---
        # Make sure your AV1Dataset stores the list in an attribute named 'image_pairs'
        if cached_train_image_pairs is None and hasattr(train_dset, 'image_pairs'):
             self.train_image_pairs_cache[train_cache_key] = train_dset.image_pairs
        if cached_val_image_pairs is None and hasattr(val_dset, 'image_pairs'):
             self.val_image_pairs_cache[val_cache_key] = val_dset.image_pairs
        # --- End Caching Logic ---
        
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
            batch_size=batch_size * 2,
            shuffle=False,
            num_workers=max(2, num_workers // 2),
            pin_memory=pin_memory,
            drop_last=False,
            persistent_workers=(num_workers > 0),
            prefetch_factor=2 if num_workers > 0 else None
        )
        
        logger.info(
            f"Dataloaders: {len(train_dset):,} train, "
            f"{len(val_dset):,} val samples"
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
        
        # LR Scheduler
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
            logger.info("  Scheduler: None")
        
        return optimizer, scheduler
    
    # --------------------------------------------------------------------------
    # Training Loop Logic (train, _run_stage, _train_epoch, _validate)
    # --------------------------------------------------------------------------

    def train(self):
        """Main training loop orchestrating curriculum stages."""
        logger.info("\n" + "=" * 80 + "\nSTARTING TRAINING\n" + "=" * 80 + "\n")

        curriculum = self.config['curriculum']
        if not isinstance(curriculum, list):
            raise ValueError("Config 'curriculum' must be a list of stages.")

        # --- Estimate total steps across all stages for scheduler ---
        # This requires creating dummy dataloaders to get accurate lengths
        estimated_total_steps = 0
        stage_loader_lengths = [] # Store lengths to reuse later
        try:
            logger.info("Estimating total steps for scheduler...")
            for stage_cfg in curriculum:
                # Create temporary dataloader just to get length
                temp_train_loader, _ = self._create_dataloaders(stage_cfg)
                current_loader_len = len(temp_train_loader)
                stage_loader_lengths.append(current_loader_len) # Store length

                stage_epochs = stage_cfg['epochs']
                # Correctly sum epochs for progressive stages
                if isinstance(stage_epochs, list):
                    estimated_total_steps += sum(stage_epochs) * current_loader_len
                else: # Simple stage
                    estimated_total_steps += stage_epochs * current_loader_len
                del temp_train_loader # Clean up memory
            logger.info(f"Accurately estimated total steps across all stages: {estimated_total_steps:,}")
        except Exception as e:
            logger.warning(f"Could not accurately estimate total steps due to dataloader error: {e}")
            logger.warning("Using fallback estimation (500 steps/epoch). Scheduler might be inaccurate.")
            # Fallback estimation
            estimated_total_steps = sum(
                sum(s['epochs']) if isinstance(s['epochs'], list) else s['epochs']
                for s in curriculum
            ) * 500
            stage_loader_lengths = [500] * len(curriculum) # Fallback length estimate

        # --- Setup optimizer and scheduler ONCE ---
        optimizer, scheduler = self._setup_optimizer(estimated_total_steps)

        # Restore optimizer/scheduler state if resuming from beginning of training (stage 0, epoch 0)
        # We handle state restoration *within* the stage loop if resuming mid-training
        if self.start_stage == 0 and self.start_epoch == 0 and self.saved_optimizer_state:
            logger.info("Attempting to restore optimizer/scheduler state for initial start...")
            try:
                optimizer.load_state_dict(self.saved_optimizer_state)
                logger.info("Optimizer state restored for initial start.")
                if scheduler and self.saved_scheduler_state:
                    scheduler.load_state_dict(self.saved_scheduler_state)
                    logger.info("Scheduler state restored for initial start.")
            except Exception as e:
                logger.error(f"Failed to restore optimizer/scheduler state at start: {e}. Starting fresh.")
            finally:
                # Clear saved states after attempting load, regardless of success
                self.saved_optimizer_state = None
                self.saved_scheduler_state = None

        # --- Curriculum Loop ---
        for stage_idx in range(self.start_stage, len(curriculum)):
            stage_cfg = curriculum[stage_idx]
            logger.info(f"\n▶ Starting Curriculum Stage {stage_idx + 1}/{len(curriculum)}")

            # --- Run the actual stage (handles epochs and potential sub-stages) ---
            # Pass loader length estimate to avoid recalculating inside _run_stage
            current_stage_loader_len = stage_loader_lengths[stage_idx]
            self._run_stage(stage_idx, stage_cfg, optimizer, scheduler, current_stage_loader_len)

            # --- Save completion checkpoint ---
            # Determine the last epoch number *completed* for this stage (0-based)
            epochs_in_stage = stage_cfg['epochs']
            last_epoch_completed_idx = (sum(epochs_in_stage) - 1) if isinstance(epochs_in_stage, list) else (epochs_in_stage - 1)
            self._save_checkpoint(
                last_epoch_completed_idx, stage_idx, optimizer, scheduler, is_stage_complete=True
            )

            # Reset the epoch counter *for the next stage*
            self.start_epoch = 0

        # --- Training Complete ---
        logger.info("\n" + "=" * 80 + "\n🎉 TRAINING COMPLETE!\n" + f"  Best validation loss: {self.best_val_loss:.6f}\n" + "=" * 80 + "\n")
        # Ensure W&B finishes cleanly
        if WANDB_AVAILABLE and self.config['project']['log_to_wandb'] and wandb.run:
            try:
                wandb.finish()
                logger.info("W&B run finished.")
            except Exception as e:
                logger.error(f"Error finishing W&B run: {e}")

    # --------------------------------------------------------------------------
    # Helper for Epoch Loop
    # --------------------------------------------------------------------------

    def _run_epochs_for_stage(
        self,
        stage_idx: int,
        start_epoch_in_segment: int, # Relative start epoch (0 if starting fresh)
        num_epochs_in_segment: int,  # Epochs to run for this segment
        global_epoch_offset: int,    # Epoch offset from the start of the stage
        total_epochs_in_stage: int,  # Total epochs for the entire stage (for logging)
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[Any],
        sub_stage_idx: int = -1      # Pass sub-stage index for logging
    ):
        """Runs the training and validation loop for a specified number of epochs."""
        # Loop from the relative start epoch within this segment up to the number of epochs for this segment
        for epoch_in_segment in range(start_epoch_in_segment, num_epochs_in_segment):
            # Calculate the overall epoch index within the stage (0-based) for logging/checkpointing
            current_global_epoch_in_stage = global_epoch_offset + epoch_in_segment
            # --- Train Epoch ---
            self._train_epoch(
                epoch=current_global_epoch_in_stage, # Log using global stage epoch
                total_epochs_in_stage=total_epochs_in_stage,
                stage_idx=stage_idx,
                train_loader=train_loader,
                optimizer=optimizer,
                scheduler=scheduler,
                sub_stage_idx=sub_stage_idx # Pass along sub-stage index
            )

            # --- Validation and Checkpointing ---
            # Use the global stage epoch index for validation checks and saving
            self._run_validation_and_checkpointing(
                epoch=current_global_epoch_in_stage, # Pass global stage epoch
                total_epochs_in_stage=total_epochs_in_stage,
                stage_idx=stage_idx,
                val_loader=val_loader,
                optimizer=optimizer,
                scheduler=scheduler
            )

    def _run_stage(
        self,
        stage_idx: int,
        stage_cfg: dict,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[Any],
        loader_len_estimate: int # Passed from train() - currently unused but kept for signature consistency
    ):
        """Runs a single curriculum stage, delegating epoch loops to a helper."""

        # Check if this stage uses progressive sizing
        is_progressive = isinstance(stage_cfg.get('patch_size'), list)

        # --- Restore Optimizer/Scheduler State if resuming mid-stage ---
        # (Keep your existing try/except block for restoring state here)
        if stage_idx == self.start_stage and self.start_epoch > 0 and self.saved_optimizer_state:
            logger.info(f"Attempting to restore optimizer/scheduler state for resuming Stage {stage_idx+1} at epoch {self.start_epoch+1}...")
            try:
                optimizer.load_state_dict(self.saved_optimizer_state)
                logger.info("Optimizer state restored for stage resume.")
                if scheduler and self.saved_scheduler_state:
                    scheduler.load_state_dict(self.saved_scheduler_state)
                    logger.info("Scheduler state restored for stage resume.")
            except Exception as e:
                logger.error(f"Failed to restore optimizer/scheduler state for resume: {e}. Optimizer/scheduler might reset or be incorrect.")
            finally:
                self.saved_optimizer_state = None
                self.saved_scheduler_state = None


        if is_progressive:
            # --- Progressive Stage ---
            patch_sizes = stage_cfg['patch_size']
            batch_sizes = stage_cfg['batch_size']
            epochs_per_sub = stage_cfg['epochs']
            total_epochs_in_stage = sum(epochs_per_sub)

            if not (len(patch_sizes) == len(batch_sizes) == len(epochs_per_sub)):
                raise ValueError(f"Progressive stage {stage_idx+1} has mismatched list lengths.")

            logger.info(f"Running progressive stage with {len(patch_sizes)} sub-stages.")
            cumulative_epochs_offset = 0 # Tracks start epoch offset for each sub-stage

            for sub_idx in range(len(patch_sizes)):
                sub_patch = patch_sizes[sub_idx]
                sub_batch = batch_sizes[sub_idx]
                sub_epochs = epochs_per_sub[sub_idx]

                # Check if resuming skips this sub-stage entirely
                # self.start_epoch is the global 0-based index within the stage
                if self.start_epoch >= cumulative_epochs_offset + sub_epochs:
                    logger.info(f"  Skipping Sub-stage {stage_idx+1}.{sub_idx+1} (completed based on resume epoch {self.start_epoch}).")
                    cumulative_epochs_offset += sub_epochs
                    continue

                logger.info(
                    f"  → Starting Sub-stage {stage_idx+1}.{sub_idx+1}: "
                    f"{sub_epochs} epochs @ {sub_patch}px, batch {sub_batch}"
                )

                # Create dataloaders for this sub-stage
                sub_stage_dl_cfg = stage_cfg.copy()
                sub_stage_dl_cfg['patch_size'] = sub_patch
                sub_stage_dl_cfg['batch_size'] = sub_batch
                train_loader, val_loader = self._create_dataloaders(sub_stage_dl_cfg)

                # Determine relative start epoch *within this sub-stage* (0-based)
                start_epoch_in_sub = max(0, self.start_epoch - cumulative_epochs_offset)

                # *** Call the helper function for the epoch loop ***
                self._run_epochs_for_stage(
                    stage_idx=stage_idx,
                    start_epoch_in_segment=start_epoch_in_sub, # Relative start
                    num_epochs_in_segment=sub_epochs,          # Epochs for this sub-stage
                    global_epoch_offset=cumulative_epochs_offset, # Offset from stage start
                    total_epochs_in_stage=total_epochs_in_stage,  # Total for logging
                    train_loader=train_loader,
                    val_loader=val_loader,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    sub_stage_idx=sub_idx # Pass sub-stage index
                )

                cumulative_epochs_offset += sub_epochs
                # Reset start_epoch relative counter only if running sequentially
                # If resuming, self.start_epoch already holds the correct global stage epoch index
                # This ensures the skip logic works correctly on resume.
                self.start_epoch = 0 # This reset is safe for sequential runs

        else:
            # --- Simple Stage ---
            epochs = stage_cfg['epochs']
            logger.info(f"Running simple stage: {epochs} epochs.")
            train_loader, val_loader = self._create_dataloaders(stage_cfg)

            # *** Call the helper function for the epoch loop ***
            # self.start_epoch is already the correct starting epoch (0-based) within this stage
            self._run_epochs_for_stage(
                stage_idx=stage_idx,
                start_epoch_in_segment=self.start_epoch, # Use directly
                num_epochs_in_segment=epochs,            # Epochs for the whole stage
                global_epoch_offset=0,                   # No offset needed
                total_epochs_in_stage=epochs,            # Total is just stage epochs
                train_loader=train_loader,
                val_loader=val_loader,
                optimizer=optimizer,
                scheduler=scheduler,
                sub_stage_idx=-1 # Indicate not a sub-stage
            )
            # Resetting self.start_epoch for the *next* stage happens in train()


    def _run_validation_and_checkpointing(self, epoch, total_epochs_in_stage, stage_idx, val_loader, optimizer, scheduler):
        """Helper function to run validation and save checkpoints based on config."""
        # --- Validation ---
        val_freq = self.config['training'].get('validate_every_n_epochs', 1)
        # Ensure validation runs on the very last epoch of the stage
        # Check epoch index (0-based) against total number of epochs - 1
        is_last_epoch = (epoch == total_epochs_in_stage - 1)
        is_best_model = False

        # Run validation if frequency met OR if it's the last epoch of the stage
        if val_freq > 0 and ((epoch + 1) % val_freq == 0 or is_last_epoch):
            val_loss = self._validate(epoch, total_epochs_in_stage, stage_idx, val_loader)

            # Check if this validation loss is the new best
            # Use >= to handle potential floating point inaccuracies if loss doesn't change
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                is_best_model = True
                logger.info(f"✨ New best validation loss: {val_loss:.6f} at epoch {epoch+1}")
                # Save as the best model
                self._save_checkpoint(
                    epoch, stage_idx, optimizer, scheduler, is_best=is_best_model
                )

            # Save latest checkpoint after every validation run (including potentially the best one)
            # Avoids double-saving if it was already saved as 'best'
            if not is_best_model: # Only save latest if it wasn't just saved as best
                self._save_checkpoint(epoch, stage_idx, optimizer, scheduler)

        # --- Periodic Checkpoint (independent of validation frequency) ---
        save_freq = self.config['checkpoint'].get('save_every_n_epochs', 0) # Default 0 = disabled
        if save_freq > 0 and (epoch + 1) % save_freq == 0:
            # Avoid saving 'latest' again if validation just ran and saved it
            validation_ran_this_epoch = (val_freq > 0 and ((epoch + 1) % val_freq == 0 or is_last_epoch))
            if not validation_ran_this_epoch:
                self._save_checkpoint(epoch, stage_idx, optimizer, scheduler)


    def _train_epoch(
        self,
        epoch: int,               # Current epoch index (0-based, global within stage)
        total_epochs_in_stage: int, # Total epochs for this stage (for logging)
        stage_idx: int,           # Current stage index (0-based)
        train_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[Any],
        sub_stage_idx: int = -1   # Sub-stage index if progressive (-1 otherwise)
    ):
        """Train the model for one epoch."""
        self.model.train() # Set model to training mode

        # --- Setup Progress Bar ---
        stage_desc = f"Stage {stage_idx+1}"
        if sub_stage_idx != -1:
            stage_desc += f".{sub_stage_idx+1}" # e.g., Stage 1.1
        # Display 1-based epoch number in progress bar
        pbar_desc = f"{stage_desc} | Epoch {epoch+1}/{total_epochs_in_stage}"
        pbar = tqdm(train_loader, desc=pbar_desc, leave=True, dynamic_ncols=True)

        # --- Accumulators for Epoch Average Loss ---
        epoch_total_loss = 0.0
        # Use a default dictionary for easier accumulation
        epoch_loss_components = defaultdict(float)

        # --- Batch Loop ---
        num_batches = len(train_loader)
        for batch_idx, batch in enumerate(pbar):
            # Move data to device
            try:
                lq = batch['lq'].to(self.device, non_blocking=True)
                hq = batch['hq'].to(self.device, non_blocking=True)
                if self.is_conditional_model:
                    crf = batch['crf'].to(self.device, non_blocking=True)
                    preset = batch['preset'].to(self.device, non_blocking=True)
            except KeyError as e:
                logger.error(f"Missing key in batch data: {e}. Check dataset output.")
                raise
            except Exception as e:
                logger.error(f"Error moving batch data to device: {e}")
                raise # Re-raise error

            # --- Forward Pass ---
            # Use autocast context manager for AMP
            with autocast(device_type=str(self.device.type), enabled=self.use_amp):
                try:
                    # Handle conditional vs. non-conditional models
                    if self.is_conditional_model:
                        restored = self.model(lq, crf, preset)
                    else:
                        restored = self.model(lq)

                    # Calculate loss
                    loss, loss_dict = self.loss_fn(restored, hq)

                    # --- CRITICAL: Check for NaN/Inf Loss ---
                    if not torch.isfinite(loss):
                        logger.error(f"Loss is NaN or Inf at Global Step {self.global_step}, Epoch {epoch+1}, Batch {batch_idx+1}.")
                        logger.error(f"    Loss components: { {k: v.item() for k, v in loss_dict.items()} }")
                        # Attempt emergency save before raising error
                        self._save_checkpoint(epoch, stage_idx, optimizer, scheduler, is_emergency=True)
                        raise RuntimeError(f"Loss became NaN/Inf at step {self.global_step}. Training stopped.")

                except Exception as e:
                    logger.error(f"Error during forward pass or loss calculation at step {self.global_step}: {e}", exc_info=True)
                    # Attempt emergency save
                    self._save_checkpoint(epoch, stage_idx, optimizer, scheduler, is_emergency=True)
                    raise # Re-raise error

            # --- Backward Pass & Optimization ---
            optimizer.zero_grad(set_to_none=True) # More efficient zeroing

            try:
                if self.scaler: # CUDA AMP Path
                    self.scaler.scale(loss).backward()
                    # Gradient Clipping (before optimizer step, after unscaling)
                    grad_clip_norm = self.config['training'].get('grad_clip_norm', 0.0) # Default to 0.0 (disabled)
                    if grad_clip_norm > 0:
                        # Unscale first before clipping
                        self.scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip_norm)
                    # Optimizer Step (advances model parameters)
                    self.scaler.step(optimizer)
                    # Update GradScaler scale for next iteration
                    self.scaler.update()
                else: # MPS/CPU AMP or No AMP Path
                    loss.backward()
                    # Gradient Clipping
                    grad_clip_norm = self.config['training'].get('grad_clip_norm', 0.0)
                    if grad_clip_norm > 0:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip_norm)
                    # Optimizer Step
                    optimizer.step()
            except Exception as e:
                logger.error(f"Error during backward pass or optimizer step at {self.global_step}: {e}", exc_info=True)
                # Attempt emergency save
                self._save_checkpoint(epoch, stage_idx, optimizer, scheduler, is_emergency=True)
                raise # Re-raise error


            # --- Scheduler & EMA Update (after optimizer step) ---
            # Scheduler steps per *iteration* (batch)
            if scheduler:
                scheduler.step()
            if self.ema:
                self.ema.update()

            # --- Metrics Accumulation ---
            batch_loss = loss.item() # Get Python number
            epoch_total_loss += batch_loss
            for k, v in loss_dict.items():
                epoch_loss_components[k] += v.item() # Accumulate Python numbers


            # --- Update Progress Bar ---
            lr = optimizer.param_groups[0]['lr'] # Get current learning rate
            # Display current batch loss and current LR in TQDM postfix
            pbar.set_postfix(
                Loss=f"{batch_loss:.4f}", # Current batch loss
                LR=f"{lr:.2e}",
                refresh= (batch_idx % 10 == 0) # Refresh less frequently
            )

            # --- W&B Logging (per step/iteration) ---
            log_freq = self.config['training'].get('log_every_n_steps', 50) # Configurable log frequency
            if WANDB_AVAILABLE and self.config['project']['log_to_wandb'] and wandb.run:
                if self.global_step % log_freq == 0:
                    # Log individual loss components and LR per step
                    wandb_step_log = {f'train/{k.replace("loss_", "")}': v.item() for k, v in loss_dict.items()}
                    wandb_step_log['train/learning_rate'] = lr
                    wandb.log(wandb_step_log, step=self.global_step)

            self.global_step += 1 # Increment global step counter *after* logging

        # --- Epoch End Logging ---
        if num_batches > 0:
            avg_epoch_loss = epoch_total_loss / num_batches
            avg_epoch_components = {k: v / num_batches for k,v in epoch_loss_components.items()}

            # Log average loss for the completed epoch to console and to W&B
            logger.info(f"Epoch {epoch+1}/{total_epochs_in_stage} | Avg Train Loss: {avg_epoch_loss:.6f}")
            if WANDB_AVAILABLE and self.config['project']['log_to_wandb'] and wandb.run:
                wandb_epoch_log = {
                    f'epoch/avg_{k.replace("loss_", "")}': v
                    for k, v in avg_epoch_components.items()
                }
                wandb_epoch_log['epoch/avg_total_loss'] = avg_epoch_loss
                # Log these average values against the epoch number
                wandb.log(wandb_epoch_log, step=self.global_step)   # epoch+1
        else:
            logger.warning(f"Epoch {epoch+1}/{total_epochs_in_stage} | Train loader was empty. No training occurred.")


    @torch.no_grad()
    def _validate(
        self,
        epoch: int,               # Current epoch index (0-based, global within stage)
        total_epochs_in_stage: int, # Total epochs for this stage (for logging)
        stage_idx: int,           # Current stage index (0-based)
        val_loader: DataLoader
    ) -> float:
        """Run validation, log metrics, and return average total loss."""
        self.model.eval() # Set model to evaluation mode
        if self.ema:
            self.ema.apply_shadow() # Use EMA weights for validation

        # --- Accumulators ---
        total_loss = 0.0
        # Use defaultdict for easier accumulation of component losses
        total_loss_components = defaultdict(float)
        baseline_l1_sum = 0.0
        baseline_l2_sum = 0.0 # Add L2/MSE
        restored_l1_sum = 0.0
        restored_l2_sum = 0.0 # Add L2/MSE

        pbar = tqdm(val_loader, desc=f"Validating Epoch {epoch+1}/{total_epochs_in_stage}", leave=False, dynamic_ncols=True)

        # --- Validation Loop ---
        for batch_idx, batch in enumerate(pbar):
            try:
                # Move data
                lq = batch['lq'].to(self.device, non_blocking=True)
                hq = batch['hq'].to(self.device, non_blocking=True)

                # --- Inference ---
                # Use model's dedicated .inference() method if it exists, otherwise use .forward()
                # Check for existence robustly
                inference_fn = getattr(self.model, 'inference', self.model.forward)

                # Use autocast for potential AMP benefits during validation inference (optional but can be faster)
                with autocast(device_type=str(self.device.type), enabled=self.use_amp):
                    if self.is_conditional_model:
                        crf = batch['crf'].to(self.device, non_blocking=True)
                        preset = batch['preset'].to(self.device, non_blocking=True)
                        restored = inference_fn(lq, crf, preset)
                    else:
                        restored = inference_fn(lq)

                    # --- Loss Calculation ---
                    # Calculate loss even during validation to get a quantitative metric
                    loss, loss_dict = self.loss_fn(restored, hq)

                # --- Accumulate Metrics ---
                # Use .item() to get Python numbers and free up GPU memory
                total_loss += loss.item()
                for key, value in loss_dict.items():
                    total_loss_components[key] += value.item()

                # --- Baseline Metrics (LQ vs HQ) ---
                baseline_l1_sum += F.l1_loss(lq, hq).item()
                baseline_l2_sum += F.mse_loss(lq, hq).item() # Calculate MSE

                # --- Restored Metrics (Restored vs HQ) ---
                restored_l1_sum += F.l1_loss(restored, hq).item()
                restored_l2_sum += F.mse_loss(restored, hq).item() # Calculate MSE

            except Exception as e:
                logger.error(f"Error during validation batch {batch_idx+1}: {e}", exc_info=True)
                # Decide whether to skip batch or raise error
                continue # Skip this batch and continue validation

        # --- Calculate Averages ---
        num_batches = len(val_loader)
        if num_batches == 0:
            logger.warning("Validation loader was empty. Cannot calculate validation metrics.")
            if self.ema: self.ema.restore() # Ensure model weights are restored
            return float('inf') # Indicate validation failure

        avg_loss = total_loss / num_batches
        avg_components = {k: v / num_batches for k, v in total_loss_components.items()}
        avg_baseline_l1 = baseline_l1_sum / num_batches
        avg_baseline_l2 = baseline_l2_sum / num_batches # Average baseline L2
        avg_restored_l1 = restored_l1_sum / num_batches
        avg_restored_l2 = restored_l2_sum / num_batches # Average restored L2

        # --- Calculate Improvement Percentages ---
        # Add small epsilon to denominator to prevent division by zero if baseline loss is 0
        epsilon = 1e-9
        l1_improvement = ((avg_baseline_l1 - avg_restored_l1) / (avg_baseline_l1 + epsilon)) * 100
        l2_improvement = ((avg_baseline_l2 - avg_restored_l2) / (avg_baseline_l2 + epsilon)) * 100 # Calculate L2 improvement

        # --- Console Logging ---
        # Use epoch+1 for 1-based display
        logger.info(f"📊 Validation Epoch {epoch+1}/{total_epochs_in_stage} Results:")
        logger.info(f"    Avg Total Loss: {avg_loss:.6f}")
        # Log individual components if needed: logger.info(f"    Loss Components: { {k: f'{v:.4f}' for k, v in avg_components.items()} }")
        logger.info(f"    Metrics vs Baseline (LQ):")
        logger.info(f"      L1: {avg_restored_l1:.6f} (Baseline: {avg_baseline_l1:.6f}, ↓ {l1_improvement:.2f}%)")
        logger.info(f"      L2 (MSE): {avg_restored_l2:.6f} (Baseline: {avg_baseline_l2:.6f}, ↓ {l2_improvement:.2f}%)") # Log L2

        # --- W&B Logging ---
        if WANDB_AVAILABLE and self.config['project']['log_to_wandb'] and wandb.run:
            # Use nested structure for clarity in W&B UI
            wandb_dict = {
                'val/epoch': epoch + 1, # 1-based epoch number
                'val/stage': stage_idx + 1, # 1-based stage number
                'val/loss/total': avg_loss,
                # Log individual loss components under 'val/loss/' group
                **{f'val/loss/{k.replace("loss_", "")}': v for k, v in avg_components.items()},
                # Baseline Metrics Group
                'val/baseline/l1': avg_baseline_l1,
                'val/baseline/l2': avg_baseline_l2,
                # Restored Metrics Group
                'val/restored/l1': avg_restored_l1,
                'val/restored/l2': avg_restored_l2,
                # Improvement Metrics Group
                'val/improvement/l1_pct': l1_improvement,
                'val/improvement/l2_pct': l2_improvement,
            }

            # Log visual samples (only if validation samples were successfully loaded)
            if self.val_samples:
                try:
                    # Pass 1-based epoch and stage index for caption clarity
                    self._log_visual_samples(epoch + 1, stage_idx + 1)
                except Exception as e:
                    logger.warning(f"Failed to log visual samples to W&B: {e}", exc_info=False) # Keep it concise

            # Log all validation metrics against the current global step
            wandb.log(wandb_dict, step=self.global_step)

        if self.ema:
            self.ema.restore() # Restore original model weights AFTER validation and logging

        return avg_loss # Return the primary validation metric (total loss)

    @torch.no_grad()
    def _log_visual_samples(self, epoch_num: int, stage_num: int): # Use 1-based indices
        """Generate and log visual comparison samples to W&B."""
        if not self.val_samples:
            logger.debug("No validation samples available to log.")
            return # No samples loaded

        self.model.eval() # Ensure model is in eval mode
        # EMA weights should already be applied if self.ema exists (called from _validate)

        lq = self.val_samples['lq']
        hq = self.val_samples['hq']
        num_samples_to_log = lq.size(0) # Number of samples we actually loaded

        # --- Inference ---
        # Use model's dedicated .inference() method if it exists, otherwise use .forward()
        inference_fn = getattr(self.model, 'inference', self.model.forward)

        restored_list = []
        # Process samples individually - safer if inference fn doesn't support batching well
        for i in range(num_samples_to_log):
            lq_i = lq[i:i+1] # Keep batch dimension [1, C, H, W]
            try:
                # Use autocast here as well if using AMP during validation
                with autocast(device_type=str(self.device.type), enabled=self.use_amp):
                    if self.is_conditional_model:
                        crf_i = self.val_samples['crf'][i:i+1]
                        preset_i = self.val_samples['preset'][i:i+1]
                        restored_i = inference_fn(lq_i, crf_i, preset_i)
                    else:
                        restored_i = inference_fn(lq_i)
                restored_list.append(restored_i)
            except Exception as e:
                logger.warning(f"Error during inference for visual sample {i}: {e}. Skipping sample.")
                # Add a placeholder or skip? Skipping is safer.
                continue # Skip to next sample if inference fails

        if not restored_list:
            logger.warning("No visual samples could be generated due to inference errors.")
            return

        restored = torch.cat(restored_list, dim=0) # Reassemble batch


        # --- Denormalize for Display ---
        norm_range = tuple(self.config['dataset'].get('norm_range', [-1, 1]))
        if norm_range == (-1, 1):
            # Denormalize function: x' = (x + 1) / 2
            denormalize = lambda t: (t.float() + 1.0) / 2.0 # Ensure float before op
            lq_display = denormalize(lq[:restored.size(0)]) # Match batch size if some failed
            restored_display = denormalize(restored)
            hq_display = denormalize(hq[:restored.size(0)])
        else: # Assumes [0, 1] range already
            lq_display = lq[:restored.size(0)].float()
            restored_display = restored.float()
            hq_display = hq[:restored.size(0)].float()

        # Clamp results to [0, 1] range AFTER denormalization
        lq_display.clamp_(0, 1)
        restored_display.clamp_(0, 1)
        hq_display.clamp_(0, 1)

        # --- Create list of wandb.Image objects ---
        images_to_log = []
        actual_samples_logged = restored_display.size(0) # How many samples we actually have after potential errors
        for i in range(actual_samples_logged):
            # Get metadata for caption if available in self.val_samples
            # Use .get with default values for robustness
            base_name = self.val_samples.get('base_name', [""] * actual_samples_logged)[i]
            crf_val_str = ""
            if 'crf' in self.val_samples:
                try: crf_val_str = f"CRF{int(self.val_samples['crf'][i].item())}"
                except: pass # Ignore errors if CRF is not scalar
            preset_val_str = ""
            if 'preset' in self.val_samples:
                try: preset_val_str = f" P{int(self.val_samples['preset'][i].item())}"
                except: pass

            caption = f"S{stage_num} E{epoch_num} | {base_name} {crf_val_str}{preset_val_str} | LQ - Restored - HQ"

            # Create the horizontal strip: LQ | Restored | HQ
            # Ensure dimensions match before cat: [C, H, W]
            try:
                img_strip = torch.cat([lq_display[i], restored_display[i], hq_display[i]], dim=2) # Cat width-wise
                images_to_log.append(wandb.Image(img_strip, caption=caption))
            except Exception as e:
                logger.warning(f"Error creating image strip for sample {i}: {e}. Skipping.")
                continue

        # --- Log to W&B ---
        if images_to_log:
            # Log list of images under a single key
            wandb.log({"Validation Samples": images_to_log}, step=self.global_step)
        else:
            logger.warning("No visual samples were successfully processed for logging.")

        # Note: EMA restore happens in _validate after this function returns

    # --------------------------------------------------------------------------
    # Checkpointing Logic (_get_checkpoint_path, _save_checkpoint, _load_checkpoint)
    # --------------------------------------------------------------------------

    def _get_checkpoint_path(self, key: Optional[str]) -> Optional[Path]:
        """Resolve a checkpoint key ('latest', 'best', path) to an absolute file path."""
        if not key: return None # Handle case where resume_from is None or empty

        key_path = Path(key).expanduser()

        # 1. Check if the key itself is an existing absolute or relative file path
        if key_path.is_file():
            return key_path.resolve()

        # 2. Handle symbolic keys relative to the checkpoint directory
        if self.checkpoint_dir: # Ensure checkpoint_dir is set
            if key.lower() == 'latest':
                return (self.checkpoint_dir / 'latest.pth').resolve()
            if key.lower() == 'best':
                return (self.checkpoint_dir / 'best.pth').resolve()

            # 3. If not 'latest' or 'best', assume it's a filename within the checkpoint directory
            # Add .pth extension if missing for convenience
            filename = key_path.name
            if not filename.endswith('.pth'):
                filename += '.pth'
            potential_path = (self.checkpoint_dir / filename).resolve()
            # Check if this resolved path actually exists
            if potential_path.is_file():
                return potential_path
            else:
                # If the file doesn't exist in the checkpoint dir, return None or raise error
                logger.warning(f"Checkpoint key '{key}' resolved to non-existent file: {potential_path}")
                return None # Or raise FileNotFoundError based on desired strictness
        else:
            logger.error("Checkpoint directory not configured. Cannot resolve checkpoint key.")
            return None


    def _save_checkpoint(
        self,
        epoch: int,       # Epoch number completed (0-based)
        stage_idx: int,   # Stage index completed (0-based)
        optimizer: Optional[torch.optim.Optimizer], # Make optional for emergency
        scheduler: Optional[Any], # Make optional for emergency
        is_best: bool = False,
        is_stage_complete: bool = False,
        is_emergency: bool = False # Flag for emergency saves
    ):
        """Save the current training state to a checkpoint file(s)."""
        if not self.checkpoint_dir:
            logger.error("Checkpoint directory not set. Cannot save checkpoint.")
            return

        # --- Prepare State Dictionary ---
        state = {
            'config': self.config, # Store config for reproducibility and loading checks
            'model_state_dict': self.model.state_dict(),
            'stage_idx': stage_idx,
            'epoch': epoch, # Record the epoch number *just completed* (0-based)
            'global_step': self.global_step,
            'best_val_loss': self.best_val_loss,
            'wandb_id': self.wandb_id
        }
        # Add optional components if available
        if optimizer: state['optimizer_state_dict'] = optimizer.state_dict()
        if scheduler: state['scheduler_state_dict'] = scheduler.state_dict()
        if self.ema: state['ema_state_dict'] = self.ema.shadow

        # --- Determine Filename and Log Prefix ---
        save_path = None
        log_prefix = ""
        if is_emergency:
            save_path = self.checkpoint_dir / "emergency_interrupt.pth"
            log_prefix = "🚨 Emergency"
        elif is_best:
            save_path = self.checkpoint_dir / "best.pth"
            log_prefix = "✨ Best"
        elif is_stage_complete:
            save_path = self.checkpoint_dir / f"stage_{stage_idx+1:02d}_complete.pth"
            log_prefix = f"🏁 Stage {stage_idx+1} Complete"
        else: # Standard 'latest' or periodic save triggered by validation/save_freq
            save_path = self.checkpoint_dir / "latest.pth"
            log_prefix = "Periodic/Latest" # Distinguish periodic saves if needed

        # --- Save Primary Checkpoint ---
        if save_path:
            try:
                # Ensure parent directory exists (should be handled in __init__, but good practice)
                save_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(state, save_path)
                # Log completion for significant saves
                if is_best or is_stage_complete or is_emergency:
                    logger.info(f"{log_prefix} checkpoint saved: {save_path.name}")
                # Use debug for frequent 'latest' saves to reduce noise
                elif log_prefix == "Periodic/Latest":
                    logger.debug(f"Saved latest checkpoint: {save_path.name}")

            except Exception as e:
                logger.error(f"Failed to save {log_prefix.lower()} checkpoint to {save_path.name}: {e}", exc_info=True)
                return # Stop if primary save fails

        # --- Update 'latest.pth' Symlink (or copy) ---
        # Always update latest.pth to point to the most recently saved valid state
        # This includes best, stage complete, and periodic saves. Exclude emergency for stability?
        if not is_emergency and save_path != (self.checkpoint_dir / "latest.pth"):
            latest_path = self.checkpoint_dir / "latest.pth"
            try:
                # Option 1: Create a symlink (more efficient, POSIX only)
                # if latest_path.exists(): latest_path.unlink()
                # latest_path.symlink_to(save_path.name)

                # Option 2: Copy the file (safer, cross-platform)
                torch.save(state, latest_path) # Just re-save the state to latest.pth
                logger.debug(f"Updated latest checkpoint pointer to {save_path.name}")

            except Exception as e:
                logger.error(f"Failed to update latest checkpoint link/copy: {e}", exc_info=True)


    def _load_checkpoint(self):
        """Load training state from a checkpoint file specified by self.resume_from."""
        if not self.resume_from: return # No resume requested

        ckpt_path = None # Initialize path variable
        try:
            ckpt_path = self._get_checkpoint_path(self.resume_from)
            if not ckpt_path or not ckpt_path.exists():
                logger.error(f"Resume checkpoint '{self.resume_from}' resolved to '{ckpt_path}' but it does not exist.")
                logger.warning("Starting training from scratch.")
                self.resume_from = None # Clear resume flag to avoid retry loops
                return # Exit loading process

            logger.info(f"Loading checkpoint: {ckpt_path}")
            # Load checkpoint onto the correct device directly to potentially save memory
            ckpt = torch.load(ckpt_path, map_location=self.device)

            # --- (Optional but Recommended) Config Compatibility Check ---
            # if 'config' in ckpt and ckpt['config']['model'] != self.config['model']:
            #     logger.warning("Checkpoint model config differs from current config. Loading may fail.")
            # Add more checks as needed (dataset, etc.)

            # --- Restore Model ---
            if 'model_state_dict' not in ckpt:
                logger.error("Checkpoint missing 'model_state_dict'. Cannot restore model state.")
                raise ValueError("Invalid checkpoint: missing model state.")

            try:
                # Load state dict using strict=False first to report issues
                missing_keys, unexpected_keys = self.model.load_state_dict(ckpt['model_state_dict'], strict=False)
                if unexpected_keys:
                    # Log unexpected keys, might indicate architecture mismatch
                    logger.warning(f"Unexpected keys found in checkpoint model state (ignored): {unexpected_keys}")
                if missing_keys:
                    # Log missing keys, indicates current model has layers not in checkpoint
                    logger.warning(f"Missing keys in model state_dict (initialized randomly): {missing_keys}")
                logger.info("Model state restored (strict=False used for compatibility check)")
            except Exception as e:
                logger.error(f"Error loading model state_dict even with strict=False: {e}", exc_info=True)
                raise # Re-raise critical error


            # --- Restore Training State ---
            # Get saved progress (use defaults if keys are missing)
            saved_stage_idx = ckpt.get('stage_idx', 0)
            saved_epoch_completed = ckpt.get('epoch', -1) # Epoch index *completed* (0-based)
            self.global_step = ckpt.get('global_step', 0)
            self.best_val_loss = ckpt.get('best_val_loss', float('inf'))

            # --- Determine Resume Point (Stage and Epoch) ---
            self.start_stage = saved_stage_idx
            self.start_epoch = saved_epoch_completed + 1 # Start from the epoch *after* the saved one

            try:
                current_stage_config = self.config['curriculum'][self.start_stage]
                epochs_config = current_stage_config['epochs']
                total_epochs_in_saved_stage = sum(epochs_config) if isinstance(epochs_config, list) else epochs_config

                # Check if the start_epoch is beyond the number of epochs in the saved stage
                if self.start_epoch >= total_epochs_in_saved_stage:
                    logger.info(f"Checkpoint indicates Stage {self.start_stage + 1} completed ({self.start_epoch}/{total_epochs_in_saved_stage} epochs done).")
                    self.start_stage += 1 # Advance to the next stage index
                    self.start_epoch = 0 # Start from epoch 0 of the new stage
                    logger.info(f"Resuming from the beginning of Stage {self.start_stage + 1}.")
                # else: # Resuming within the saved stage is handled by start_epoch value

            except IndexError:
                logger.error(f"Checkpoint stage index {saved_stage_idx} is out of bounds for current curriculum ({len(self.config['curriculum'])} stages). Cannot reliably resume.")
                # Option: Default to starting fresh or raise error
                raise ValueError("Checkpoint stage index incompatible with current curriculum configuration.")
            except Exception as e:
                logger.error(f"Error processing stage/epoch for resume: {e}. Attempting basic resume from saved values.", exc_info=True)
                # If error, proceed with self.start_stage and self.start_epoch as loaded


            # Check if training is already finished based on stages
            if self.start_stage >= len(self.config['curriculum']):
                logger.warning(f"Checkpoint indicates training was already completed after Stage {self.start_stage}. No further stages to run. Exiting.")
                sys.exit(0) # Successful exit, training is done


            # --- Restore W&B ID ---
            ckpt_wandb_id = ckpt.get('wandb_id')
            if ckpt_wandb_id and not self.wandb_id: # Only use if not provided via CLI
                self.wandb_id = ckpt_wandb_id
                logger.info(f"Resuming W&B run ID from checkpoint: {self.wandb_id}")
            elif ckpt_wandb_id and self.wandb_id and ckpt_wandb_id != self.wandb_id:
                logger.warning(f"W&B ID mismatch! CLI provided '{self.wandb_id}', checkpoint has '{ckpt_wandb_id}'. Using CLI-provided ID.")


            # --- Store Optimizer/Scheduler States for later restoration ---
            # These will be loaded by _run_stage or train() before the relevant stage starts
            self.saved_optimizer_state = ckpt.get('optimizer_state_dict')
            self.saved_scheduler_state = ckpt.get('scheduler_state_dict')
            if self.saved_optimizer_state: logger.info("Optimizer state found (will load before stage training).")
            if self.saved_scheduler_state: logger.info("Scheduler state found (will load before stage training).")


            # --- Restore EMA ---
            if self.ema: # Only attempt if EMA is enabled for the current run
                ema_state = ckpt.get('ema_state_dict')
                if ema_state:
                    try:
                        # Basic check: Ensure number of keys match roughly
                        if len(ema_state) == len(self.ema.shadow):
                            self.ema.shadow = ema_state
                            logger.info("EMA shadow state restored")
                        else:
                            logger.warning(f"EMA state keys mismatch ({len(ema_state)} vs {len(self.ema.shadow)}). Resetting EMA.")
                            self.ema = EMA(self.model, decay=self.ema.decay) # Reinitialize
                    except Exception as e:
                        logger.warning(f"Failed to restore EMA state: {e}. EMA will start fresh.")
                else:
                    logger.warning("EMA enabled but no EMA state found in checkpoint. EMA will start fresh.")


            # --- Final Log ---
            logger.info(f"Checkpoint loaded successfully.")
            logger.info(f"  --> Resuming at: Stage {self.start_stage + 1}, Epoch {self.start_epoch + 1}") # Display 1-based indices
            logger.info(f"  Global Step: {self.global_step}, Best Val Loss: {self.best_val_loss:.6f}")

        except FileNotFoundError:
            # This case is handled at the start of the try block now
            pass # Error already logged
        except Exception as e:
            # Catch any other unexpected errors during loading
            logger.error(f"Failed to load checkpoint '{ckpt_path if ckpt_path else self.resume_from}' due to unexpected error: {e}", exc_info=True)
            logger.warning("Starting training from scratch due to critical checkpoint load error.")
            # Reset state variables to ensure a clean start
            self.start_stage = 0
            self.start_epoch = 0
            self.global_step = 0
            self.best_val_loss = float('inf')
            self.saved_optimizer_state = None
            self.saved_scheduler_state = None
            self.resume_from = None # Prevent re-attempting load


# ==============================================================================
# SECTION 3: Entry Point & Argument Parsing
# ==============================================================================
def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Universal AV1 Restorer Trainer",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter # Show defaults in help
    )
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help="Path to the YAML configuration file."
    )
    parser.add_argument(
        '--resume',
        type=str,
        default=None,
        help="Resume training from a checkpoint. Options: 'latest', 'best', path/to/checkpoint.pth, or specific filename in checkpoint dir."
    )
    parser.add_argument(
        '--wandb_id',
        type=str,
        default=None,
        help="Weights & Biases run ID to resume logging to an existing run."
    )
    return parser.parse_args()

def main():
    """Main execution function with robust error handling and emergency save."""
    args = parse_args()
    trainer = None # Initialize trainer variable for access in exception blocks

    try:
        # --- Initialize Trainer ---
        # This step includes loading the config, setting up system, model, loss,
        # logging, and potentially loading checkpoint state into trainer attributes.
        trainer = UniversalRestorerTrainer(
            config_path=args.config,
            resume_from=args.resume,
            wandb_id=args.wandb_id
        )

        # --- Start Training ---
        # The train() method orchestrates the curriculum stages and epochs.
        trainer.train()

    except KeyboardInterrupt:
        logger.warning("\nTraining interrupted by user (KeyboardInterrupt).")
        if trainer: # Check if trainer was successfully initialized
            logger.info("Attempting to save emergency checkpoint...")
            try:
                # In case of interrupt, save the current state before exiting.
                # Determine the last fully completed epoch. If interrupted mid-epoch,
                # save the state corresponding to the beginning of the current epoch.
                # start_epoch is the *next* epoch to run, so subtract 1 for completed epoch.
                epoch_completed = trainer.start_epoch - 1 if trainer.start_epoch > 0 else -1 # -1 if interrupted before epoch 0 starts

                # Save essential state. Avoid saving optimizer/scheduler as they might be
                # in an inconsistent state during an interrupt. Resuming will re-initialize them.
                trainer._save_checkpoint(
                    epoch=epoch_completed, # Save index of last completed epoch
                    stage_idx=trainer.start_stage,
                    optimizer=None, # Exclude potentially inconsistent optimizer state
                    scheduler=None, # Exclude potentially inconsistent scheduler state
                    is_emergency=True
                )
            except Exception as e:
                # Log error during emergency save but proceed to exit
                logger.error(f"Failed to save emergency checkpoint during interrupt: {e}", exc_info=True)
        else:
            # Trainer initialization itself might have failed
            logger.warning("Trainer object not available, cannot save emergency checkpoint.")

        # Attempt to finish W&B run cleanly, marking as interrupted
        if WANDB_AVAILABLE and wandb.run:
            logger.info("Finishing W&B run (interrupted)...")
            wandb.finish(exit_code=130) # Standard exit code for SIGINT (Ctrl+C)
        sys.exit(130) # Exit with interrupt code

    except Exception as e:
        # Catch any other unexpected error during trainer initialization or training
        logger.error(f"Training failed due to unexpected error: {e}", exc_info=True)

        # Attempt emergency save if trainer exists
        if trainer and hasattr(trainer, 'model'): # Check if model exists
            logger.info("Attempting to save emergency checkpoint due to error...")
            try:
                epoch_completed = trainer.start_epoch - 1 if trainer.start_epoch > 0 else -1
                trainer._save_checkpoint(
                    epoch=epoch_completed,
                    stage_idx=trainer.start_stage,
                    optimizer=None, # Exclude potentially inconsistent state
                    scheduler=None,
                    is_emergency=True
                )
            except Exception as save_e:
                logger.error(f"Failed to save emergency checkpoint during error handling: {save_e}", exc_info=True)

        # Attempt to finish W&B run cleanly, marking as failed
        if WANDB_AVAILABLE and wandb.run:
            logger.info("Finishing W&B run (error)...")
            wandb.finish(exit_code=1) # Indicate general error exit
        sys.exit(1) # Exit with non-zero code to indicate failure

if __name__ == "__main__":
    # Ensures the script runs only when executed directly
    main()