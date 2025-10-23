# train_av1_nano_restorer.py
"""
Training Script for Non-Conditional AV1 Restorers

Supports lightweight models specialized for specific CRF ranges.
Models: NanoUNet, FBCNN, NanoResNet, NanoMamba

==============================================================================
STRATEGY
==============================================================================
Train separate models per CRF bucket for optimal quality:
    - Low:      CRF 23-33 (light artifacts)
    - Medium:   CRF 34-43 (moderate artifacts)
    - High:     CRF 44-53 (heavy artifacts)
    - Extreme:  CRF 54-63 (severe artifacts)

At inference, select model based on input CRF value.

==============================================================================
FEATURES
==============================================================================
- Automatic model selection based on config.model.type
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
    python train_av1_nano_restorer.py --config configs/nano_unet_crf54-63.yaml

Resume:
    python train_av1_nano_restorer.py --config configs/nano_unet_crf54-63.yaml --resume latest

Resume W&B:
    python train_av1_nano_restorer.py --config configs/nano_unet_crf54-63.yaml \
        --resume best --wandb_id abc123def

==============================================================================
Author: Soham Mukherjee
Version: 2.0 (Non-Conditional, CRF-Bucket-Specialized)
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
torch.autograd.set_detect_anomaly(True)
from tqdm import tqdm

# Cross-platform AMP support
import torch.amp

# Project imports
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from av1_restorer.models.av1_nano_unet_restorer import create_av1_nano_unet_restorer
from av1_restorer.models.av1_nano_fbcnn_restorer import create_av1_nano_fbcnn_restorer
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
logger = logging.getLogger("NonConditionalTrainer")


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
# SECTION 2: Non-Conditional Trainer
# ==============================================================================

class NonConditionalTrainer:
    """
    Config-driven trainer for Non-Conditional AV1 Restorers.
    
    These models are specialized per CRF range and do NOT use
    CRF/Preset conditioning during inference (forward pass).
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

        # Dataset file list caching
        self.train_image_pairs_cache = {}
        self.val_image_pairs_cache = {}
        
        # System setup
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
            self.scaler = torch.amp.GradScaler('cuda', enabled=self.use_amp)
            logger.info("AMP: Enabled with GradScaler (CUDA)")
        elif self.use_amp:
            self.scaler = None
            logger.info(f"AMP: Enabled without GradScaler ({self.device.type})")
        else:
            self.scaler = None
            logger.info("AMP: Disabled")
    
    def _setup_model(self) -> nn.Module:
        """
        Create non-conditional model based on config.
        
        Supported types:
            - nano_unet: Lightweight U-Net (0.8-2.5M params)
            - nano_fbcnn: FBCNN-inspired (1.8M params)
            - nano_resnet: Single-scale ResNet (1.2-3.3M params)
            - nano_mamba: Hybrid CNN+SSM (2.0M params)
        """
        model_cfg = self.config['model']
        dset_cfg = self.config['dataset']
        
        model_type = model_cfg.get('type')
        valid_types = ['nano_unet', 'nano_fbcnn', 'nano_resnet', 'nano_mamba']
        
        if model_type not in valid_types:
            raise ValueError(
                f"This script only supports {valid_types}, "
                f"found '{model_type}' in config"
            )
        
        size = model_cfg['size']
        crf_range = tuple(dset_cfg['crf_range'])
        norm_range = tuple(dset_cfg.get('norm_range', [-1, 1]))
        
        logger.info(f"Creating model: {model_type} (size={size})")
        logger.info(f"CRF Specialization: {crf_range}")
        
        # Model factory
        if model_type == 'nano_unet':
            model = create_av1_nano_unet_restorer(
                size=size,
                crf_min=crf_range[0],
                crf_max=crf_range[1],
                norm_range=norm_range
            )
        elif model_type == 'nano_fbcnn':
            model = create_av1_fbcnn_restorer(
                size=size,
                crf_min=crf_range[0],
                crf_max=crf_range[1],
                norm_range=norm_range
            )
        elif model_type == 'nano_resnet':
            model = create_av1_nano_resnet_restorer(
                size=size,
                crf_min=crf_range[0],
                crf_max=crf_range[1],
                norm_range=norm_range
            )
        elif model_type == 'nano_mamba':
            model = create_av1_nano_mamba_restorer(
                size=size,
                crf_min=crf_range[0],
                crf_max=crf_range[1],
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
        logger.info(f"{'AV1 Non-Conditional - Training Configuration':^80}")
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
        logger.info(f"Conditioning: None (CRF-specialized)")
        
        # Dataset info
        dset_cfg = self.config['dataset']
        logger.info(f"CRF Range (Dataset): {dset_cfg['crf_range']}")
        logger.info(f"Preset Range (Dataset): {dset_cfg['preset_range']}")
        
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

            val_cache_key = f"{data_cfg['val_lq_root']}_{tuple(dset_cfg['crf_range'])}"
            cached_val_image_pairs = self.val_image_pairs_cache.get(val_cache_key)
            
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

            if cached_val_image_pairs is None and hasattr(val_dataset, 'image_pairs'):
                self.val_image_pairs_cache[val_cache_key] = val_dataset.image_pairs
            
            loader = DataLoader(val_dataset, batch_size=num_samples, shuffle=False)
            batch = next(iter(loader))
            
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
    # Dataset & Optimizer Setup (Similar to Conditional Trainer)
    # --------------------------------------------------------------------------
    
    def _create_dataloaders(self, stage_config: dict) -> Tuple[DataLoader, DataLoader]:
        """Create train and validation dataloaders."""
        data_cfg = self.config['data']
        dset_cfg = self.config['dataset']
        sys_cfg = self.config['system']
        
        patch_size = stage_config['patch_size']
        batch_size = stage_config['batch_size']
        crf_range = stage_config.get('crf_range', dset_cfg['crf_range'])

        train_cache_key = f"{data_cfg['train_lq_root']}_{tuple(crf_range)}"
        val_cache_key = f"{data_cfg['val_lq_root']}_{tuple(crf_range)}"
        cached_train = self.train_image_pairs_cache.get(train_cache_key)
        cached_val = self.val_image_pairs_cache.get(val_cache_key)
        
        if isinstance(patch_size, list):
            patch_size = patch_size[0]
            batch_size = batch_size[0]
        
        lq_ext = data_cfg.get('lq_ext', '.avif').lower()
        DatasetClass = AV1DatasetFast if lq_ext == '.npy' else AV1Dataset
        
        common_args = {
            'patch_size': patch_size,
            'crf_range': tuple(crf_range),
            'preset_range': tuple(dset_cfg['preset_range']),
            'norm_range': tuple(dset_cfg.get('norm_range', [-1, 1])),
            'hq_ext': data_cfg.get('hq_ext', '.png')
        }
        
        train_dset = DatasetClass(
            lq_root_dir=data_cfg['train_lq_root'],
            hq_root_dir=data_cfg['train_hq_root'],
            cached_image_pairs=cached_train,
            augment=True,
            **common_args
        )
        
        val_dset = DatasetClass(
            lq_root_dir=data_cfg['val_lq_root'],
            hq_root_dir=data_cfg['val_hq_root'],
            cached_image_pairs=cached_val,
            augment=False,
            **common_args
        )

        if cached_train is None and hasattr(train_dset, 'image_pairs'):
            self.train_image_pairs_cache[train_cache_key] = train_dset.image_pairs
        if cached_val is None and hasattr(val_dset, 'image_pairs'):
            self.val_image_pairs_cache[val_cache_key] = val_dset.image_pairs
        
        num_workers = sys_cfg.get('num_workers', 8)
        pin_memory = (self.device.type == 'cuda')
        
        train_loader = DataLoader(
            train_dset, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=pin_memory, drop_last=True,
            persistent_workers=(num_workers > 0), prefetch_factor=4 if num_workers > 0 else None
        )
        
        val_loader = DataLoader(
            val_dset, batch_size=batch_size * 2, shuffle=False,
            num_workers=max(2, num_workers // 2), pin_memory=pin_memory, drop_last=False,
            persistent_workers=(num_workers > 0), prefetch_factor=2 if num_workers > 0 else None
        )
        
        logger.info(f"Dataloaders: {len(train_dset):,} train, {len(val_dset):,} val")
        return train_loader, val_loader
    
    def _setup_optimizer(self, total_steps: int) -> Tuple[torch.optim.Optimizer, Optional[Any]]:
        """Create optimizer and LR scheduler with warmup."""
        opt_cfg = self.config['optimizer']
        sched_cfg = self.config['scheduler']
        
        lr = opt_cfg['lr']
        if opt_cfg['type'].lower() == 'adamw':
            optimizer = AdamW(
                self.model.parameters(), lr=lr,
                betas=tuple(opt_cfg.get('betas', [0.9, 0.999])),
                weight_decay=opt_cfg.get('weight_decay', 1e-4)
            )
        else:
            optimizer = Adam(
                self.model.parameters(), lr=lr,
                betas=tuple(opt_cfg.get('betas', [0.9, 0.999]))
            )
        
        logger.info(f"Optimizer: {opt_cfg['type'].upper()}, lr={lr}")
        
        scheduler_type = sched_cfg.get('type', 'cosine').lower()
        warmup_steps = sched_cfg.get('warmup_steps', 500)
        
        if scheduler_type == 'cosine':
            main_scheduler = CosineAnnealingLR(
                optimizer, T_max=max(1, total_steps - warmup_steps),
                eta_min=sched_cfg.get('min_lr', 1e-6)
            )
            warmup_scheduler = LinearLR(
                optimizer, start_factor=1e-5, end_factor=1.0, total_iters=warmup_steps
            )
            scheduler = SequentialLR(
                optimizer, schedulers=[warmup_scheduler, main_scheduler],
                milestones=[warmup_steps]
            )
            logger.info(f"Scheduler: Cosine with {warmup_steps}-step warmup")
        else:
            scheduler = None
            logger.info("Scheduler: None")
        
        return optimizer, scheduler
    
    # --------------------------------------------------------------------------
    # Training Loop (Same structure as Conditional, simpler forward pass)
    # --------------------------------------------------------------------------
    
    def train(self):
        """Main training loop."""
        logger.info("\n" + "=" * 80)
        logger.info("STARTING TRAINING")
        logger.info("=" * 80 + "\n")

        curriculum = self.config['curriculum']
        
        # Estimate total steps
        estimated_total_steps = 0
        stage_loader_lengths = []
        try:
            for stage_cfg in curriculum:
                temp_loader, _ = self._create_dataloaders(stage_cfg)
                stage_loader_lengths.append(len(temp_loader))
                stage_epochs = stage_cfg['epochs']
                estimated_total_steps += (sum(stage_epochs) if isinstance(stage_epochs, list) else stage_epochs) * len(temp_loader)
                del temp_loader
            logger.info(f"Estimated total steps: {estimated_total_steps:,}")
        except:
            estimated_total_steps = sum(sum(s['epochs']) if isinstance(s['epochs'], list) else s['epochs'] for s in curriculum) * 500
            stage_loader_lengths = [500] * len(curriculum)

        optimizer, scheduler = self._setup_optimizer(estimated_total_steps)

        # Restore state if resuming
        if self.start_stage == 0 and self.start_epoch == 0 and self.saved_optimizer_state:
            try:
                optimizer.load_state_dict(self.saved_optimizer_state)
                if scheduler and self.saved_scheduler_state:
                    scheduler.load_state_dict(self.saved_scheduler_state)
                logger.info("Optimizer/scheduler state restored")
            except Exception as e:
                logger.error(f"Failed to restore state: {e}")
            finally:
                self.saved_optimizer_state = None
                self.saved_scheduler_state = None

        # Curriculum loop
        for stage_idx in range(self.start_stage, len(curriculum)):
            stage_cfg = curriculum[stage_idx]
            logger.info(f"\n▶ Starting Stage {stage_idx + 1}/{len(curriculum)}")

            self._run_stage(stage_idx, stage_cfg, optimizer, scheduler, stage_loader_lengths[stage_idx])

            epochs_in_stage = stage_cfg['epochs']
            last_epoch = (sum(epochs_in_stage) - 1) if isinstance(epochs_in_stage, list) else (epochs_in_stage - 1)
            self._save_checkpoint(last_epoch, stage_idx, optimizer, scheduler, is_stage_complete=True)
            self.start_epoch = 0

        logger.info("\n" + "=" * 80)
        logger.info("🎉 TRAINING COMPLETE!")
        logger.info(f"Best validation loss: {self.best_val_loss:.6f}")
        logger.info("=" * 80 + "\n")
        
        if WANDB_AVAILABLE and wandb.run:
            try:
                wandb.finish()
            except:
                pass

    def _run_stage(self, stage_idx, stage_cfg, optimizer, scheduler, loader_len):
        """Run single curriculum stage."""
        is_progressive = isinstance(stage_cfg.get('patch_size'), list)

        if stage_idx == self.start_stage and self.start_epoch > 0 and self.saved_optimizer_state:
            try:
                optimizer.load_state_dict(self.saved_optimizer_state)
                if scheduler and self.saved_scheduler_state:
                    scheduler.load_state_dict(self.saved_scheduler_state)
            except:
                pass
            finally:
                self.saved_optimizer_state = None
                self.saved_scheduler_state = None

        if is_progressive:
            patch_sizes = stage_cfg['patch_size']
            batch_sizes = stage_cfg['batch_size']
            epochs_per_sub = stage_cfg['epochs']
            total_epochs = sum(epochs_per_sub)

            cumulative_epochs = 0
            for sub_idx in range(len(patch_sizes)):
                sub_patch, sub_batch, sub_epochs = patch_sizes[sub_idx], batch_sizes[sub_idx], epochs_per_sub[sub_idx]

                if self.start_epoch >= cumulative_epochs + sub_epochs:
                    cumulative_epochs += sub_epochs
                    continue

                sub_cfg = stage_cfg.copy()
                sub_cfg['patch_size'] = sub_patch
                sub_cfg['batch_size'] = sub_batch

                train_loader, val_loader = self._create_dataloaders(sub_cfg)
                start_epoch_in_sub = max(0, self.start_epoch - cumulative_epochs)

                self._run_epochs(
                    stage_idx, start_epoch_in_sub, sub_epochs,
                    cumulative_epochs, total_epochs,
                    train_loader, val_loader, optimizer, scheduler, sub_idx
                )

                cumulative_epochs += sub_epochs
                self.start_epoch = 0 # Reset for sequential runs

        else:
            # Simple stage
            epochs = stage_cfg['epochs']
            logger.info(f"Simple stage: {epochs} epochs")
            train_loader, val_loader = self._create_dataloaders(stage_cfg)

            self._run_epochs(
                stage_idx, self.start_epoch, epochs,
                0, epochs,
                train_loader, val_loader, optimizer, scheduler, -1
            )
            # self.start_epoch is reset in train() loop after stage completes

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

        stage_desc = f"Stage {stage_idx+1}"
        if sub_stage_idx != -1: stage_desc += f".{sub_stage_idx+1}"
        pbar_desc = f"{stage_desc} | Epoch {epoch+1}/{total_epochs}"
        pbar = tqdm(train_loader, desc=pbar_desc, leave=True, dynamic_ncols=True)

        epoch_total_loss = 0.0
        epoch_loss_components = defaultdict(float)

        for batch_idx, batch in enumerate(pbar):
            try:
                lq = batch['lq'].to(self.device, non_blocking=True)
                hq = batch['hq'].to(self.device, non_blocking=True)
            except KeyError as e:
                logger.error(f"Missing key: {e}")
                raise
            except Exception as e:
                logger.error(f"Error moving batch: {e}")
                raise

            with torch.amp.autocast(device_type=str(self.device.type), enabled=self.use_amp):
                try:
                    restored = self.model(lq) # Non-conditional forward pass
                    loss, loss_dict = self.loss_fn(restored, hq)
                    if not torch.isfinite(loss):
                        logger.error(f"Loss NaN/Inf @ Step {self.global_step}")
                        self._save_checkpoint(epoch, stage_idx, optimizer, scheduler, is_emergency=True)
                        raise RuntimeError("Loss NaN/Inf")
                except Exception as e:
                    logger.error(f"Forward/Loss error: {e}")
                    self._save_checkpoint(epoch, stage_idx, optimizer, scheduler, is_emergency=True)
                    raise

            optimizer.zero_grad(set_to_none=True)
            try:
                if self.scaler: # CUDA AMP
                    self.scaler.scale(loss).backward()
                    grad_clip = self.config['training'].get('grad_clip_norm', 0)
                    if grad_clip > 0:
                        self.scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip)
                    self.scaler.step(optimizer)
                    self.scaler.update()
                else: # MPS/CPU AMP or No AMP
                    loss.backward()
                    grad_clip = self.config['training'].get('grad_clip_norm', 0)
                    if grad_clip > 0:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip)
                    optimizer.step()
            except Exception as e:
                logger.error(f"Backward/Opt error: {e}")
                self._save_checkpoint(epoch, stage_idx, optimizer, scheduler, is_emergency=True)
                raise

            if scheduler:
                scheduler.step()
            if self.ema:
                self.ema.update()

            batch_loss = loss.item()
            epoch_total_loss += batch_loss
            for k, v in loss_dict.items():
                epoch_loss_components[k] += v.item()

            lr = optimizer.param_groups[0]['lr']
            pbar.set_postfix(Loss=f"{batch_loss:.4f}", LR=f"{lr:.2e}", refresh=(batch_idx % 10 == 0))

            if WANDB_AVAILABLE and wandb.run:
                log_freq = self.config['training'].get('log_every_n_steps', 50)
                if self.global_step % log_freq == 0:
                    wandb_log = {f'train/{k.replace("loss_", "")}': v.item() for k, v in loss_dict.items()}
                    wandb_log['train/learning_rate'] = lr
                    wandb.log(wandb_log, step=self.global_step)

            self.global_step += 1

        if len(train_loader) > 0:
            avg_loss = epoch_total_loss / len(train_loader)
            avg_components = {k: v / len(train_loader) for k, v in epoch_loss_components.items()}
            logger.info(f"Epoch {epoch+1}/{total_epochs} | Avg Train Loss: {avg_loss:.6f}")
            if WANDB_AVAILABLE and wandb.run:
                wandb_log = {f'epoch/avg_{k.replace("loss_", "")}': v for k, v in avg_components.items()}
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

        if val_freq > 0 and ((epoch + 1) % val_freq == 0 or is_last_epoch):
            val_loss = self._validate(epoch, total_epochs, stage_idx, val_loader)
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                is_best = True
                logger.info(f"✨ New best validation loss: {val_loss:.6f}")
                self._save_checkpoint(epoch, stage_idx, optimizer, scheduler, is_best=True)
            if not is_best:
                self._save_checkpoint(epoch, stage_idx, optimizer, scheduler)

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
        if self.ema: self.ema.apply_shadow()

        total_loss, total_loss_components = 0.0, defaultdict(float)
        base_l1, base_l2, rest_l1, rest_l2 = 0.0, 0.0, 0.0, 0.0
        pbar = tqdm(val_loader, desc=f"Validating Epoch {epoch+1}/{total_epochs}", leave=False, dynamic_ncols=True)

        for batch_idx, batch in enumerate(pbar):
            try:
                lq = batch['lq'].to(self.device, non_blocking=True)
                hq = batch['hq'].to(self.device, non_blocking=True)

                inference_fn = getattr(self.model, 'inference', self.model.forward)
                with torch.amp.autocast(device_type=str(self.device.type), enabled=self.use_amp):
                    restored = inference_fn(lq) # Non-conditional inference
                    loss, loss_dict = self.loss_fn(restored, hq)

                total_loss += loss.item()
                for k, v in loss_dict.items(): total_loss_components[k] += v.item()
                base_l1 += F.l1_loss(lq, hq).item()
                base_l2 += F.mse_loss(lq, hq).item()
                rest_l1 += F.l1_loss(restored, hq).item()
                rest_l2 += F.mse_loss(restored, hq).item()
            except Exception as e:
                logger.error(f"Validation error: {e}")
                continue

        if len(val_loader) == 0: 
            logger.warning("Val loader empty.")
            return float('inf')

        avg_loss = total_loss / len(val_loader)
        avg_comps = {k: v / len(val_loader) for k, v in total_loss_components.items()}
        avg_base_l1, avg_base_l2 = base_l1 / len(val_loader), base_l2 / len(val_loader)
        avg_rest_l1, avg_rest_l2 = rest_l1 / len(val_loader), rest_l2 / len(val_loader)
        epsilon = 1e-9
        l1_imp = ((avg_base_l1 - avg_rest_l1) / (avg_base_l1 + epsilon)) * 100
        l2_imp = ((avg_base_l2 - avg_rest_l2) / (avg_base_l2 + epsilon)) * 100

        logger.info(f"📊 Validation Epoch {epoch+1}/{total_epochs}:")
        logger.info(f"    Avg Loss: {avg_loss:.6f}")
        logger.info(f"    L1: {avg_rest_l1:.6f} (Baseline: {avg_base_l1:.6f}, ↓ {l1_imp:.2f}%)")
        logger.info(f"    L2: {avg_rest_l2:.6f} (Baseline: {avg_base_l2:.6f}, ↓ {l2_imp:.2f}%)")

        if WANDB_AVAILABLE and wandb.run:
            wandb_dict = {'val/epoch': epoch + 1, 'val/stage': stage_idx + 1, 'val/avg_total_loss': avg_loss, **{f'val/avg_{k.replace("loss_", "")}': v for k, v in avg_comps.items()}, 'val/baseline_l1': avg_base_l1, 'val/baseline_l2': avg_base_l2, 'val/restored_l1': avg_rest_l1, 'val/restored_l2': avg_rest_l2, 'val/improvement_l1': l1_imp, 'val/improvement_l2': l2_imp}
            if self.val_samples:
                try:
                    self._log_visual_samples(epoch + 1, stage_idx + 1) 
                except Exception as e:
                    logger.warning(f"Failed log samples: {e}")
            wandb.log(wandb_dict, step=self.global_step)

        if self.ema:
            self.ema.restore()
        return avg_loss

    @torch.no_grad()
    def _log_visual_samples(self, epoch_num: int, stage_num: int):
        """Generate and log visual samples to W&B."""
        if not self.val_samples: return
        self.model.eval()
        lq, hq = self.val_samples['lq'], self.val_samples['hq']

        inference_fn = getattr(self.model, 'inference', self.model.forward)
        restored_list = []
        for i in range(lq.size(0)):
            lq_i = lq[i:i+1]
            try:
                with torch.amp.autocast(device_type=str(self.device.type), enabled=self.use_amp):
                    restored_i = inference_fn(lq_i) # Non-conditional inference
                restored_list.append(restored_i)
            except Exception as e:
                logger.warning(f"Error infer sample {i}: {e}")
                continue
        if not restored_list:
            logger.warning("No samples generated.")
            return
        restored = torch.cat(restored_list, dim=0)

        # Denormalize
        norm_range = tuple(self.config['dataset'].get('norm_range', [-1, 1]))
        if norm_range == (-1, 1):
            denorm = lambda t: (t.float() + 1.0) / 2.0
            lq_d, r_d, hq_d = denorm(lq[:restored.size(0)]), denorm(restored), denorm(hq[:restored.size(0)])
        else: lq_d, r_d, hq_d = lq[:restored.size(0)].float(), restored.float(), hq[:restored.size(0)].float()
        lq_d.clamp_(0, 1)
        r_d.clamp_(0, 1)
        hq_d.clamp_(0, 1)

        images_to_log = []
        for i in range(r_d.size(0)):
            base_name = self.val_samples.get('base_name', [""] * r_d.size(0))[i]
            # Simpler caption without CRF/Preset
            caption = f"S{stage_num} E{epoch_num} | {base_name} | LQ-Restored-HQ"
            try:
                img_strip = torch.cat([lq_d[i], r_d[i], hq_d[i]], dim=2)
                images_to_log.append(wandb.Image(img_strip, caption=caption))
            except Exception as e:
                logger.warning(f"Error image strip {i}: {e}")
                continue
        if images_to_log:
            wandb.log({"Validation Samples": images_to_log}, step=self.global_step)
        else:
            logger.warning("No visual samples processed.")

    # --------------------------------------------------------------------------
    # Checkpointing Logic (Identical to Conditional Trainer)
    # --------------------------------------------------------------------------
    def _get_checkpoint_path(self, key: Optional[str]) -> Optional[Path]:
        """Resolve checkpoint path based on key."""
        if not key:
            return None

        key_path = Path(key).expanduser()
        if key_path.is_file():
            return key_path.resolve()

        if not self.checkpoint_dir:
            logger.error("Ckpt dir not configured.")
            return None

        key_lower = key.lower()
        if key_lower == 'latest':
            return (self.checkpoint_dir / 'latest.pth').resolve()
        if key_lower == 'best':
            return (self.checkpoint_dir / 'best.pth').resolve()

        filename = key_path.name
        if not filename.endswith('.pth'):
            filename += '.pth'

        potential_path = (self.checkpoint_dir / filename).resolve()
        if potential_path.is_file():
            return potential_path

        logger.warning(f"Ckpt key '{key}' not file: {potential_path}")
        return None

    def _save_checkpoint(
        self,
        epoch,
        stage_idx,
        optimizer,
        scheduler,
        is_best=False,
        is_stage_complete=False,
        is_emergency=False
    ):
        """Save model, optimizer, and scheduler state."""
        if not self.checkpoint_dir:
            logger.error("Ckpt dir not set.")
            return

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

        save_path, log_prefix = None, ""
        if is_emergency:
            save_path = self.checkpoint_dir / "emergency_interrupt.pth"
            log_prefix = "🚨 Emergency"
        elif is_best:
            save_path = self.checkpoint_dir / "best.pth"
            log_prefix = "✨ Best"
        elif is_stage_complete:
            save_path = self.checkpoint_dir / f"stage_{stage_idx + 1:02d}_complete.pth"
            log_prefix = f"🏁 Stage {stage_idx + 1} Complete"
        else:
            save_path = self.checkpoint_dir / "latest.pth"
            log_prefix = "Periodic/Latest"

        try:
            save_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(state, save_path)

            if is_best or is_stage_complete or is_emergency:
                logger.info(f"{log_prefix} ckpt saved: {save_path.name}")
            elif log_prefix == "Periodic/Latest":
                logger.debug(f"Saved latest ckpt: {save_path.name}")

        except Exception as e:
            logger.error(f"Failed save {log_prefix} ckpt: {e}")
            return

        if not is_emergency and save_path != (self.checkpoint_dir / "latest.pth"):
            latest_path = self.checkpoint_dir / "latest.pth"
            try:
                torch.save(state, latest_path)
                logger.debug(f"Updated latest ckpt to {save_path.name}")
            except Exception as e:
                logger.error(f"Failed update latest ckpt: {e}")

    def _load_checkpoint(self):
        """Load model, optimizer, scheduler, and EMA state from checkpoint."""
        if not self.resume_from:
            return

        ckpt_path = None
        try:
            ckpt_path = self._get_checkpoint_path(self.resume_from)
            if not ckpt_path or not ckpt_path.exists():
                logger.error(f"Resume ckpt not found: {ckpt_path}")
                self.resume_from = None
                return

            logger.info(f"Loading ckpt: {ckpt_path}")
            ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=False)

            if 'model_state_dict' not in ckpt:
                logger.error("Ckpt missing model_state_dict")
                raise ValueError("Invalid ckpt")

            try:
                missing, unexpected = self.model.load_state_dict(ckpt['model_state_dict'], strict=False)
                if unexpected:
                    logger.warning(f"Unexpected keys: {unexpected}")
                if missing:
                    logger.warning(f"Missing keys: {missing}")
                logger.info("Model state restored")
            except Exception as e:
                logger.error(f"Error loading model state: {e}")
                raise

            # Restore progress tracking
            self.global_step = ckpt.get('global_step', 0)
            self.best_val_loss = ckpt.get('best_val_loss', float('inf'))
            self.start_stage = ckpt.get('stage_idx', 0)
            self.start_epoch = ckpt.get('epoch', -1) + 1

            # Handle stage completion and progression
            try:
                curr_cfg = self.config['curriculum'][self.start_stage]
                epochs_cfg = curr_cfg['epochs']
                total_epochs = sum(epochs_cfg) if isinstance(epochs_cfg, list) else epochs_cfg

                if self.start_epoch >= total_epochs:
                    logger.info(f"Stage {self.start_stage + 1} completed.")
                    self.start_stage += 1
                    self.start_epoch = 0
                    logger.info(f"Resuming from Stage {self.start_stage + 1}.")
            except IndexError:
                logger.error("Ckpt stage index out of bounds.")
                raise ValueError("Incompatible ckpt.")
            except Exception as e:
                logger.error(f"Error processing stage/epoch: {e}")

            if self.start_stage >= len(self.config['curriculum']):
                logger.warning("Training already completed.")
                sys.exit(0)

            # W&B resumption
            ckpt_wandb = ckpt.get('wandb_id')
            if ckpt_wandb and not self.wandb_id:
                self.wandb_id = ckpt_wandb
                logger.info(f"Resuming W&B ID: {self.wandb_id}")
            elif ckpt_wandb and self.wandb_id and ckpt_wandb != self.wandb_id:
                logger.warning("W&B ID mismatch!")

            # Optimizer and scheduler
            self.saved_optimizer_state = ckpt.get('optimizer_state_dict')
            self.saved_scheduler_state = ckpt.get('scheduler_state_dict')
            if self.saved_optimizer_state:
                logger.info("Opt state found.")
            if self.saved_scheduler_state:
                logger.info("Sched state found.")

            # EMA restoration
            if self.ema:
                ema_state = ckpt.get('ema_state_dict')
                if ema_state:
                    try:
                        if len(ema_state) == len(self.ema.shadow):
                            self.ema.shadow = ema_state
                            logger.info("EMA state restored.")
                        else:
                            logger.warning("EMA key mismatch.")
                            self.ema = EMA(self.model, decay=self.ema.decay)
                    except Exception as e:
                        logger.warning(f"Failed restore EMA: {e}")
                else:
                    logger.warning("No EMA state in ckpt.")

            logger.info(
                f"Ckpt loaded. Resuming @ Stage {self.start_stage + 1}, "
                f"Epoch {self.start_epoch + 1}"
            )

        except FileNotFoundError:
            pass
        except Exception as e:
            logger.error(f"Failed load ckpt '{ckpt_path}': {e}")
            self.start_stage = 0
            self.start_epoch = 0
            self.global_step = 0
            self.best_val_loss = float('inf')
            self.saved_optimizer_state = None
            self.saved_scheduler_state = None
            self.resume_from = None


# ==============================================================================
# SECTION 3: Entry Point & Argument Parsing
# ==============================================================================
def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train Non-Conditional AV1 Restorer",
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
        trainer = NonConditionalTrainer(
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