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
Key Optimizations:
  1. Persistent workers (eliminate respawn overhead)
  2. Smart caching (file lists shared across stages)
  3. Prefetch factor tuning (maximize GPU utilization)
  4. Mixed precision training (2× faster, half VRAM)
  5. EMA with smart validation (stable metrics)
  6. Gradient accumulation support (virtual large batches)
  7. Stage-specific loss configs (curriculum flexibility)
  8. Robust checkpointing (resume from any point)

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
Version: 4.0
License: MIT
==============================================================================
"""

import sys
import yaml
import argparse
import logging
from pathlib import Path
from typing import Dict, Any, Tuple, Optional
from collections import OrderedDict, defaultdict
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

from av1_restorer.models.av1_restorer import create_av1_restorer

try:
    from utils.loss import CombinedLoss
    from utils.av1_dataset import create_dataset  # Optimized factory
except ImportError as e:
    print(f"Failed to import utilities: {e}")
    sys.exit(1)

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(message)s'
)
logger = logging.getLogger("ConditionalTrainer")


# ============================================================================
# SECTION 1: EXPONENTIAL MOVING AVERAGE (EMA)
# ============================================================================

class EMA:
    """
    Exponential Moving Average for stable validation.
    
    Formula: shadow = decay × shadow + (1 - decay) × params
    Typical decay: 0.9999 (averages ~10K steps)
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
        """Update shadow parameters after optimizer step."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name].data.mul_(self.decay).add_(
                    param.data, alpha=1.0 - self.decay
                )
    
    def apply_shadow(self):
        """Swap model params with EMA shadow for validation."""
        self.backup = {
            name: param.data.clone()
            for name, param in self.model.named_parameters()
            if param.requires_grad
        }
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data.copy_(self.shadow[name])
    
    def restore(self):
        """Restore original params after validation."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data.copy_(self.backup[name])
        self.backup = {}


# ============================================================================
# SECTION 2: OPTIMIZED TRAINER
# ============================================================================

class ConditionalUNetTrainer:
    """
    Config-driven trainer for Conditional AV1 U-Net Restorers.
    Production-ready with zero CPU bottleneck.
     
    Automatically detects CRF-only vs CRF+Preset based on preset_range:
        - If preset_range is single value (e.g., [4,4]): CRF-only mode
        - Otherwise: CRF+Preset mode

    Features:
      - Smart dataset caching (file lists shared across stages)
      - Persistent workers (eliminate respawn overhead)
      - Optimal prefetching (maximize GPU utilization)
      - Mixed precision training (AMP)
      - EMA validation
      - Stage-specific loss configurations
      - Robust checkpointing
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
        
        # File list caches (shared across stages)
        self.train_cache = {}
        self.val_cache = {}
        
        # Setup
        self._setup_system()
        self.model = self._setup_model()
        self.loss_fn = self._setup_loss(self.config['loss'])
        
        # EMA
        ema_cfg = self.config['optimizer']
        self.ema = None
        if ema_cfg.get('use_ema', False):
            self.ema = EMA(self.model, decay=ema_cfg.get('ema_decay', 0.9999))
            logger.info(f"EMA enabled (decay={ema_cfg['ema_decay']})")
        
        # Logging
        self._setup_logging()
        self._print_summary()
        
        # Training state
        self.start_stage = 0
        self.start_epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')
        self.saved_optimizer_state = None
        self.saved_scheduler_state = None
        self.val_samples = None
        
        # Resume if requested
        if self.resume_from:
            self._load_checkpoint()
        
        # Compile model (PyTorch 2.0+)
        if self.device.type == 'cuda' and not self.resume_from:
            try:
                logger.info("Compiling model (torch.compile)...")
                self.model = torch.compile(self.model, mode='reduce-overhead')
                logger.info("✓ Model compiled successfully")
            except Exception as e:
                logger.warning(f"torch.compile failed: {e}")
    
    # ========== SETUP METHODS ==========
    
    def _load_config(self, path: str) -> Dict[str, Any]:
        """Load YAML configuration."""
        path = Path(path).expanduser()
        if not path.exists():
            raise FileNotFoundError(f"Config not found: {path}")
        
        with open(path, 'r') as f:
            config = yaml.safe_load(f)
        
        logger.info(f"Config loaded: {path}")
        return config
    
    def _setup_system(self):
        """Setup device, seeds, checkpointing, AMP."""
        sys_cfg = self.config['system']
        
        # Device auto-detection
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
                logger.warning("Device: CPU (slow)")
        else:
            self.device = torch.device(device_str)
            logger.info(f"Device: {self.device}")
        
        # CUDA optimizations
        if self.device.type == 'cuda':
            # old api
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            # new api
            # torch.backends.cuda.matmul.fp32_precision = 'tf32' 
            logger.info("Enabled TF32 for CUDA")
        
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
        
        # Mixed precision
        self.use_amp = sys_cfg.get('mixed_precision', False)
        if self.use_amp and self.device.type == 'cuda':
            self.scaler = torch.amp.GradScaler('cuda', enabled=True)
            logger.info("AMP: Enabled with GradScaler")
        elif self.use_amp:
            self.scaler = None
            logger.info(f"AMP: Enabled without GradScaler ({self.device.type})")
        else:
            self.scaler = None
            logger.info("AMP: Disabled")
    
    def _setup_model(self) -> nn.Module:
        """Create model and detect conditioning mode."""
        model_cfg = self.config['model']
        dset_cfg = self.config['dataset']
        
        if model_cfg.get('type') != 'unet':
            raise ValueError(f"Only 'unet' supported, got '{model_cfg['type']}'")
        
        size = model_cfg['size']
        crf_range = tuple(dset_cfg['crf_range'])
        preset_range = tuple(dset_cfg.get('preset_range', [0, 8]))
        norm_range = tuple(dset_cfg.get('norm_range', [-1, 1]))
        
        # Auto-detect CRF-only mode
        self.model_needs_preset = (preset_range[0] != preset_range[1])
        mode_str = "CRF+Preset" if self.model_needs_preset else "CRF-Only"
        logger.info(f"Creating model: unet/{size} ({mode_str})")
        
        model = create_av1_restorer(
            size=size,
            crf_range=crf_range,
            preset_range=preset_range,
            norm_range=norm_range
        )
        
        return model.to(self.device)
    
    def _setup_loss(self, loss_config: dict) -> nn.Module:
        """Initialize loss function."""
        norm_range = tuple(self.config['dataset'].get('norm_range', [-1, 1]))
        loss_fn = CombinedLoss(loss_config, norm_range=norm_range)
        logger.info("Loss function initialized")
        return loss_fn.to(self.device)
    
    def _setup_logging(self):
        """Initialize W&B logging."""
        if not (self.config['project']['log_to_wandb'] and WANDB_AVAILABLE):
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
        """Print training configuration."""
        logger.info("="*80)
        logger.info(f"{'CONDITIONAL U-NET TRAINING CONFIGURATION':^80}")
        logger.info("="*80)
        
        # Project
        proj = self.config['project']
        logger.info(f"Project: {proj['name']}")
        logger.info(f"Experiment: {proj['experiment_name']}")
        
        # Model
        model_cfg = self.config['model']
        params = sum(p.numel() for p in self.model.parameters())
        mode_str = "CRF+Preset" if self.model_needs_preset else "CRF-Only"
        logger.info(f"Model: {model_cfg['type']}/{model_cfg['size']} ({mode_str})")
        logger.info(f"Parameters: {params:,} ({params/1e6:.2f}M)")
        
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
            if isinstance(patch, list):
                logger.info(
                    f"  Stage {i+1}: {epochs[0]}+{epochs[1]} epochs, "
                    f"{patch[0]}/{patch[1]}px, batch {batch[0]}/{batch[1]}"
                )
            else:
                logger.info(
                    f"  Stage {i+1}: {epochs} epochs, {patch}px, batch {batch}"
                )
        
        logger.info(f"Device: {self.device} | AMP: {self.use_amp} | EMA: {self.ema is not None}")
        logger.info("="*80)
    
    # ========== DATASET CREATION (OPTIMIZED) ==========
    
    def _create_dataloaders(
        self,
        stage_config: dict
    ) -> Tuple[DataLoader, DataLoader]:
        """
        Create optimized dataloaders with smart caching.
        
        Optimization: File lists are scanned once and cached across stages.
        """
        data_cfg = self.config['data']
        dset_cfg = self.config['dataset']
        sys_cfg = self.config['system']
        
        # Extract stage params
        patch_size = stage_config['patch_size']
        batch_size = stage_config['batch_size']
        crf_range = stage_config.get('crf_range', dset_cfg['crf_range'])
        augment_factor_train = stage_config.get('augment_factor_train', 1)
        augment_factor_val = stage_config.get('augment_factor_val', 1)
        
        # Handle progressive training
        if isinstance(patch_size, list):
            patch_size = patch_size[0]
            batch_size = batch_size[0]
        
        # Cache keys
        train_key = f"{data_cfg['train_lq_root']}_{tuple(crf_range)}"
        val_key = f"{data_cfg['val_lq_root']}_{tuple(crf_range)}"
        
        # Common args
        common_args = {
            'hq_ext': data_cfg.get('hq_ext', '.png'),
            'lq_ext': data_cfg.get('lq_ext', '.avif'),
            'crf_range': tuple(crf_range),
            'preset_range': tuple(dset_cfg['preset_range']),
            'norm_range': tuple(dset_cfg.get('norm_range', [-1, 1])),
        }
        
        # Training dataset
        train_ds = create_dataset(
            lq_root=data_cfg['train_lq_root'],
            hq_root=data_cfg['train_hq_root'],
            patch_size=patch_size,
            crop_mode='random',
            augment_factor=augment_factor_train,
            cached_image_pairs=self.train_cache.get(train_key),
            **common_args
        )
        
        # Cache file list if first time
        if train_key not in self.train_cache and hasattr(train_ds, 'image_pairs'):
            self.train_cache[train_key] = train_ds.image_pairs
            logger.info(f"Cached train file list ({len(train_ds.image_pairs):,} pairs)")
        
        # Validation dataset
        val_ds = create_dataset(
            lq_root=data_cfg['val_lq_root'],
            hq_root=data_cfg['val_hq_root'],
            patch_size=patch_size,
            crop_mode='center',
            augment_factor=augment_factor_val,
            cached_image_pairs=self.val_cache.get(val_key),
            return_metadata=True,  # For W&B logging
            **common_args
        )
        
        # Cache file list if first time
        if val_key not in self.val_cache and hasattr(val_ds, 'image_pairs'):
            self.val_cache[val_key] = val_ds.image_pairs
            logger.info(f"Cached val file list ({len(val_ds.image_pairs):,} pairs)")
        
        # Dataloader configuration
        num_workers = sys_cfg.get('num_workers', 8)
        pin_memory = (self.device.type == 'cuda')
        
        # Training loader (optimized for throughput)
        train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=True,
            persistent_workers=(num_workers > 0),  # KEY OPTIMIZATION
            prefetch_factor=4 if num_workers > 0 else None  # Aggressive prefetch
        )
        
        # Validation loader (less aggressive)
        val_loader = DataLoader(
            val_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=max(2, num_workers // 2),
            pin_memory=pin_memory,
            drop_last=False,
            persistent_workers=(num_workers > 0),
            prefetch_factor=2 if num_workers > 0 else None
        )
        
        logger.info(
            f"Dataloaders: {len(train_ds):,} train, {len(val_ds):,} val "
            f"({num_workers} workers, prefetch={4 if num_workers > 0 else 0})"
        )
        
        return train_loader, val_loader
    
    def _setup_optimizer(
        self,
        total_steps: int
    ) -> Tuple[torch.optim.Optimizer, Optional[Any]]:
        """Create optimizer and scheduler."""
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
        
        # Scheduler
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
    
    # ========== TRAINING LOOP ==========
    
    def train(self):
        """Main training loop."""
        logger.info("\n" + "="*80)
        logger.info("STARTING TRAINING")
        logger.info("="*80 + "\n")
        
        curriculum = self.config['curriculum']
        
        # Estimate total steps
        estimated_steps = 0
        stage_lengths = []
        try:
            for stage_cfg in curriculum:
                temp_loader, _ = self._create_dataloaders(stage_cfg)
                stage_lengths.append(len(temp_loader))
                epochs = stage_cfg['epochs']
                if isinstance(epochs, list):
                    estimated_steps += sum(epochs) * len(temp_loader)
                else:
                    estimated_steps += epochs * len(temp_loader)
                del temp_loader
            logger.info(f"Estimated total steps: {estimated_steps:,}")
        except:
            estimated_steps = 10000
            stage_lengths = [100] * len(curriculum)
        
        # Setup optimizer/scheduler
        optimizer, scheduler = self._setup_optimizer(estimated_steps)
        
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
            logger.info(f"\n▶ Starting Stage {stage_idx+1}/{len(curriculum)}")
            
            # Sample validation images for this stage
            self.val_samples = self._get_val_samples(stage_cfg)
            
            self._run_stage(stage_idx, stage_cfg, optimizer, scheduler, stage_lengths[stage_idx])
            
            # Save stage completion checkpoint
            epochs = stage_cfg['epochs']
            last_epoch = (sum(epochs) - 1) if isinstance(epochs, list) else (epochs - 1)
            self._save_checkpoint(last_epoch, stage_idx, optimizer, scheduler, is_stage_complete=True)
            
            self.start_epoch = 0
        
        logger.info("\n" + "="*80)
        logger.info("🎉 TRAINING COMPLETE!")
        logger.info(f"Best validation loss: {self.best_val_loss:.6f}")
        logger.info("="*80 + "\n")
        
        if WANDB_AVAILABLE and wandb.run:
            wandb.finish()
    
    def _get_val_samples(self, stage_cfg: dict) -> Optional[Dict[str, Any]]:
        """Sample fixed validation images for W&B logging."""
        try:
            data_cfg = self.config['data']
            dset_cfg = self.config['dataset']
            num_samples = self.config['training'].get('num_val_samples_to_log', 4)
            
            # Get patch size from stage
            patch_size = stage_cfg['patch_size']
            if isinstance(patch_size, list):
                patch_size = patch_size[-1]
            
            # Check cache
            val_key = f"{data_cfg['val_lq_root']}_{tuple(dset_cfg['crf_range'])}"
            cached = self.val_cache.get(val_key)
            
            # Create dataset
            val_ds = create_dataset(
                lq_root=data_cfg['val_lq_root'],
                hq_root=data_cfg['val_hq_root'],
                hq_ext=data_cfg.get('hq_ext', '.png'),
                lq_ext=data_cfg.get('lq_ext', '.avif'),
                patch_size=patch_size,
                crop_mode='center',
                augment_factor=1,
                crf_range=tuple(dset_cfg['crf_range']),
                preset_range=tuple(dset_cfg['preset_range']),
                norm_range=tuple(dset_cfg.get('norm_range', [-1, 1])),
                cached_image_pairs=cached,
                return_metadata=True
            )
            
            if len(val_ds) == 0:
                return None
            
            num_samples = min(num_samples, len(val_ds))
            loader = DataLoader(val_ds, batch_size=num_samples, shuffle=False)
            batch = next(iter(loader))
            
            return {
                k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }
        except Exception as e:
            logger.warning(f"Could not sample validation images: {e}")
            return None
    
    def _run_stage(self, stage_idx, stage_cfg, optimizer, scheduler, loader_len):
        """Run a single curriculum stage."""
        # Update loss function if stage has custom config
        current_loss_config = stage_cfg.get('loss', self.config['loss'])
        self.loss_fn = self._setup_loss(current_loss_config)
        
        is_progressive = isinstance(stage_cfg.get('patch_size'), list)
        
        # Restore state if resuming mid-stage
        if stage_idx == self.start_stage and self.start_epoch > 0 and self.saved_optimizer_state:
            try:
                optimizer.load_state_dict(self.saved_optimizer_state)
                if scheduler and self.saved_scheduler_state:
                    scheduler.load_state_dict(self.saved_scheduler_state)
                logger.info("Restored optimizer/scheduler state")
            except Exception as e:
                logger.error(f"Failed to restore state: {e}")
            finally:
                self.saved_optimizer_state = None
                self.saved_scheduler_state = None
        
        if is_progressive:
            # Progressive stage
            patch_sizes = stage_cfg['patch_size']
            batch_sizes = stage_cfg['batch_size']
            epochs_per_sub = stage_cfg['epochs']
            total_epochs = sum(epochs_per_sub)
            
            cumulative_epochs = 0
            for sub_idx in range(len(patch_sizes)):
                sub_patch = patch_sizes[sub_idx]
                sub_batch = batch_sizes[sub_idx]
                sub_epochs = epochs_per_sub[sub_idx]
                
                # Skip completed sub-stages
                if self.start_epoch >= cumulative_epochs + sub_epochs:
                    cumulative_epochs += sub_epochs
                    continue
                
                logger.info(
                    f"  → Sub-stage {stage_idx+1}.{sub_idx+1}: "
                    f"{sub_epochs} epochs @ {sub_patch}px, batch {sub_batch}"
                )
                
                # Create loaders
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
                self.start_epoch = 0
        else:
            # Simple stage
            epochs = stage_cfg['epochs']
            train_loader, val_loader = self._create_dataloaders(stage_cfg)
            
            self._run_epochs(
                stage_idx, self.start_epoch, epochs,
                0, epochs,
                train_loader, val_loader, optimizer, scheduler, -1
            )
    
    def _run_epochs(self, stage_idx, start_epoch, num_epochs, epoch_offset, total_epochs,
                   train_loader, val_loader, optimizer, scheduler, sub_stage_idx=-1):
        """Run training and validation for specified epochs."""
        for epoch_in_segment in range(start_epoch, num_epochs):
            current_epoch = epoch_offset + epoch_in_segment
            
            self._train_epoch(
                current_epoch, total_epochs, stage_idx,
                train_loader, optimizer, scheduler, sub_stage_idx
            )
            
            self._validate_and_checkpoint(
                current_epoch, total_epochs, stage_idx,
                val_loader, optimizer, scheduler
            )
    
    def _train_epoch(self, epoch, total_epochs, stage_idx, train_loader, optimizer, scheduler, sub_stage_idx=-1):
        """Train for one epoch."""
        self.model.train()
        
        # Progress bar
        stage_desc = f"Stage {stage_idx+1}"
        if sub_stage_idx != -1:
            stage_desc += f".{sub_stage_idx+1}"
        pbar = tqdm(train_loader, desc=f"{stage_desc} | Epoch {epoch+1}/{total_epochs}", leave=True, dynamic_ncols=True)
        
        epoch_total_loss = 0.0
        epoch_loss_components = defaultdict(float)
        
        for batch_idx, batch in enumerate(pbar):
            try:
                lq = batch['lq'].to(self.device, non_blocking=True)
                hq = batch['hq'].to(self.device, non_blocking=True)
                crf = batch['crf'].to(self.device, non_blocking=True)
                preset = batch['preset'].to(self.device, non_blocking=True) if self.model_needs_preset else None
            except Exception as e:
                logger.error(f"Error loading batch: {e}")
                raise
            
            # --- STABILITY FIX: Run loss in float32 ---
            # Forward pass (respects mixed_precision config from your YAML)
            with torch.amp.autocast(device_type=str(self.device.type), enabled=self.use_amp and self.device.type == 'cuda'):
                restored = self.model(lq, crf, preset) if self.model_needs_preset else self.model(lq, crf)

            # Loss computation (Forced to float32 for stability)
            with torch.amp.autocast(device_type=str(self.device.type), enabled=False):
                try:
                    loss, loss_dict = self.loss_fn(restored.float(), hq.float())
                    
                    if not torch.isfinite(loss):
                        logger.error(f"Loss NaN/Inf at step {self.global_step}")
                        self._save_checkpoint(epoch, stage_idx, optimizer, scheduler, is_emergency=True)
                        raise RuntimeError("Loss became NaN/Inf")
                except Exception as e:
                    logger.error(f"Forward pass or Loss calculation error: {e}")
                    self._save_checkpoint(epoch, stage_idx, optimizer, scheduler, is_emergency=True)
                    raise
            # --- END STABILITY FIX ---

            # Backward pass
            optimizer.zero_grad(set_to_none=True)
            try:
                if self.scaler:
                    # --- This block will run for CUDA ---
                    self.scaler.scale(loss).backward()
                    self.scaler.unscale_(optimizer) 
                    grad_clip = self.config['training'].get('grad_clip_norm', 0.0)
                    if grad_clip > 0:
                        # This clip is safe because GradScaler will skip if grads are NaN/Inf
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip)
                    self.scaler.step(optimizer)
                    self.scaler.update()
                else:
                    # --- This block will run for MPS ---
                    loss.backward()
                    # --- ! PERMANENT MPS GRADIENT SAFETY NET ! ---
                    # We must check for NaN grads *before* clip_grad_norm_
                    # This acts as our safety net since GradScaler is not available.
                    for name, param in self.model.named_parameters():
                        if param.grad is not None and not torch.isfinite(param.grad).all():
                            logger.error(f"!!! NaN/Inf GRADIENT (MPS) DETECTED at step {self.global_step} !!!")
                            logger.error(f"Parameter: {name}")
                            logger.error("This will corrupt model weights. Skipping optimizer step.")
                            optimizer.zero_grad() # Clear the bad gradients
                            raise StopIteration("NaN Gradient Detected") # Use custom error to skip batch
                    # --- ! END MPS SAFETY NET ! ---
                    grad_clip = self.config['training'].get('grad_clip_norm', 0.0)
                    if grad_clip > 0:
                        # This will now only run if gradients are finite
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip)
                    optimizer.step()

            except StopIteration as e: # Catch our custom "skip batch" error
                logger.warning(f"Skipping optimizer step for batch {batch_idx} due to: {e}")
                continue # This skips the rest of the loop and goes to the next batch

            except Exception as e:
                logger.error(f"Backward pass error: {e}")
                self._save_checkpoint(epoch, stage_idx, optimizer, scheduler, is_emergency=True)
                raise
            
            if scheduler:
                scheduler.step()
            if self.ema:
                self.ema.update()
            
            # Metrics
            batch_loss = loss.item()
            epoch_total_loss += batch_loss
            for k, v in loss_dict.items():
                epoch_loss_components[k] += v.item()
            
            lr = optimizer.param_groups[0]['lr']
            pbar.set_postfix(Loss=f"{batch_loss:.4f}", LR=f"{lr:.2e}", refresh=(batch_idx % 10 == 0))
            
            # W&B logging
            if WANDB_AVAILABLE and wandb.run:
                log_freq = self.config['training'].get('log_every_n_steps', 50)
                if self.global_step % log_freq == 0:
                    wandb_log = {f'train/{k.replace("loss_", "")}': v.item() for k, v in loss_dict.items()}
                    wandb_log['train/learning_rate'] = lr
                    wandb.log(wandb_log, step=self.global_step)
            
            self.global_step += 1
        
        # Epoch summary
        if len(train_loader) > 0:
            avg_loss = epoch_total_loss / len(train_loader)
            logger.info(f"Epoch {epoch+1}/{total_epochs} | Avg Train Loss: {avg_loss:.6f}")
            
            if WANDB_AVAILABLE and wandb.run:
                wandb_log = {
                    f'epoch/avg_{k.replace("loss_", "")}': v / len(train_loader)
                    for k, v in epoch_loss_components.items()
                }
                wandb_log['epoch/avg_total_loss'] = avg_loss
                wandb.log(wandb_log, step=self.global_step)

    def _train_epoch_debug(self, epoch, total_epochs, stage_idx, train_loader, optimizer, scheduler, sub_stage_idx=-1):
        """Train for one epoch."""
        self.model.train()
        
        # Progress bar
        stage_desc = f"Stage {stage_idx+1}"
        if sub_stage_idx != -1:
            stage_desc += f".{sub_stage_idx+1}"
        pbar = tqdm(train_loader, desc=f"{stage_desc} | Epoch {epoch+1}/{total_epochs}", leave=True, dynamic_ncols=True)
        
        epoch_total_loss = 0.0
        epoch_loss_components = defaultdict(float)
        
        for batch_idx, batch in enumerate(pbar):
            try:
                lq = batch['lq'].to(self.device, non_blocking=True)
                hq = batch['hq'].to(self.device, non_blocking=True)
                crf = batch['crf'].to(self.device, non_blocking=True)
                preset = batch['preset'].to(self.device, non_blocking=True) if self.model_needs_preset else None
            except Exception as e:
                logger.error(f"Error loading batch: {e}")
                raise

            # --- ! STRATEGIC DEBUG CHECK 1: WEIGHTS ! ---
            # Check model weights *before* the forward pass.
            # If this triggers, it proves the PREVIOUS optimizer.step() corrupted the model.
            if batch_idx > 0: # Only check after the first step
                check_param = self.model.conditioning_embedder.crf_embedder[0].weight
                if torch.isnan(check_param).any():
                    logger.error(f"!!! WEIGHTS ARE NaN at start of step {self.global_step} !!!")
                    logger.error("This means the PREVIOUS optimizer step (step {self.global_step - 1}) failed.")
                    self._save_checkpoint(epoch, stage_idx, optimizer, scheduler, is_emergency=True)
                    raise ValueError(f"Weights became NaN at step {self.global_step}. Check optimizer/grad clipping.")
            # --- ! END DEBUG CHECK 1 ! ---
                
            # --- STABILITY FIX: Run loss in float32 ---
            # Forward pass (Model in mixed precision)
            with torch.amp.autocast(device_type=str(self.device.type), enabled=False): # self.use_amp):
                # This is the line that *reveals* the crash, but doesn't *cause* it
                restored = self.model(lq, crf, preset) if self.model_needs_preset else self.model(lq, crf)

            # Loss computation (Forced to float32)
            with torch.amp.autocast(device_type=str(self.device.type), enabled=False):
                try:
                    loss, loss_dict = self.loss_fn(restored.float(), hq.float())
                    
                    if not torch.isfinite(loss):
                        logger.error(f"Loss NaN/Inf at step {self.global_step}")
                        self._save_checkpoint(epoch, stage_idx, optimizer, scheduler, is_emergency=True)
                        raise RuntimeError("Loss became NaN/Inf")
                except Exception as e:
                    # This is the block that is catching your current crash
                    logger.error(f"Forward pass or Loss calculation error: {e}")
                    self._save_checkpoint(epoch, stage_idx, optimizer, scheduler, is_emergency=True)
                    raise
            # --- END STABILITY FIX ---

            # Backward pass
            optimizer.zero_grad(set_to_none=True)
            try:
                if self.scaler:
                    self.scaler.scale(loss).backward()
                    # --- ! STRATEGIC DEBUG CHECK 2: GRADIENTS ! ---
                    self.scaler.unscale_(optimizer) # Unscale before checking
                    for name, param in self.model.named_parameters():
                        if param.grad is not None and not torch.isfinite(param.grad).all():
                            logger.error(f"!!! NaN/Inf GRADIENT DETECTED at step {self.global_step} !!!")
                            logger.error(f"Parameter: {name}")
                            logger.error("This will corrupt model weights. Skipping optimizer step.")
                            optimizer.zero_grad() # Clear the bad gradients
                            raise StopIteration("NaN Gradient Detected") # Use custom error to skip batch
                    # --- ! END DEBUG CHECK 2 ! ---
                    grad_clip = self.config['training'].get('grad_clip_norm', 0.0)
                    if grad_clip > 0:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip)
                    self.scaler.step(optimizer)
                    self.scaler.update()
                else:
                    loss.backward()
                    # --- ! STRATEGIC DEBUG CHECK 2: GRADIENTS ! ---
                    for name, param in self.model.named_parameters():
                        if param.grad is not None and not torch.isfinite(param.grad).all():
                            logger.error(f"!!! NaN/Inf GRADIENT DETECTED at step {self.global_step} !!!")
                            logger.error(f"Parameter: {name}")
                            logger.error("This will corrupt model weights. Skipping optimizer step.")
                            optimizer.zero_grad() # Clear the bad gradients
                            raise StopIteration("NaN Gradient Detected") # Use custom error to skip batch
                    # --- ! END DEBUG CHECK 2 ! ---
                    grad_clip = self.config['training'].get('grad_clip_norm', 0.0)
                    if grad_clip > 0:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip)
                    optimizer.step()

            except StopIteration as e: # Catch our custom "skip batch" error
                logger.warning(f"Skipping optimizer step for batch {batch_idx} due to: {e}")
                continue # This skips the rest of the loop and goes to the next batch

            except Exception as e:
                logger.error(f"Backward pass error: {e}")
                self._save_checkpoint(epoch, stage_idx, optimizer, scheduler, is_emergency=True)
                raise
            
            if scheduler:
                scheduler.step()
            if self.ema:
                self.ema.update()
            
            # Metrics
            batch_loss = loss.item()
            epoch_total_loss += batch_loss
            for k, v in loss_dict.items():
                epoch_loss_components[k] += v.item()
            
            lr = optimizer.param_groups[0]['lr']
            pbar.set_postfix(Loss=f"{batch_loss:.4f}", LR=f"{lr:.2e}", refresh=(batch_idx % 10 == 0))
            
            # W&B logging
            if WANDB_AVAILABLE and wandb.run:
                log_freq = self.config['training'].get('log_every_n_steps', 50)
                if self.global_step % log_freq == 0:
                    wandb_log = {f'train/{k.replace("loss_", "")}': v.item() for k, v in loss_dict.items()}
                    wandb_log['train/learning_rate'] = lr
                    wandb.log(wandb_log, step=self.global_step)
            
            self.global_step += 1
        
        # Epoch summary
        if len(train_loader) > 0:
            avg_loss = epoch_total_loss / len(train_loader)
            logger.info(f"Epoch {epoch+1}/{total_epochs} | Avg Train Loss: {avg_loss:.6f}")
            
            if WANDB_AVAILABLE and wandb.run:
                wandb_log = {
                    f'epoch/avg_{k.replace("loss_", "")}': v / len(train_loader)
                    for k, v in epoch_loss_components.items()
                }
                wandb_log['epoch/avg_total_loss'] = avg_loss
                wandb.log(wandb_log, step=self.global_step)

    def _validate_and_checkpoint(self, epoch, total_epochs, stage_idx, val_loader, optimizer, scheduler):
        """Run validation and save checkpoints."""
        val_freq = self.config['training'].get('validate_every_n_epochs', 1)
        is_last_epoch = (epoch == total_epochs - 1)
        
        if val_freq > 0 and ((epoch + 1) % val_freq == 0 or is_last_epoch):
            val_loss = self._validate(epoch, total_epochs, stage_idx, val_loader)
            
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                logger.info(f"✨ New best: {val_loss:.6f}")
                self._save_checkpoint(epoch, stage_idx, optimizer, scheduler, is_best=True)
            else:
                self._save_checkpoint(epoch, stage_idx, optimizer, scheduler)
        
        save_freq = self.config['checkpoint'].get('save_every_n_epochs', 0)
        if save_freq > 0 and (epoch + 1) % save_freq == 0:
            validation_ran = (val_freq > 0 and ((epoch + 1) % val_freq == 0 or is_last_epoch))
            if not validation_ran:
                self._save_checkpoint(epoch, stage_idx, optimizer, scheduler)
    
    @torch.no_grad()
    def _validate(self, epoch, total_epochs, stage_idx, val_loader) -> float:
        """Run validation."""
        self.model.eval()
        if self.ema:
            self.ema.apply_shadow()
        
        total_loss = 0.0
        total_components = defaultdict(float)
        baseline_l1, baseline_l2 = 0.0, 0.0
        restored_l1, restored_l2 = 0.0, 0.0
        
        for batch in tqdm(val_loader, desc=f"Validating Epoch {epoch+1}/{total_epochs}", leave=False):
            try:
                lq = batch['lq'].to(self.device, non_blocking=True)
                hq = batch['hq'].to(self.device, non_blocking=True)
                crf = batch['crf'].to(self.device, non_blocking=True)
                preset = batch['preset'].to(self.device, non_blocking=True) if self.model_needs_preset else None
                
                with torch.amp.autocast(device_type=str(self.device.type), enabled=self.use_amp and self.device.type == 'cuda'):
                    inference_fn = getattr(self.model, 'inference', self.model.forward)
                    restored = inference_fn(lq, crf, preset) if self.model_needs_preset else inference_fn(lq, crf)
                    loss, loss_dict = self.loss_fn(restored, hq)
                
                total_loss += loss.item()
                for k, v in loss_dict.items():
                    total_components[k] += v.item()
                baseline_l1 += F.l1_loss(lq, hq).item()
                baseline_l2 += F.mse_loss(lq, hq).item()
                restored_l1 += F.l1_loss(restored, hq).item()
                restored_l2 += F.mse_loss(restored, hq).item()
            except Exception as e:
                logger.error(f"Validation error: {e}")
                continue
        
        num_batches = len(val_loader)
        if num_batches == 0:
            if self.ema:
                self.ema.restore()
            return float('inf')
        
        avg_loss = total_loss / num_batches
        avg_baseline_l1 = baseline_l1 / num_batches
        avg_restored_l1 = restored_l1 / num_batches
        avg_baseline_l2 = baseline_l2 / num_batches
        avg_restored_l2 = restored_l2 / num_batches
        l1_improvement = ((avg_baseline_l1 - avg_restored_l1) / (avg_baseline_l1 + 1e-9)) * 100
        l2_improvement = ((avg_baseline_l2 - avg_restored_l2) / (avg_baseline_l2 + 1e-9)) * 100

        
        logger.info(f"📊 Validation Epoch {epoch+1}/{total_epochs}:")
        logger.info(f"    Avg Loss: {avg_loss:.6f}")
        logger.info(f"    L1: {avg_restored_l1:.6f} (Baseline: {avg_baseline_l1:.6f}, ↓ {l1_improvement:.2f}%)")
        logger.info(f"    L2: {avg_restored_l2:.6f} (Baseline: {avg_baseline_l2:.6f}, ↓ {l2_improvement:.2f}%)")

        
        if WANDB_AVAILABLE and wandb.run:
            wandb_dict = {
                'val/epoch': epoch + 1,
                'val/stage': stage_idx + 1,
                'val/avg_total_loss': avg_loss,
                **{f'val/avg_{k.replace("loss_", "")}': v / num_batches for k, v in total_components.items()},
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
                    logger.warning(f"Failed to log samples: {e}")
            
            wandb.log(wandb_dict, step=self.global_step)
        
        if self.ema:
            self.ema.restore()
        
        return avg_loss
    
    @torch.no_grad()
    def _log_visual_samples(self, epoch_num, stage_num):
        """Log visual samples to W&B."""
        if not self.val_samples:
            return
        
        lq = self.val_samples['lq']
        hq = self.val_samples['hq']
        crf = self.val_samples['crf']
        preset = self.val_samples.get('preset') if self.model_needs_preset else None
        
        inference_fn = getattr(self.model, 'inference', self.model.forward)
        
        with torch.amp.autocast(device_type=str(self.device.type), enabled=self.use_amp and self.device.type == 'cuda'):
            if self.model_needs_preset:
                restored = inference_fn(lq, crf, preset)
            else:
                restored = inference_fn(lq, crf)
        
        # Denormalize
        norm_range = tuple(self.config['dataset'].get('norm_range', [-1, 1]))
        if norm_range == (-1, 1):
            denorm = lambda t: (t + 1.0) / 2.0
            lq_display = denorm(lq)
            restored_display = denorm(restored)
            hq_display = denorm(hq)
        else:
            lq_display = lq
            restored_display = restored
            hq_display = hq
        
         # --- FIX: Replace NaNs before casting to uint8 ---
        lq_display = torch.nan_to_num(lq_display.clamp(0, 1), nan=0.0)
        restored_display = torch.nan_to_num(restored_display.clamp(0, 1), nan=0.0)
        hq_display = torch.nan_to_num(hq_display.clamp(0, 1), nan=0.0)
        
        images_to_log = []
        for i in range(min(4, lq.size(0))):
            base_name = self.val_samples.get('base_name', [""] * lq.size(0))[i]
            crf_val = int(crf[i].item())
            caption = f"S{stage_num} E{epoch_num} | {base_name} CRF{crf_val} | LQ-Restored-HQ"
            
            img_strip = torch.cat([lq_display[i], restored_display[i], hq_display[i]], dim=2)
            images_to_log.append(wandb.Image(img_strip, caption=caption))
        
        if images_to_log:
            wandb.log({"Validation Samples": images_to_log}, step=self.global_step)
    
    # ========== CHECKPOINTING ==========
    
    def _get_checkpoint_path(self, key: Optional[str]) -> Optional[Path]:
        """Resolve checkpoint key to path."""
        if not key:
            return None
        
        key_path = Path(key).expanduser()
        if key_path.is_file():
            return key_path.resolve()
        
        if self.checkpoint_dir:
            if key.lower() == 'latest':
                return (self.checkpoint_dir / 'latest.pth').resolve()
            if key.lower() == 'best':
                return (self.checkpoint_dir / 'best.pth').resolve()
            
            filename = key_path.name
            if not filename.endswith('.pth'):
                filename += '.pth'
            return (self.checkpoint_dir / filename).resolve()
        
        return None
    
    def _save_checkpoint(self, epoch, stage_idx, optimizer, scheduler,
                        is_best=False, is_stage_complete=False, is_emergency=False):
        """Save training state."""
        if not self.checkpoint_dir:
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
        
        if is_emergency:
            save_path = self.checkpoint_dir / "emergency_interrupt.pth"
        elif is_best:
            save_path = self.checkpoint_dir / "best.pth"
        elif is_stage_complete:
            save_path = self.checkpoint_dir / f"stage_{stage_idx+1:02d}_complete.pth"
        else:
            save_path = self.checkpoint_dir / "latest.pth"
        
        try:
            torch.save(state, save_path)
            if is_best or is_stage_complete or is_emergency:
                logger.info(f"✓ Saved checkpoint: {save_path.name}")
        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")
    
    def _load_checkpoint(self):
        """Load training state from checkpoint."""
        if not self.resume_from:
            return
        
        ckpt_path = self._get_checkpoint_path(self.resume_from)
        if not ckpt_path or not ckpt_path.exists():
            logger.error(f"Checkpoint not found: {self.resume_from}")
            self.resume_from = None
            return
        
        logger.info(f"Loading checkpoint: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=False)
        
        # Restore model
        ckpt_state_dict = ckpt['model_state_dict']
        new_state_dict = OrderedDict()

        for k, v in ckpt_state_dict.items():
            name = k[10:] if k.startswith('_orig_mod.') else k
            new_state_dict[name] = v

        missing, unexpected = self.model.load_state_dict(new_state_dict, strict=False)
        if unexpected:
            logger.warning(f"Unexpected keys: {unexpected}")
        if missing:
            logger.warning(f"Missing keys: {missing}")
        logger.info("Model state restored")
        
        # Restore training state
        self.global_step = ckpt.get('global_step', 0)
        self.best_val_loss = ckpt.get('best_val_loss', float('inf'))
        self.start_stage = ckpt.get('stage_idx', 0)
        self.start_epoch = ckpt.get('epoch', -1) + 1
        
        # Handle stage completion
        try:
            if self.start_stage >= len(self.config['curriculum']):
                logger.warning("Training already completed")
                sys.exit(0)
            
            curr_cfg = self.config['curriculum'][self.start_stage]
            epochs_cfg = curr_cfg['epochs']
            total_epochs = sum(epochs_cfg) if isinstance(epochs_cfg, list) else epochs_cfg
            
            if self.start_epoch >= total_epochs:
                self.start_stage += 1
                self.start_epoch = 0
                logger.info(f"Resuming from Stage {self.start_stage+1}")
        except IndexError:
            logger.error("Invalid checkpoint")
            sys.exit(1)
        
        # Restore W&B ID
        if ckpt.get('wandb_id') and not self.wandb_id:
            self.wandb_id = ckpt['wandb_id']
        
        # Store optimizer/scheduler state
        self.saved_optimizer_state = ckpt.get('optimizer_state_dict')
        self.saved_scheduler_state = ckpt.get('scheduler_state_dict')
        
        # Restore EMA
        if self.ema and 'ema_state_dict' in ckpt:
            try:
                self.ema.shadow = ckpt['ema_state_dict']
                logger.info("EMA state restored")
            except:
                logger.warning("Failed to restore EMA")
        
        logger.info(f"Resuming at Stage {self.start_stage+1}, Epoch {self.start_epoch+1}")


# ============================================================================
# ENTRY POINT
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="Train Conditional AV1 U-Net")
    parser.add_argument('--config', type=str, required=True, help="Path to YAML config")
    parser.add_argument('--resume', type=str, default=None, help="Resume from checkpoint")
    parser.add_argument('--wandb_id', type=str, default=None, help="W&B run ID")
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