# av1_restorer/train_av1_restorer.py
"""
Production Training Script for AV1 U-Net Restorer

Cross-platform support for CUDA, MPS, and CPU with proper AMP handling.

Author: Soham Mukherjee
Version: 2.1 Final (Cross-Platform Fix)
"""

import os
import sys
import yaml
import argparse
import logging
from pathlib import Path
from typing import Dict, Any, Tuple, Optional
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, OneCycleLR
from tqdm import tqdm
import torchvision

# FIXED: Cross-platform AMP imports
from torch import autocast  # General-purpose autocast (works for CUDA, MPS, CPU)
from torch.cuda.amp import GradScaler  # GradScaler is CUDA-specific (ignored on MPS/CPU)

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from av1_restorer.models.av1_unet_restorer import create_av1_restorer
from utils.loss import CombinedLoss
from utils.av1_dataset import AV1Dataset

# Optional W&B
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("⚠️  Warning: wandb not available. Install with: pip install wandb")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(name)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


# ==============================================================================
# SECTION 1: Exponential Moving Average (EMA)
# ==============================================================================

class EMA:
    """
    Exponential Moving Average of model parameters.
    Improves generalization by smoothing weight updates.
    """
    def __init__(self, model: nn.Module, decay: float = 0.9999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        
        # Initialize shadow parameters
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()
    
    def update(self):
        """Update EMA parameters."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_average = (
                    self.decay * self.shadow[name] + 
                    (1.0 - self.decay) * param.data
                )
                self.shadow[name] = new_average.clone()
    
    def apply_shadow(self):
        """Replace model parameters with EMA shadow parameters."""
        self.backup = {}
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
                param.data = self.shadow[name].clone()
    
    def restore(self):
        """Restore original model parameters."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data = self.backup[name].clone()
        self.backup = {}


# ==============================================================================
# SECTION 2: Main Trainer Class
# ==============================================================================

class AV1RestorerTrainer:
    """
    Professional trainer for AV1 U-Net Restorer with full cross-platform support.
    """
    
    def __init__(
        self,
        config_path: str,
        resume_from: Optional[str] = None,
        wandb_id: Optional[str] = None
    ):
        """Initialize trainer."""
        # Load configuration
        self.config = self._load_config(config_path)
        self.resume_from = resume_from
        self.wandb_id = wandb_id
        
        # Setup system (device, seeds, directories)
        self._setup_system()
        
        # Initialize model and loss
        self.model = self._setup_model()
        self.loss_fn = self._setup_loss()
        
        # Initialize EMA if enabled
        self.ema = None
        if self.config['training'].get('use_ema', False):
            ema_decay = self.config['training'].get('ema_decay', 0.9999)
            self.ema = EMA(self.model, decay=ema_decay)
            logger.info(f"✓ EMA enabled with decay={ema_decay}")
        
        # Setup W&B logging
        self._setup_logging()
        
        # Print training summary
        self._print_summary()
        
        # Sample fixed validation images for visualization
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
    
    def _load_config(self, path: str) -> Dict[str, Any]:
        """Load and validate YAML configuration."""
        path = Path(path).expanduser()
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")
        
        with open(path, 'r') as f:
            config = yaml.safe_load(f)
        
        logger.info(f"✓ Configuration loaded from: {path}")
        return config
    
    def _setup_system(self):
        """Setup device, seeds, directories, and AMP."""
        sys_cfg = self.config['system']
        
        # Device
        device_str = sys_cfg.get('device', 'auto')
        if device_str == 'auto':
            if torch.cuda.is_available():
                self.device = torch.device('cuda')
                logger.info(f"✓ Using CUDA: {torch.cuda.get_device_name(0)}")
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                self.device = torch.device('mps')
                logger.info("✓ Using Apple Silicon MPS")
            else:
                self.device = torch.device('cpu')
                logger.warning("⚠️  Using CPU (training will be slow)")
        else:
            self.device = torch.device(device_str)
            logger.info(f"✓ Using device: {self.device}")
        
        # Random seed
        seed = sys_cfg.get('seed', 42)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        logger.info(f"✓ Random seed set to {seed}")
        
        # Directories
        self.checkpoint_dir = Path(self.config['checkpoint']['dir']).expanduser()
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"✓ Checkpoint directory: {self.checkpoint_dir}")
        
        # FIXED: Cross-platform AMP setup
        self.use_amp = sys_cfg.get('mixed_precision', False)
        
        # GradScaler only works on CUDA
        if self.use_amp and self.device.type == 'cuda':
            self.scaler = GradScaler(enabled=True)
            logger.info("✓ Mixed precision (AMP) enabled with GradScaler")
        elif self.use_amp:
            # MPS/CPU: AMP works but no GradScaler
            self.scaler = None
            logger.info(f"✓ Mixed precision (AMP) enabled for {self.device.type} (no GradScaler)")
        else:
            self.scaler = None
            logger.info("  Mixed precision disabled")
    
    def _setup_model(self) -> nn.Module:
        """Initialize model based on config."""
        model_cfg = self.config['model']
        size = model_cfg['size']
        crf_range = tuple(self.config['dataset']['crf_range'])
        preset_range = tuple(self.config['dataset']['preset_range'])
        
        logger.info(f"Creating model: size={size}")
        model = create_av1_restorer(
            size=size,
            crf_range=crf_range,
            preset_range=preset_range
        )
        
        model = model.to(self.device)
        
        # Log parameter count
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"✓ Model created")
        logger.info(f"  Total parameters: {total_params:,} ({total_params/1e6:.2f}M)")
        logger.info(f"  Trainable: {trainable_params:,} ({trainable_params/1e6:.2f}M)")
        
        return model
    
    def _setup_loss(self) -> nn.Module:
        """Initialize loss function from config."""
        logger.info("Initializing loss function")
        
        # CRITICAL: Pass norm_range from dataset config to loss function
        norm_range = tuple(self.config['dataset'].get('norm_range', [-1, 1]))
        loss_fn = CombinedLoss(self.config['loss'], norm_range=norm_range)
        
        return loss_fn.to(self.device)
    
    def _setup_logging(self):
        """Initialize W&B logging if enabled."""
        if not (self.config['project']['log_to_wandb'] and WANDB_AVAILABLE):
            if self.config['project']['log_to_wandb']:
                logger.warning("⚠️  W&B logging requested but not available")
            return
        
        # Determine run ID
        run_id = self.wandb_id
        if self.resume_from and not run_id:
            try:
                ckpt_path = self._get_checkpoint_path(self.resume_from)
                if ckpt_path.exists():
                    ckpt = torch.load(ckpt_path, map_location='cpu')
                    run_id = ckpt.get('wandb_id')
                    if run_id:
                        logger.info(f"Found W&B run ID in checkpoint: {run_id}")
            except Exception as e:
                logger.warning(f"Could not read W&B ID from checkpoint: {e}")
        
        # Initialize W&B
        project = self.config['project']['name']
        name = self.config['project'].get('experiment_name', f"av1_restorer_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        
        if run_id:
            logger.info(f"Resuming W&B run: {run_id}")
            wandb.init(
                project=project,
                name=name,
                config=self.config,
                id=run_id,
                resume="must"
            )
        else:
            logger.info("Starting new W&B run")
            wandb.init(
                project=project,
                name=name,
                config=self.config,
                tags=['av1_restorer', 'unet', 'artifact_removal']
            )
        
        self.wandb_id = wandb.run.id
        logger.info(f"✓ W&B logging enabled")
        logger.info(f"  Run ID: {self.wandb_id}")
        logger.info(f"  URL: {wandb.run.url}")
    
    def _print_summary(self):
        """Print comprehensive training summary."""
        logger.info("=" * 80)
        logger.info(f"{'AV1 U-Net Restorer - Training Configuration':^80}")
        logger.info("=" * 80)
        
        # Project info
        proj = self.config['project']
        logger.info(f"Project: {proj['name']}")
        logger.info(f"Experiment: {proj.get('experiment_name', 'N/A')}")
        logger.info("")
        
        # Model info
        model_cfg = self.config['model']
        logger.info(f"Model: AV1UNetRestorer ({model_cfg['size']})")
        total_params = sum(p.numel() for p in self.model.parameters())
        logger.info(f"Parameters: {total_params:,} ({total_params/1e6:.2f}M)")
        logger.info("")
        
        # Dataset info
        data_cfg = self.config['data']
        dset_cfg = self.config['dataset']
        logger.info(f"Training Data:")
        logger.info(f"  LQ: {data_cfg['train_lq_root']}")
        logger.info(f"  HQ: {data_cfg['train_hq_root']}")
        logger.info(f"  CRF Range: {dset_cfg['crf_range']}")
        logger.info(f"  Preset Range: {dset_cfg['preset_range']}")
        logger.info("")
        
        # Curriculum stages
        curr = self.config['curriculum']
        logger.info(f"Curriculum Stages: {len(curr['patch_sizes'])}")
        for i in range(len(curr['patch_sizes'])):
            logger.info(
                f"  Stage {i+1}: {curr['epochs_per_stage'][i]} epochs, "
                f"{curr['patch_sizes'][i]}×{curr['patch_sizes'][i]} patches, "
                f"batch_size={curr['batch_sizes'][i]}"
            )
        logger.info("")
        
        # Loss functions
        logger.info("Loss Functions:")
        for name, cfg in self.config['loss'].items():
            if cfg.get('enabled', False):
                logger.info(f"  {name}: weight={cfg['weight']}")
        logger.info("")
        
        # Training config
        train_cfg = self.config['training']
        logger.info(f"Training:")
        logger.info(f"  Device: {self.device}")
        logger.info(f"  Mixed Precision: {self.use_amp}")
        logger.info(f"  EMA: {train_cfg.get('use_ema', False)}")
        logger.info(f"  Grad Clip: {train_cfg.get('grad_clip_norm', 0)}")
        logger.info("")
        
        # Optimizer
        opt_cfg = self.config['optimizer']
        logger.info(f"Optimizer: {opt_cfg['type'].upper()}")
        logger.info(f"  LR: {opt_cfg['lr']}")
        logger.info(f"  Weight Decay: {opt_cfg.get('weight_decay', 0)}")
        
        logger.info("=" * 80)
    
    def _get_fixed_val_samples(self) -> Optional[Dict[str, torch.Tensor]]:
        """Sample fixed validation images for visualization."""
        try:
            data_cfg = self.config['data']
            dset_cfg = self.config['dataset']
            num_samples = self.config['training'].get('num_val_samples_to_log', 4)
            
            # Use largest patch size for validation
            patch_size = self.config['curriculum']['patch_sizes'][-1]
            
            val_dataset = AV1Dataset(
                lq_root_dir=data_cfg['val_lq_root'],
                hq_root_dir=data_cfg['val_hq_root'],
                hq_ext=data_cfg['hq_ext'],
                patch_size=patch_size,
                crf_range=tuple(dset_cfg['crf_range']),
                preset_range=tuple(dset_cfg['preset_range']),
                augment=False,
                norm_range=tuple(dset_cfg.get('norm_range', [-1, 1])),
                return_metadata=True
            )
            
            loader = DataLoader(
                val_dataset,
                batch_size=num_samples,
                shuffle=False,
                num_workers=0
            )
            
            batch = next(iter(loader))
            
            # Move to device
            result = {}
            for key, value in batch.items():
                if isinstance(value, torch.Tensor):
                    result[key] = value.to(self.device)
                else:
                    result[key] = value
            
            logger.info(f"✓ Sampled {num_samples} validation images for visualization")
            return result
            
        except Exception as e:
            logger.warning(f"Could not sample validation images: {e}")
            return None
    
    def _create_dataloaders(
        self,
        patch_size: int,
        batch_size: int
    ) -> Tuple[DataLoader, DataLoader]:
        """Create train and validation dataloaders for a curriculum stage."""
        data_cfg = self.config['data']
        dset_cfg = self.config['dataset']
        sys_cfg = self.config['system']
        
        # Training dataset
        train_dataset = AV1Dataset(
            lq_root_dir=data_cfg['train_lq_root'],
            hq_root_dir=data_cfg['train_hq_root'],
            hq_ext=data_cfg['hq_ext'],
            patch_size=patch_size,
            crf_range=tuple(dset_cfg['crf_range']),
            preset_range=tuple(dset_cfg['preset_range']),
            augment=True,
            norm_range=tuple(dset_cfg.get('norm_range', [-1, 1])),
            return_metadata=False
        )
        
        # Validation dataset
        val_dataset = AV1Dataset(
            lq_root_dir=data_cfg['val_lq_root'],
            hq_root_dir=data_cfg['val_hq_root'],
            hq_ext=data_cfg['hq_ext'],
            patch_size=patch_size,
            crf_range=tuple(dset_cfg['crf_range']),
            preset_range=tuple(dset_cfg['preset_range']),
            augment=False,
            norm_range=tuple(dset_cfg.get('norm_range', [-1, 1])),
            return_metadata=False
        )
        
        # FIXED: pin_memory only on CUDA
        pin_memory = (self.device.type == 'cuda')
        
        # Dataloaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=sys_cfg.get('num_workers', 4),
            pin_memory=pin_memory,
            drop_last=True,
            persistent_workers=sys_cfg.get('num_workers', 4) > 0,
            prefetch_factor=4,
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=min(batch_size * 2, 32),
            shuffle=False,
            num_workers=sys_cfg.get('num_workers', 4),
            pin_memory=pin_memory,
            drop_last=False,
            persistent_workers=sys_cfg.get('num_workers', 4) > 0,
            prefetch_factor=2
        )
        
        logger.info(f"✓ Dataloaders created")
        logger.info(f"  Train: {len(train_dataset):,} samples, {len(train_loader)} batches")
        logger.info(f"  Val: {len(val_dataset):,} samples, {len(val_loader)} batches")
        
        return train_loader, val_loader
    
    def _setup_optimizer(self, total_steps: int) -> Tuple[torch.optim.Optimizer, Optional[Any]]:
        """Create optimizer and learning rate scheduler."""
        opt_cfg = self.config['optimizer']
        sched_cfg = self.config['scheduler']
        
        # Optimizer
        optimizer_type = opt_cfg['type'].lower()
        lr = opt_cfg['lr']
        
        if optimizer_type == 'adamw':
            optimizer = AdamW(
                self.model.parameters(),
                lr=lr,
                betas=tuple(opt_cfg.get('betas', [0.9, 0.999])),
                weight_decay=opt_cfg.get('weight_decay', 1e-4)
            )
        elif optimizer_type == 'adam':
            optimizer = Adam(
                self.model.parameters(),
                lr=lr,
                betas=tuple(opt_cfg.get('betas', [0.9, 0.999]))
            )
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_type}")
        
        logger.info(f"✓ Optimizer: {optimizer_type.upper()}, lr={lr}")
        
        # Scheduler
        scheduler = None
        scheduler_type = sched_cfg.get('type', 'cosine').lower()
        
        if scheduler_type == 'cosine':
            scheduler = CosineAnnealingLR(
                optimizer,
                T_max=total_steps,
                eta_min=sched_cfg.get('min_lr', 1e-6)
            )
            logger.info(f"✓ Scheduler: CosineAnnealingLR, T_max={total_steps}")
        elif scheduler_type == 'onecycle':
            scheduler = OneCycleLR(
                optimizer,
                max_lr=lr,
                total_steps=total_steps,
                pct_start=0.3
            )
            logger.info(f"✓ Scheduler: OneCycleLR, total_steps={total_steps}")
        elif scheduler_type == 'none':
            logger.info("  No scheduler")
        else:
            logger.warning(f"Unknown scheduler type: {scheduler_type}, using none")
        
        return optimizer, scheduler
    
    def train(self):
        """Main training loop with curriculum learning."""
        logger.info("\n" + "=" * 80)
        logger.info("STARTING TRAINING")
        logger.info("=" * 80 + "\n")
        
        curr = self.config['curriculum']
        patch_sizes = curr['patch_sizes']
        batch_sizes = curr['batch_sizes']
        epochs_per_stage = curr['epochs_per_stage']
        
        # Validate curriculum configuration
        if not (len(patch_sizes) == len(batch_sizes) == len(epochs_per_stage)):
            raise ValueError(
                "Curriculum patch_sizes, batch_sizes, and epochs_per_stage must have same length"
            )
        
        # Main curriculum loop
        for stage_idx in range(self.start_stage, len(patch_sizes)):
            patch_size = patch_sizes[stage_idx]
            batch_size = batch_sizes[stage_idx]
            epochs = epochs_per_stage[stage_idx]
            
            logger.info("\n" + "=" * 80)
            logger.info(f"CURRICULUM STAGE {stage_idx + 1}/{len(patch_sizes)}")
            logger.info(f"  Patch Size: {patch_size}×{patch_size}")
            logger.info(f"  Batch Size: {batch_size}")
            logger.info(f"  Epochs: {epochs}")
            logger.info("=" * 80 + "\n")
            
            # Create dataloaders for this stage
            train_loader, val_loader = self._create_dataloaders(patch_size, batch_size)
            
            # Create optimizer and scheduler
            total_steps = epochs * len(train_loader)
            optimizer, scheduler = self._setup_optimizer(total_steps)
            
            # Restore optimizer/scheduler state if resuming this stage
            if stage_idx == self.start_stage and self.saved_optimizer_state:
                optimizer.load_state_dict(self.saved_optimizer_state)
                logger.info("✓ Optimizer state restored")
                
                if scheduler and self.saved_scheduler_state:
                    scheduler.load_state_dict(self.saved_scheduler_state)
                    logger.info("✓ Scheduler state restored")
                
                # Clear saved states
                self.saved_optimizer_state = None
                self.saved_scheduler_state = None
            
            # Training loop for this stage
            for epoch in range(self.start_epoch, epochs):
                # Train one epoch
                self._train_epoch(
                    epoch=epoch,
                    total_epochs=epochs,
                    stage_idx=stage_idx,
                    train_loader=train_loader,
                    optimizer=optimizer,
                    scheduler=scheduler
                )
                
                # Validation
                val_freq = self.config['training'].get('validate_every_n_epochs', 5)
                if (epoch + 1) % val_freq == 0:
                    val_loss = self._validate(epoch, stage_idx, val_loader)
                    
                    # Save best model
                    if val_loss < self.best_val_loss:
                        self.best_val_loss = val_loss
                        self._save_checkpoint(
                            epoch, stage_idx, optimizer, scheduler, is_best=True
                        )
                
                # Periodic checkpoint
                save_freq = self.config['checkpoint'].get('save_every_n_epochs', 10)
                if (epoch + 1) % save_freq == 0:
                    self._save_checkpoint(epoch, stage_idx, optimizer, scheduler)
            
            # Save stage completion checkpoint
            self._save_checkpoint(
                epochs - 1, stage_idx, optimizer, scheduler, is_stage_complete=True
            )
            
            # Reset start_epoch for next stage
            self.start_epoch = 0
        
        # Training complete
        logger.info("\n" + "=" * 80)
        logger.info("TRAINING COMPLETE!")
        logger.info(f"  Best validation loss: {self.best_val_loss:.6f}")
        logger.info(f"  Total steps: {self.global_step:,}")
        logger.info("=" * 80 + "\n")
        
        # Close W&B
        if WANDB_AVAILABLE and self.config['project']['log_to_wandb']:
            wandb.finish()
    
    def _train_epoch(
        self,
        epoch: int,
        total_epochs: int,
        stage_idx: int,
        train_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[Any]
    ):
        """Train for one epoch."""
        self.model.train()
        
        # Progress bar
        pbar = tqdm(
            train_loader,
            desc=f"Stage {stage_idx+1} | Epoch {epoch+1}/{total_epochs}",
            leave=True
        )
        
        # Accumulators
        epoch_loss = 0.0
        epoch_loss_components = {}
        
        for batch_idx, batch in enumerate(pbar):
            # Move data to device
            lq = batch['lq'].to(self.device, non_blocking=True)
            hq = batch['hq'].to(self.device, non_blocking=True)
            crf = batch['crf'].to(self.device, non_blocking=True)
            preset = batch['preset'].to(self.device, non_blocking=True)
            
            # FIXED: Cross-platform autocast
            # Use device type string ('cuda', 'mps', 'cpu')
            with autocast(device_type=str(self.device.type), enabled=self.use_amp):
                restored = self.model(lq, crf, preset)
                loss, loss_dict = self.loss_fn(restored, hq)
            
            # Backward pass with optional GradScaler
            optimizer.zero_grad(set_to_none=True)
            
            if self.scaler is not None:
                # CUDA path: use GradScaler
                self.scaler.scale(loss).backward()
                
                # Gradient clipping
                grad_clip = self.config['training'].get('grad_clip_norm', 0)
                if grad_clip > 0:
                    self.scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip)
                
                self.scaler.step(optimizer)
                self.scaler.update()
            else:
                # MPS/CPU path: standard backward
                loss.backward()
                
                # Gradient clipping
                grad_clip = self.config['training'].get('grad_clip_norm', 0)
                if grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip)
                
                optimizer.step()
            
            # Update scheduler
            if scheduler:
                scheduler.step()
            
            # Update EMA
            if self.ema:
                self.ema.update()
            
            # Accumulate metrics
            epoch_loss += loss.item()
            for key, value in loss_dict.items():
                if key not in epoch_loss_components:
                    epoch_loss_components[key] = 0.0
                epoch_loss_components[key] += value.item()
            
            # Update progress bar
            current_lr = optimizer.param_groups[0]['lr']
            pbar_dict = {
                'loss': f"{loss.item():.4f}",
                'lr': f"{current_lr:.2e}"
            }
            pbar.set_postfix(pbar_dict)
            
            # Log to W&B
            if WANDB_AVAILABLE and self.config['project']['log_to_wandb']:
                if self.global_step % 50 == 0:
                    wandb_dict = {
                        'train/total_loss': loss.item(),
                        'train/learning_rate': current_lr,
                        'train/epoch': epoch,
                        'train/stage': stage_idx,
                        'step': self.global_step
                    }
                    for key, value in loss_dict.items():
                        wandb_dict[f'train/{key}'] = value.item()
                    
                    wandb.log(wandb_dict)
            
            self.global_step += 1
        
        # Log epoch summary
        avg_loss = epoch_loss / len(train_loader)
        logger.info(
            f"Epoch {epoch+1}/{total_epochs} | "
            f"Stage {stage_idx+1} | "
            f"Avg Loss: {avg_loss:.6f}"
        )
    
    # @torch.no_grad()
    # def _validate(
    #     self,
    #     epoch: int,
    #     stage_idx: int,
    #     val_loader: DataLoader
    # ) -> float:
    #     """Run validation and return average loss."""
    #     self.model.eval()
        
    #     # Apply EMA weights if enabled
    #     if self.ema:
    #         self.ema.apply_shadow()
        
    #     # Accumulators
    #     total_loss = 0.0
    #     total_loss_components = {}
        
    #     # Validation loop
    #     pbar = tqdm(val_loader, desc=f"Validating Epoch {epoch+1}", leave=False)
    #     for batch in pbar:
    #         lq = batch['lq'].to(self.device, non_blocking=True)
    #         hq = batch['hq'].to(self.device, non_blocking=True)
    #         crf = batch['crf'].to(self.device, non_blocking=True)
    #         preset = batch['preset'].to(self.device, non_blocking=True)
            
    #         # Forward pass
    #         restored = self.model(lq, crf, preset)
    #         loss, loss_dict = self.loss_fn(restored, hq)
            
    #         total_loss += loss.item()
    #         for key, value in loss_dict.items():
    #             if key not in total_loss_components:
    #                 total_loss_components[key] = 0.0
    #             total_loss_components[key] += value.item()
        
    #     # Calculate averages
    #     avg_loss = total_loss / len(val_loader)
    #     avg_components = {k: v / len(val_loader) for k, v in total_loss_components.items()}
        
    #     # Log validation results
    #     logger.info(
    #         f"Validation Epoch {epoch+1} | Stage {stage_idx+1} | "
    #         f"Avg Loss: {avg_loss:.6f}"
    #     )
        
    #     # Log to W&B
    #     if WANDB_AVAILABLE and self.config['project']['log_to_wandb']:
    #         wandb_dict = {
    #             'val/total_loss': avg_loss,
    #             'val/epoch': epoch,
    #             'val/stage': stage_idx,
    #             'step': self.global_step
    #         }
    #         for key, value in avg_components.items():
    #             wandb_dict[f'val/{key}'] = value
            
    #         # Add visual samples if available
    #         if self.val_samples is not None:
    #             try:
    #                 self._log_visual_samples(epoch, stage_idx)
    #             except Exception as e:
    #                 logger.warning(f"Failed to log visual samples: {e}")
            
    #         wandb.log(wandb_dict)
        
    #     # Restore original weights if EMA applied
    #     if self.ema:
    #         self.ema.restore()
        
    #     return avg_loss
    @torch.no_grad()
    def _validate(
        self,
        epoch: int,
        stage_idx: int,
        val_loader: DataLoader
    ) -> float:
        """Run validation and return average loss."""
        self.model.eval()
        
        # Apply EMA weights if enabled
        if self.ema:
            self.ema.apply_shadow()
        
        # Accumulators
        total_loss = 0.0
        total_loss_components = {}
        baseline_l1_sum = 0.0
        baseline_l2_sum = 0.0
        restored_l1_sum = 0.0
        restored_l2_sum = 0.0
        
        # Validation loop
        pbar = tqdm(val_loader, desc=f"Validating Epoch {epoch+1}", leave=False)
        for batch in pbar:
            lq = batch['lq'].to(self.device, non_blocking=True)
            hq = batch['hq'].to(self.device, non_blocking=True)
            crf = batch['crf'].to(self.device, non_blocking=True)
            preset = batch['preset'].to(self.device, non_blocking=True)
            
            # Forward pass
            restored = self.model(lq, crf, preset)
            loss, loss_dict = self.loss_fn(restored, hq)
            
            # Accumulate main loss
            total_loss += loss.item()
            for key, value in loss_dict.items():
                if key not in total_loss_components:
                    total_loss_components[key] = 0.0
                total_loss_components[key] += value.item()
            
            # Compute baseline metrics (LQ vs HQ - "do nothing")
            baseline_l1_sum += F.l1_loss(lq, hq).item()
            baseline_l2_sum += F.mse_loss(lq, hq).item()
            
            # Compute restored metrics (Restored vs HQ)
            restored_l1_sum += F.l1_loss(restored, hq).item()
            restored_l2_sum += F.mse_loss(restored, hq).item()
        
        # Calculate averages
        num_batches = len(val_loader)
        avg_loss = total_loss / num_batches
        avg_components = {k: v / num_batches for k, v in total_loss_components.items()}
        
        # Calculate metric averages
        avg_baseline_l1 = baseline_l1_sum / num_batches
        avg_baseline_l2 = baseline_l2_sum / num_batches
        avg_restored_l1 = restored_l1_sum / num_batches
        avg_restored_l2 = restored_l2_sum / num_batches
        
        # Calculate improvement percentages
        l1_improvement = ((avg_baseline_l1 - avg_restored_l1) / avg_baseline_l1) * 100 if avg_baseline_l1 > 0 else 0
        l2_improvement = ((avg_baseline_l2 - avg_restored_l2) / avg_baseline_l2) * 100 if avg_baseline_l2 > 0 else 0
        
        # Log validation results
        logger.info(
            f"Validation Epoch {epoch+1} | Stage {stage_idx+1} | "
            f"Avg Loss: {avg_loss:.6f}"
        )
        logger.info(
            f"  Baseline L1: {avg_baseline_l1:.6f} | Restored L1: {avg_restored_l1:.6f} "
            f"(↓{l1_improvement:.2f}%)"
        )
        logger.info(
            f"  Baseline L2: {avg_baseline_l2:.6f} | Restored L2: {avg_restored_l2:.6f} "
            f"(↓{l2_improvement:.2f}%)"
        )
        
        # Log to W&B
        if WANDB_AVAILABLE and self.config['project']['log_to_wandb']:
            wandb_dict = {
                'val/total_loss': avg_loss,
                'val/epoch': epoch,
                'val/stage': stage_idx,
                'step': self.global_step,
                # Baseline metrics (do nothing - LQ vs HQ)
                'val/baseline_l1': avg_baseline_l1,
                'val/baseline_l2': avg_baseline_l2,
                # Restored metrics (model output vs HQ)
                'val/restored_l1': avg_restored_l1,
                'val/restored_l2': avg_restored_l2,
                # Improvement metrics
                'val/l1_improvement_pct': l1_improvement,
                'val/l2_improvement_pct': l2_improvement,
            }
            for key, value in avg_components.items():
                wandb_dict[f'val/{key}'] = value
            
            # Add visual samples if available
            if self.val_samples is not None:
                try:
                    self._log_visual_samples(epoch, stage_idx)
                except Exception as e:
                    logger.warning(f"Failed to log visual samples: {e}")
            
            wandb.log(wandb_dict)
        
        # Restore original weights if EMA applied
        if self.ema:
            self.ema.restore()
        
        return avg_loss
    
    def _log_visual_samples(self, epoch: int, stage_idx: int):
        """Generate and log visual samples to W&B."""
        if self.val_samples is None:
            return
        
        lq = self.val_samples['lq']
        hq = self.val_samples['hq']
        crf = self.val_samples['crf']
        preset = self.val_samples['preset']
        
        # Generate restorations
        with torch.no_grad():
            restored = self.model(lq, crf, preset)
        
        # Create comparison grid: LQ | Restored | HQ
        # Handle normalization range (convert to [0, 1] for display)
        norm_range=tuple(self.config['dataset'].get('norm_range', [-1, 1]))

        if norm_range == (-1,1):
            # Convert from [-1, 1] to [0, 1]
            lq_display = (lq + 1.0) / 2.0
            restored_display = (restored + 1.0) / 2.0
            hq_display = (hq + 1.0) / 2.0
        else:
            lq_display = lq
            restored_display = restored
            hq_display = hq
        
        # Clamp to valid range
        lq_display = torch.clamp(lq_display, 0, 1)
        restored_display = torch.clamp(restored_display, 0, 1)
        hq_display = torch.clamp(hq_display, 0, 1)
        
        # Create grid for each sample
        images = []
        for i in range(lq.size(0)):
            # Concatenate horizontally: LQ | Restored | HQ
            row = torch.cat([lq_display[i], restored_display[i], hq_display[i]], dim=2)
            images.append(row)
        
        # Stack all samples vertically
        grid = torch.cat(images, dim=1)  # Stack along height
        
        # Convert to numpy for wandb
        grid_np = grid.cpu().permute(1, 2, 0).numpy()
        
        # Log to wandb
        wandb.log({
            f'val/visual_samples': wandb.Image(
                grid_np,
                caption=f"Stage {stage_idx+1}, Epoch {epoch+1} | LQ - Restored - HQ"
            )
        })
    
    # --------------------------------------------------------------------------
    # Checkpointing
    # --------------------------------------------------------------------------
    
    def _get_checkpoint_path(self, key: str) -> Path:
        """Resolve checkpoint key to path."""
        cd = self.checkpoint_dir
        
        # If already a valid path
        key_path = Path(key)
        if key_path.exists():
            return key_path
        
        # Standard keys
        if key == 'latest':
            return cd / 'latest.pth'
        if key == 'best':
            return cd / 'best.pth'
        
        # Assume filename in checkpoint dir
        return cd / key
    
    def _save_checkpoint(
        self,
        epoch: int,
        stage_idx: int,
        optimizer: Optional[torch.optim.Optimizer],
        scheduler: Optional[Any],
        is_best: bool = False,
        is_stage_complete: bool = False
    ):
        """Save checkpoint with full training state."""
        ckpt = {
            'config': self.config,
            'model_state': self.model.state_dict(),
            'epoch': epoch,
            'stage_idx': stage_idx,
            'global_step': self.global_step,
            'best_val_loss': self.best_val_loss,
            'wandb_id': self.wandb_id
        }
        
        if optimizer is not None:
            ckpt['optimizer_state'] = optimizer.state_dict()
        if scheduler is not None:
            ckpt['scheduler_state'] = scheduler.state_dict()
        if self.ema is not None:
            ckpt['ema_shadow'] = self.ema.shadow
        
        # Save latest
        latest_path = self.checkpoint_dir / 'latest.pth'
        torch.save(ckpt, latest_path)
        logger.info(f"✓ Saved checkpoint: {latest_path}")
        
        # Save best
        if is_best:
            best_path = self.checkpoint_dir / 'best.pth'
            torch.save(ckpt, best_path)
            logger.info(f"✓ Saved best checkpoint (val_loss={self.best_val_loss:.6f}): {best_path}")
        
        # Save stage complete
        if is_stage_complete:
            complete_path = self.checkpoint_dir / f'stage{stage_idx+1:02d}_complete.pth'
            torch.save(ckpt, complete_path)
            logger.info(f"✓ Saved stage-complete checkpoint: {complete_path}")
    
    def _load_checkpoint(self):
        """Load checkpoint and restore training state."""
        try:
            ckpt_path = self._get_checkpoint_path(self.resume_from)
            if not ckpt_path.exists():
                raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
            
            logger.info(f"Loading checkpoint: {ckpt_path}")
            ckpt = torch.load(ckpt_path, map_location=self.device)
            
            # Restore model
            self.model.load_state_dict(ckpt['model_state'])
            logger.info("✓ Model state restored")
            
            # Restore training state
            self.start_stage = ckpt.get('stage_idx', 0)
            self.start_epoch = ckpt.get('epoch', 0) + 1  # Resume from next epoch
            self.global_step = ckpt.get('global_step', 0)
            self.best_val_loss = ckpt.get('best_val_loss', float('inf'))
            
            # Restore W&B ID
            if 'wandb_id' in ckpt and not self.wandb_id:
                self.wandb_id = ckpt['wandb_id']
            
            # Save optimizer/scheduler states for later restoration
            self.saved_optimizer_state = ckpt.get('optimizer_state')
            self.saved_scheduler_state = ckpt.get('scheduler_state')
            
            # Restore EMA
            if 'ema_shadow' in ckpt and self.ema is not None:
                self.ema.shadow = ckpt['ema_shadow']
                logger.info("✓ EMA shadow restored")
            
            logger.info(
                f"✓ Checkpoint loaded: stage={self.start_stage}, epoch={self.start_epoch}, "
                f"step={self.global_step}, best_loss={self.best_val_loss:.6f}"
            )
        
        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")
            logger.warning("Starting training from scratch")


# ==============================================================================
# SECTION 3: Entry Point
# ==============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Train AV1 U-Net Restorer",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '-c', '--config',
        type=str,
        required=True,
        help='Path to YAML configuration file'
    )
    parser.add_argument(
        '-r', '--resume',
        type=str,
        default=None,
        help='Checkpoint to resume from (latest, best, or path)'
    )
    parser.add_argument(
        '--wandb_id',
        type=str,
        default=None,
        help='W&B run ID to resume logging'
    )
    return parser.parse_args()


def main():
    args = parse_args()
    
    try:
        # Initialize trainer
        trainer = AV1RestorerTrainer(
            config_path=args.config,
            resume_from=args.resume,
            wandb_id=args.wandb_id
        )
        
        # Start training
        trainer.train()
        
    except KeyboardInterrupt:
        logger.warning("\n⚠️  Training interrupted by user")
        logger.info("Saving emergency checkpoint...")
        
        try:
            if hasattr(trainer, 'model') and hasattr(trainer, 'checkpoint_dir'):
                emergency_ckpt = {
                    'model_state': trainer.model.state_dict(),
                    'epoch': trainer.start_epoch,
                    'stage_idx': trainer.start_stage,
                    'global_step': trainer.global_step,
                    'best_val_loss': trainer.best_val_loss,
                    'wandb_id': trainer.wandb_id
                }
                if trainer.ema:
                    emergency_ckpt['ema_shadow'] = trainer.ema.shadow
                
                emergency_path = trainer.checkpoint_dir / 'emergency_interrupt.pth'
                torch.save(emergency_ckpt, emergency_path)
                logger.info(f"✓ Emergency checkpoint saved: {emergency_path}")
        except Exception as e:
            logger.error(f"Failed to save emergency checkpoint: {e}")
        
        # Close W&B
        if WANDB_AVAILABLE and trainer.config['project'].get('log_to_wandb', False):
            wandb.finish()
        
        raise
    
    except Exception as e:
        logger.error(f"❌ Training failed with error: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()