"""
Phase 1 Training: AV1 Prior Embedder with Dual Learning

This script trains the Compression-aware Prior Embedder (CaPE) using a curriculum
learning strategy with dual objectives:
1. Explicit Learning: CRF prediction to distinguish compression levels
2. Implicit Learning: Image reconstruction to understand artifacts

Training Features:
- Curriculum learning: progressively larger patches
- Multi-stage training with different batch sizes
- Comprehensive logging and checkpointing
- Weights & Biases integration
- Resume from checkpoint capability
- Validation during training

Author: Soham Mukherjee
"""

import os
import sys
import logging
import argparse
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime

import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, OneCycleLR
from tqdm import tqdm

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from core.dataset import AV1Dataset
from core.av1_prior_embedder import AV1PriorEmbedder
from core.losses import DualLearningLoss

# Optional: Weights & Biases for experiment tracking
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("Warning: wandb not available. Install with: pip install wandb")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(name)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

class CurriculumTrainer:
    """
    Trainer for Phase 1 with curriculum learning strategy.

    Implements multi-stage training where the model progressively learns from:
    Stage 1: Small patches (128x128) with larger batch size → Learn local patterns
    Stage 2: Medium patches (256x256) with medium batch size → Learn broader context
    Stage 3 (optional): Large patches (512x512) with small batch size → Refine on full context

    Args:
        config: Training configuration dictionary
        device: Device to train on ('cuda', 'mps', or 'cpu')
        resume_from: Optional checkpoint path to resume training
    """

    def __init__(
        self,
        config: Dict[str, Any],
        device: torch.device,
        resume_from: Optional[str] = None
    ):
        self.config = config
        self.device = device
        self.resume_from = resume_from

        # Extract configuration sections
        self.project_cfg = config
        self.phase1_cfg = config['phase1']

        # Setup directories
        self.checkpoint_dir = Path(self.phase1_cfg['checkpoint_save_path'])
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Initialize components
        self.model = self._build_model()
        self.criterion = self._build_criterion()

        # Training state
        self.global_step = 0
        self.best_val_loss = float('inf')
        self.start_stage = 0
        self.start_epoch = 0

        # Initialize logging
        self._init_logging()

        # Resume from checkpoint if specified
        if resume_from:
            self._load_checkpoint(resume_from)

        logger.info("CurriculumTrainer initialized successfully")

    def _build_model(self) -> nn.Module:
        """Build the AV1 Prior Embedder model."""
        model = AV1PriorEmbedder(
            embed_dim=self.phase1_cfg.get('embed_dim', 256),
            swin_model_name=self.phase1_cfg.get('swin_model_name', 'swin_tiny_patch4_window7_224'),
            use_pretrained_swin=self.phase1_cfg.get('use_pretrained_swin', True),
            freeze_swin_layers=self.phase1_cfg.get('freeze_swin_layers', 0)
        )

        model = model.to(self.device)

        # Log parameter count
        param_counts = model.get_num_params()
        logger.info(f"Model parameters:")
        for name, count in param_counts.items():
            logger.info(f"  {name}: {count:,}")

        return model

    def _build_criterion(self) -> nn.Module:
        """Build the loss function."""
        return DualLearningLoss(
            reconstruction_weight=self.phase1_cfg.get('reconstruction_loss_weight', 1.0),
            crf_weight=self.phase1_cfg.get('crf_prediction_loss_weight', 0.1),
            use_perceptual=self.phase1_cfg.get('use_perceptual_loss', False)
        )

    def _init_logging(self):
        """Initialize Weights & Biases logging if enabled."""
        if self.project_cfg.get('log_to_wandb', False) and WANDB_AVAILABLE:
            wandb.init(
                project=self.project_cfg['project_name'],
                config=self.config,
                name=f"phase1_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                tags=['phase1', 'prior_embedder', 'dual_learning']
            )
            logger.info("Weights & Biases logging initialized")
        elif self.project_cfg.get('log_to_wandb', False):
            logger.warning("wandb logging requested but not available")

    def _create_dataloader(self, patch_size: int, batch_size: int, is_train: bool = True) -> DataLoader:
        """
        Create dataloader for a specific curriculum stage.

        Args:
            patch_size: Size of image patches
            batch_size: Batch size for this stage
            is_train: Whether this is for training or validation

        Returns:
            DataLoader instance
        """
        dataset = AV1Dataset(
            lq_root_dir=self.project_cfg['lq_data_root'],
            hq_root_dir=self.project_cfg['hq_data_root'],
            hq_ext=self.project_cfg['hq_file_extension'],
            patch_size=patch_size,
            crf_range=tuple(self.project_cfg['crf_range']),
            preset_range=tuple(self.project_cfg['preset_range']),
            augment=is_train,
            return_metadata=False
        )

        # Log dataset statistics
        if is_train:
            logger.info(f"{'='*60}")
            logger.info(f"Dataset Statistics (patch_size={patch_size}):")
            logger.info(f"  Total samples: {len(dataset)}")
            logger.info(f"  CRF distribution: {dataset.get_crf_distribution()}")
            logger.info(f"  Preset distribution: {dataset.get_preset_distribution()}")
            logger.info(f"{'='*60}")

        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=is_train,
            num_workers=self.project_cfg.get('num_workers', 4),
            pin_memory=True,
            drop_last=is_train,
            persistent_workers=True if self.project_cfg.get('num_workers', 4) > 0 else False
        )

        return dataloader

    def _create_optimizer(self, lr: float) -> torch.optim.Optimizer:
        """Create optimizer for current stage."""
        optimizer_type = self.phase1_cfg.get('optimizer', 'adamw')

        if optimizer_type.lower() == 'adamw':
            optimizer = AdamW(
                self.model.parameters(),
                lr=lr,
                betas=(0.9, 0.999),
                weight_decay=self.phase1_cfg.get('weight_decay', 1e-4)
            )
        elif optimizer_type.lower() == 'adam':
            optimizer = Adam(
                self.model.parameters(),
                lr=lr,
                betas=(0.9, 0.999)
            )
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_type}")

        logger.info(f"Created optimizer: {optimizer_type} with lr={lr}")
        return optimizer

    def _create_scheduler(self, optimizer: torch.optim.Optimizer, total_steps: int):
        """Create learning rate scheduler."""
        scheduler_type = self.phase1_cfg.get('scheduler', 'cosine')

        if scheduler_type == 'cosine':
            scheduler = CosineAnnealingLR(
                optimizer,
                T_max=total_steps,
                eta_min=self.phase1_cfg.get('min_lr', 1e-6)
            )
        elif scheduler_type == 'onecycle':
            scheduler = OneCycleLR(
                optimizer,
                max_lr=self.phase1_cfg.get('learning_rate', 1e-4),
                total_steps=total_steps,
                pct_start=0.3
            )
        else:
            scheduler = None
            logger.info("No learning rate scheduler used")

        return scheduler

    def train_epoch(
        self,
        dataloader: DataLoader,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[Any],
        epoch: int,
        stage_idx: int,
        patch_size: int
    ) -> Dict[str, float]:
        """
        Train for one epoch.

        Args:
            dataloader: Training dataloader
            optimizer: Optimizer
            scheduler: Learning rate scheduler (optional)
            epoch: Current epoch number
            stage_idx: Current curriculum stage
            patch_size: Current patch size

        Returns:
            Dictionary of average metrics for the epoch
        """
        self.model.train()

        # Metrics accumulation
        total_loss = 0.0
        total_recon_loss = 0.0
        total_crf_loss = 0.0
        num_batches = 0

        pbar = tqdm(
            dataloader,
            desc=f"Stage {stage_idx+1} | Epoch {epoch+1} | Patch {patch_size}",
            leave=True
        )

        for batch_idx, batch in enumerate(pbar):
            # Move data to device
            lq_images = batch['lq'].to(self.device, non_blocking=True)
            hq_images = batch['hq'].to(self.device, non_blocking=True)
            crf_values = batch['crf'].to(self.device, non_blocking=True)

            with torch.autocast(device_type=self.device.type, enabled=self.project_cfg.get('mixed_precision', False)):
                 # Forward pass
                reconstructed_images, predicted_crf, _ = self.model(lq_images)
                loss_dict = self.criterion(
                    reconstructed_images, hq_images,
                    predicted_crf, crf_values
                )
                # Compute loss
                loss = loss_dict['total_loss']

            # MODIFY THESE 3 LINES: Use the scaler for the backward pass
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            # Backward pass
            optimizer.zero_grad()
            loss.backward()

            # Gradient clipping
            if self.phase1_cfg.get('grad_clip', 0) > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.phase1_cfg['grad_clip']
                )

            if scheduler is not None:
                scheduler.step()

            # Accumulate metrics
            total_loss += loss.item()
            total_recon_loss += loss_dict['reconstruction_loss'].item()
            total_crf_loss += loss_dict['crf_loss'].item()
            num_batches += 1

            # Update progress bar
            current_lr = optimizer.param_groups[0]['lr']
            pbar.set_postfix({
                'Loss': f"{loss.item():.4f}",
                'Recon': f"{loss_dict['reconstruction_loss'].item():.4f}",
                'CRF': f"{loss_dict['crf_loss'].item():.4f}",
                'LR': f"{current_lr:.2e}"
            })

            # Log to wandb
            if self.project_cfg.get('log_to_wandb', False) and WANDB_AVAILABLE:
                wandb.log({
                    'train/total_loss': loss.item(),
                    'train/reconstruction_loss': loss_dict['reconstruction_loss'].item(),
                    'train/crf_loss': loss_dict['crf_loss'].item(),
                    'train/learning_rate': current_lr,
                    'train/epoch': epoch,
                    'train/stage': stage_idx,
                    'train/patch_size': patch_size,
                    'step': self.global_step
                })

            self.global_step += 1

        # Calculate epoch averages
        avg_metrics = {
            'total_loss': total_loss / num_batches,
            'reconstruction_loss': total_recon_loss / num_batches,
            'crf_loss': total_crf_loss / num_batches
        }

        return avg_metrics

    @torch.no_grad()
    def validate(self, dataloader: DataLoader, epoch: int, stage_idx: int) -> Dict[str, float]:
        """
        Run validation.

        Args:
            dataloader: Validation dataloader
            epoch: Current epoch
            stage_idx: Current curriculum stage

        Returns:
            Dictionary of validation metrics
        """
        self.model.eval()

        total_loss = 0.0
        total_recon_loss = 0.0
        total_crf_loss = 0.0
        num_batches = 0

        pbar = tqdm(dataloader, desc=f"Validation | Epoch {epoch+1}", leave=False)

        for batch in pbar:
            lq_images = batch['lq'].to(self.device, non_blocking=True)
            hq_images = batch['hq'].to(self.device, non_blocking=True)
            crf_values = batch['crf'].to(self.device, non_blocking=True)

            # Forward pass
            reconstructed_images, predicted_crf, _ = self.model(lq_images)

            # Compute loss
            loss_dict = self.criterion(
                reconstructed_images, hq_images,
                predicted_crf, crf_values
            )

            total_loss += loss_dict['total_loss'].item()
            total_recon_loss += loss_dict['reconstruction_loss'].item()
            total_crf_loss += loss_dict['crf_loss'].item()
            num_batches += 1

        # Calculate averages
        avg_metrics = {
            'val_total_loss': total_loss / num_batches,
            'val_reconstruction_loss': total_recon_loss / num_batches,
            'val_crf_loss': total_crf_loss / num_batches
        }

        logger.info(f"Validation Results - Loss: {avg_metrics['val_total_loss']:.4f}")

        # Log to wandb
        if self.project_cfg.get('log_to_wandb', False) and WANDB_AVAILABLE:
            wandb.log({
                **{f'val/{k}': v for k, v in avg_metrics.items()},
                'val/epoch': epoch,
                'val/stage': stage_idx
            })

        return avg_metrics

    def _save_checkpoint(
        self,
        epoch: int,
        stage_idx: int,
        optimizer: torch.optim.Optimizer,
        is_best: bool = False,
        suffix: str = ''
    ):
        """Save training checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'stage_idx': stage_idx,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_val_loss': self.best_val_loss,
            'config': self.config
        }

        # Save regular checkpoint
        if suffix:
            ckpt_path = self.checkpoint_dir / f"checkpoint_{suffix}.pth"
        else:
            ckpt_path = self.checkpoint_dir / f"checkpoint_stage{stage_idx}_epoch{epoch}.pth"

        torch.save(checkpoint, ckpt_path)
        logger.info(f"Checkpoint saved: {ckpt_path}")

        # Save best model
        if is_best:
            best_path = self.checkpoint_dir / "best_model.pth"
            torch.save(checkpoint, best_path)
            logger.info(f"Best model saved: {best_path}")

    def _load_checkpoint(self, checkpoint_path: str):
        """Load checkpoint to resume training."""
        logger.info(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.global_step = checkpoint['global_step']
        self.best_val_loss = checkpoint['best_val_loss']
        self.start_stage = checkpoint['stage_idx']
        self.start_epoch = checkpoint['epoch'] + 1

        logger.info(f"Resumed from stage {self.start_stage}, epoch {self.start_epoch}")

    def train(self):
        """Main training loop with curriculum learning."""
        logger.info("="*80)
        logger.info("Starting Phase 1 Training: AV1 Prior Embedder")
        logger.info("="*80)

        patch_sizes = self.phase1_cfg['patch_sizes']
        batch_sizes = self.phase1_cfg['batch_sizes']
        epochs_per_stage = self.phase1_cfg['epochs_per_stage']

        # Validate curriculum configuration
        if not (len(patch_sizes) == len(batch_sizes) == len(epochs_per_stage)):
            raise ValueError(
                "patch_sizes, batch_sizes, and epochs_per_stage must have same length"
            )

        # Curriculum learning loop
        for stage_idx in range(self.start_stage, len(patch_sizes)):
            patch_size = patch_sizes[stage_idx]
            batch_size = batch_sizes[stage_idx]
            epochs = epochs_per_stage[stage_idx]

            logger.info("="*80)
            logger.info(f"Curriculum Stage {stage_idx + 1}/{len(patch_sizes)}")
            logger.info(f"  Patch Size: {patch_size}x{patch_size}")
            logger.info(f"  Batch Size: {batch_size}")
            logger.info(f"  Epochs: {epochs}")
            logger.info("="*80)

            # Create dataloaders for this stage
            train_loader = self._create_dataloader(
                patch_size, batch_size, is_train=True
            )

            # Optional validation loader
            val_loader = None
            if self.phase1_cfg.get('validate_every_n_epochs', 0) > 0:
                val_loader = self._create_dataloader(
                    patch_size, 
                    batch_size=min(batch_size * 2, 32),  # Larger batch for validation
                    is_train=False
                )

            # Create optimizer and scheduler for this stage
            lr = self.phase1_cfg.get('learning_rate', 1e-4)
            optimizer = self._create_optimizer(lr)

            total_steps = epochs * len(train_loader)
            scheduler = self._create_scheduler(optimizer, total_steps)

            # Training loop for this stage
            for epoch in range(self.start_epoch, epochs):
                # Train one epoch
                train_metrics = self.train_epoch(
                    train_loader, optimizer, scheduler,
                    epoch, stage_idx, patch_size
                )

                logger.info(
                    f"Stage {stage_idx+1} | Epoch {epoch+1}/{epochs} | "
                    f"Loss: {train_metrics['total_loss']:.4f} | "
                    f"Recon: {train_metrics['reconstruction_loss']:.4f} | "
                    f"CRF: {train_metrics['crf_loss']:.4f}"
                )

                # Validation
                if val_loader and (epoch + 1) % self.phase1_cfg.get('validate_every_n_epochs', 5) == 0:
                    val_metrics = self.validate(val_loader, epoch, stage_idx)

                    # Save best model
                    if val_metrics['val_total_loss'] < self.best_val_loss:
                        self.best_val_loss = val_metrics['val_total_loss']
                        self._save_checkpoint(epoch, stage_idx, optimizer, is_best=True)

                # Save checkpoint periodically
                if (epoch + 1) % self.phase1_cfg.get('save_every_n_epochs', 10) == 0:
                    self._save_checkpoint(epoch, stage_idx, optimizer)

            # Save checkpoint after completing stage
            self._save_checkpoint(epochs - 1, stage_idx, optimizer, suffix=f"stage{stage_idx}_complete")

            # Reset start_epoch for next stage
            self.start_epoch = 0

        # Save final model
        final_path = self.checkpoint_dir / "final_model.pth"
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'config': self.config,
            'global_step': self.global_step
        }, final_path)
        logger.info(f"Final model saved: {final_path}")

        logger.info("="*80)
        logger.info("Phase 1 Training Complete!")
        logger.info(f"Best validation loss: {self.best_val_loss:.4f}")
        logger.info(f"Total training steps: {self.global_step}")
        logger.info("="*80)

        # Close wandb
        if self.project_cfg.get('log_to_wandb', False) and WANDB_AVAILABLE:
            wandb.finish()

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Phase 1: Train AV1 Prior Embedder with Dual Learning",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        '--config',
        type=str,
        default='configs/train_av1_p4.yaml',
        help='Path to training configuration YAML file'
    )

    parser.add_argument(
        '--resume',
        type=str,
        default=None,
        help='Path to checkpoint to resume training from'
    )

    parser.add_argument(
        '--device',
        type=str,
        default=None,
        help='Device to use (cuda, mps, cpu). If None, auto-detect.'
    )

    parser.add_argument(
        '--debug',
        action='store_true',
        help='Enable debug mode (more verbose logging)'
    )

    return parser.parse_args()

def setup_device(device_str: Optional[str] = None) -> torch.device:
    """
    Setup and return the appropriate device for training.

    Args:
        device_str: Explicit device string or None for auto-detection

    Returns:
        torch.device object
    """
    if device_str:
        device = torch.device(device_str)
        logger.info(f"Using explicitly specified device: {device}")
        return device

    # Auto-detect best available device
    if torch.cuda.is_available():
        device = torch.device('cuda')
        logger.info(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
        logger.info(f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device('mps')
        logger.info("Using Apple Silicon MPS device")
    else:
        device = torch.device('cpu')
        logger.warning("Using CPU device - training will be slow!")

    return device

def validate_config(config: Dict[str, Any]):
    """
    Validate training configuration.

    Args:
        config: Configuration dictionary

    Raises:
        ValueError: If configuration is invalid
    """
    required_keys = ['project_name', 'lq_data_root', 'hq_data_root', 'phase1']
    for key in required_keys:
        if key not in config:
            raise ValueError(f"Missing required config key: {key}")

    phase1 = config['phase1']
    required_phase1_keys = ['patch_sizes', 'batch_sizes', 'epochs_per_stage', 'learning_rate']
    for key in required_phase1_keys:
        if key not in phase1:
            raise ValueError(f"Missing required phase1 config key: {key}")

    # Validate curriculum parameters
    if not (len(phase1['patch_sizes']) == len(phase1['batch_sizes']) == len(phase1['epochs_per_stage'])):
        raise ValueError("patch_sizes, batch_sizes, and epochs_per_stage must have same length")

    # Validate data paths
    lq_path = Path(config['lq_data_root']).expanduser()
    hq_path = Path(config['hq_data_root']).expanduser()

    if not lq_path.exists():
        raise FileNotFoundError(f"LQ data directory not found: {lq_path}")
    if not hq_path.exists():
        raise FileNotFoundError(f"HQ data directory not found: {hq_path}")

    logger.info("Configuration validation passed")

def main():
    """Main entry point."""
    args = parse_args()

    # Set debug logging if requested
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.setLevel(logging.DEBUG)

    # Load configuration
    logger.info(f"Loading configuration from: {args.config}")
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # Validate configuration
    validate_config(config)

    # Setup device
    device = setup_device(args.device)

    # Override device in config if specified
    if args.device:
        config['device'] = args.device

    # Create trainer
    trainer = CurriculumTrainer(
        config=config,
        device=device,
        resume_from=args.resume
    )

    # Start training
    try:
        trainer.train()
    except KeyboardInterrupt:
        logger.warning("Training interrupted by user")
        logger.info("Saving emergency checkpoint...")
        trainer._save_checkpoint(
            epoch=0,
            stage_idx=0,
            optimizer=None,
            suffix='emergency'
        )
    except Exception as e:
        logger.error(f"Training failed with error: {e}", exc_info=True)
        raise

if __name__ == "__main__":
    main()