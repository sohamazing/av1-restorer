"""
Phase 1 Training: AV1 Prior Embedder with Dual Learning

Complete checkpoint/resume implementation with improved loss functions.

Key updates:
- Integrated improved DualLearningLoss with CRF normalization
- Support for AdaptiveDualLearningLoss with automatic weighting
- Charbonnier loss for reconstruction
- Better handling of loss components and logging

Author: Soham Mukherjee
"""

import os
import sys
import logging
import argparse
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
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
from core.losses import DualLearningLoss, AdaptiveDualLearningLoss, BaselineRelativeLoss, AdaptiveBaselineRelativeLoss

# Optional: Weights & Biases
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
    Trainer with full checkpoint/resume capability and improved loss functions.
    """

    def __init__(
        self,
        config: Dict[str, Any],
        device: torch.device,
        resume_from: Optional[str] = None,
        wandb_id: Optional[str] = None
    ):
        self.config = config
        self.device = device
        self.resume_from = resume_from
        self.wandb_id = wandb_id

        # Extract configuration
        self.project_cfg = config
        self.phase1_cfg = config['phase1']

        # Setup directories
        self.checkpoint_dir = Path(self.phase1_cfg['checkpoint_save_path'])
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Training state (will be restored if resuming)
        self.global_step = 0
        self.best_val_loss = float('inf')
        self.start_stage = 0
        self.start_epoch = 0

        # Initialize model and criterion
        self.model = self._build_model()
        self.criterion = self._build_criterion()

        # These will be created per curriculum stage
        self.optimizer = None
        self.scheduler = None

        # Initialize logging BEFORE loading checkpoint (to get run ID)
        self._init_logging()

        # Load checkpoint if resuming
        if resume_from:
            self._load_checkpoint(resume_from)

        # Sample validation images for visual logging
        self.validation_samples = self._sample_validation_images(
            patch_size=self.phase1_cfg['patch_sizes'][-1]
        )

        logger.info("CurriculumTrainer initialized successfully")

    def _build_model(self) -> nn.Module:
        """Build the AV1 Prior Embedder model."""
        model = AV1PriorEmbedder(
            embed_dim=self.phase1_cfg.get('embed_dim', 256),
            num_blocks=self.phase1_cfg.get('num_blocks', 2)
        )

        model = model.to(self.device)

        # Log parameter count
        param_counts = model.get_num_params()
        logger.info(f"Model parameters:")
        for name, count in param_counts.items():
            logger.info(f"  {name}: {count:,}")

        return model

    def _build_criterion(self) -> nn.Module:
        """Build the loss function with support for baseline-relative learning."""
        crf_range = self.project_cfg['crf_range']
        
        # Determine loss type
        loss_type = self.phase1_cfg.get('loss_type', 'dual_learning')
        
        if loss_type == 'baseline_relative':
            logger.info("Using BaselineRelativeLoss (beat the do-nothing baseline)")
            return BaselineRelativeLoss(
                reconstruction_weight=self.phase1_cfg.get('reconstruction_loss_weight', 1.0),
                crf_weight=self.phase1_cfg.get('crf_prediction_loss_weight', 0.1),
                use_perceptual=self.phase1_cfg.get('use_perceptual_loss', False),
                perceptual_weight=self.phase1_cfg.get('perceptual_weight', 1.0),
                margin=self.phase1_cfg.get('baseline_margin', 0.0),
                soft_margin=self.phase1_cfg.get('soft_margin', True),
                crf_min=float(crf_range[0]),
                crf_max=float(crf_range[1]),
                charbonnier_eps=self.phase1_cfg.get('charbonnier_eps', 1e-3)
            )
        
        elif loss_type == 'adaptive_baseline_relative':
            logger.info("Using AdaptiveBaselineRelativeLoss (adaptive + baseline-relative)")
            return AdaptiveBaselineRelativeLoss(
                use_perceptual=self.phase1_cfg.get('use_perceptual_loss', False),
                perceptual_weight=self.phase1_cfg.get('perceptual_weight', 1.0),
                margin=self.phase1_cfg.get('baseline_margin', 0.0),
                soft_margin=self.phase1_cfg.get('soft_margin', True),
                crf_min=float(crf_range[0]),
                crf_max=float(crf_range[1]),
                charbonnier_eps=self.phase1_cfg.get('charbonnier_eps', 1e-3)
            )
        
        elif loss_type == 'adaptive_dual_learning':
            logger.info("Using AdaptiveDualLearningLoss (automatic task weighting)")
            return AdaptiveDualLearningLoss(
                use_perceptual=self.phase1_cfg.get('use_perceptual_loss', False),
                perceptual_weight=self.phase1_cfg.get('perceptual_weight', 1.0),
                crf_min=float(crf_range[0]),
                crf_max=float(crf_range[1]),
                charbonnier_eps=self.phase1_cfg.get('charbonnier_eps', 1e-3)
            )
        
        else:  # 'dual_learning' (default)
            logger.info("Using DualLearningLoss (manual weighting)")
            return DualLearningLoss(
                reconstruction_weight=self.phase1_cfg.get('reconstruction_loss_weight', 1.0),
                crf_weight=self.phase1_cfg.get('crf_prediction_loss_weight', 0.1),
                use_perceptual=self.phase1_cfg.get('use_perceptual_loss', False),
                perceptual_weight=self.phase1_cfg.get('perceptual_weight', 1.0),
                crf_min=float(crf_range[0]),
                crf_max=float(crf_range[1]),
                charbonnier_eps=self.phase1_cfg.get('charbonnier_eps', 1e-3)
            )

    def _init_logging(self):
        """Initialize W&B logging with resume support."""
        if not (self.project_cfg.get('log_to_wandb', False) and WANDB_AVAILABLE):
            if self.project_cfg.get('log_to_wandb', False):
                logger.warning("wandb logging requested but not available")
            return

        # Resume existing run or create new one
        if self.wandb_id:
            logger.info(f"Resuming W&B run: {self.wandb_id}")
            wandb.init(
                project=self.project_cfg['project_name'],
                id=self.wandb_id,
                resume="must",
                config=self.config
            )
        else:
            logger.info("Starting new W&B run")
            wandb.init(
                project=self.project_cfg['project_name'],
                config=self.config,
                name=f"phase1_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                tags=['phase1', 'prior_embedder', 'dual_learning']
            )
            self.wandb_id = wandb.run.id

        logger.info(f"W&B Run ID: {wandb.run.id}")
        logger.info(f"W&B Run URL: {wandb.run.url}")

    def _sample_validation_images(self, patch_size: int, num_samples: int = 4) -> Dict:
        """Sample fixed validation images for visual logging."""
        logger.info(f"Sampling {num_samples} validation images ({patch_size}x{patch_size})")

        dataset = AV1Dataset(
            lq_root_dir=self.project_cfg['lq_data_root'],
            hq_root_dir=self.project_cfg['hq_data_root'],
            hq_ext=self.project_cfg['hq_file_extension'],
            patch_size=patch_size,
            crf_range=tuple(self.project_cfg['crf_range']),
            preset_range=tuple(self.project_cfg['preset_range']),
            augment=False,
            return_metadata=True
        )

        dataloader = DataLoader(
            dataset, batch_size=num_samples, shuffle=False,
            num_workers=0, pin_memory=False, drop_last=False
        )

        sample_batch = next(iter(dataloader))

        # Move to device
        for key in ['lq', 'hq', 'crf']:
            if key in sample_batch:
                sample_batch[key] = sample_batch[key].to(self.device)

        return sample_batch

    def _create_dataloader(self, patch_size: int, batch_size: int, is_train: bool = True) -> DataLoader:
        """Create dataloader for a curriculum stage."""
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
        """Create optimizer."""
        optimizer_type = self.phase1_cfg.get('optimizer', 'adamw')

        # Collect all parameters
        params = list(self.model.parameters())

        # Add adaptive loss parameters if using AdaptiveDualLearningLoss
        if isinstance(self.criterion, AdaptiveDualLearningLoss):
            params.extend([self.criterion.log_var_rec, self.criterion.log_var_crf])
            logger.info("Added adaptive loss uncertainty parameters to optimizer")

        if optimizer_type.lower() == 'adamw':
            optimizer = AdamW(
                params,
                lr=lr,
                betas=(0.9, 0.999),
                weight_decay=self.phase1_cfg.get('weight_decay', 1e-4)
            )
        elif optimizer_type.lower() == 'adam':
            optimizer = Adam(
                params,
                lr=lr,
                betas=(0.9, 0.999)
            )
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_type}")

        logger.info(f"Created optimizer: {optimizer_type} with lr={lr}")
        return optimizer

    def _create_scheduler(self, optimizer: torch.optim.Optimizer, total_steps: int):
        """Create LR scheduler."""
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

    def train_epoch(self,
        dataloader: DataLoader,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[Any],
        epoch: int,
        stage_idx: int,
        patch_size: int
    ) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()

        # Accumulators
        total_loss = 0.0
        total_recon_loss = 0.0
        total_crf_loss = 0.0
        total_recon_improvement = 0.0
        total_crf_improvement = 0.0
        num_batches = 0

        pbar = tqdm(
            dataloader,
            desc=f"Stage {stage_idx+1} | Epoch {epoch+1} | Patch {patch_size}",
            leave=True
        )

        for batch_idx, batch in enumerate(pbar):
            lq_images = batch['lq'].to(self.device, non_blocking=True)
            hq_images = batch['hq'].to(self.device, non_blocking=True)
            crf_values = batch['crf'].to(self.device, non_blocking=True)

            # Forward pass
            reconstructed_images, predicted_crf, _ = self.model(lq_images)

            # Compute loss (baseline-relative needs lq_images)
            is_baseline_relative = isinstance(
                self.criterion, 
                (BaselineRelativeLoss, AdaptiveBaselineRelativeLoss)
            )
            
            if is_baseline_relative:
                loss_dict = self.criterion(
                    reconstructed_images, lq_images, hq_images,
                    predicted_crf, crf_values
                )
            else:
                loss_dict = self.criterion(
                    reconstructed_images, hq_images,
                    predicted_crf, crf_values
                )

            loss = loss_dict['total_loss']

            # Backward pass
            optimizer.zero_grad()
            loss.backward()

            if self.phase1_cfg.get('grad_clip', 0) > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.phase1_cfg['grad_clip']
                )

            optimizer.step()

            if scheduler is not None:
                scheduler.step()

            # Accumulate metrics
            total_loss += loss.item()
            total_recon_loss += loss_dict['reconstruction_loss'].item()
            total_crf_loss += loss_dict['crf_loss'].item()
            
            # Track improvements if using baseline-relative
            if is_baseline_relative:
                total_recon_improvement += loss_dict['reconstruction_improvement'].item()
                total_crf_improvement += loss_dict['crf_improvement'].item()
            
            num_batches += 1

            # Update progress bar
            current_lr = optimizer.param_groups[0]['lr']
            pbar_dict = {
                'Loss': f"{loss.item():.4f}",
                'Recon': f"{loss_dict['reconstruction_loss'].item():.4f}",
                'CRF': f"{loss_dict['crf_loss'].item():.4f}",
                'LR': f"{current_lr:.2e}"
            }

            # Add improvement metrics for baseline-relative
            if is_baseline_relative:
                pbar_dict['Δ_rec'] = f"{loss_dict['reconstruction_improvement'].item():.4f}"
                pbar_dict['Δ_crf'] = f"{loss_dict['crf_improvement'].item():.4f}"

            # Add adaptive weights if applicable
            if isinstance(self.criterion, (AdaptiveDualLearningLoss, AdaptiveBaselineRelativeLoss)):
                pbar_dict['W_rec'] = f"{loss_dict['weight_rec'].item():.2f}"
                pbar_dict['W_crf'] = f"{loss_dict['weight_crf'].item():.2f}"

            pbar.set_postfix(pbar_dict)

            # Log to wandb
            if self.project_cfg.get('log_to_wandb', False) and WANDB_AVAILABLE:
                log_dict = {
                    'train/total_loss': loss.item(),
                    'train/reconstruction_loss': loss_dict['reconstruction_loss'].item(),
                    'train/crf_loss': loss_dict['crf_loss'].item(),
                    'train/learning_rate': current_lr,
                    'train/epoch': epoch,
                    'train/stage': stage_idx,
                    'train/patch_size': patch_size,
                    'step': self.global_step
                }

                # Add baseline-relative metrics
                if is_baseline_relative:
                    log_dict.update({
                        'train/reconstruction_improvement': loss_dict['reconstruction_improvement'].item(),
                        'train/crf_improvement': loss_dict['crf_improvement'].item(),
                        'train/baseline_reconstruction_loss': loss_dict['baseline_reconstruction_loss'].item(),
                        'train/baseline_crf_loss': loss_dict['baseline_crf_loss'].item(),
                        'train/crf_baseline_value': loss_dict['crf_baseline_value'].item()
                    })

                # Add adaptive weights
                if isinstance(self.criterion, (AdaptiveDualLearningLoss, AdaptiveBaselineRelativeLoss)):
                    log_dict.update({
                        'train/weight_rec': loss_dict['weight_rec'].item(),
                        'train/weight_crf': loss_dict['weight_crf'].item(),
                        'train/uncertainty_rec': loss_dict['uncertainty_rec'].item(),
                        'train/uncertainty_crf': loss_dict['uncertainty_crf'].item()
                    })

                wandb.log(log_dict)

            self.global_step += 1

        # Calculate epoch averages
        avg_metrics = {
            'total_loss': total_loss / num_batches,
            'reconstruction_loss': total_recon_loss / num_batches,
            'crf_loss': total_crf_loss / num_batches
        }

        if is_baseline_relative:
            avg_metrics.update({
                'reconstruction_improvement': total_recon_improvement / num_batches,
                'crf_improvement': total_crf_improvement / num_batches
            })

        return avg_metrics

    @torch.no_grad()
    def validate(self, dataloader: DataLoader, epoch: int, stage_idx: int) -> Dict[str, float]:
        """Run validation with visual logging."""
        self.model.eval()

        total_loss = 0.0
        total_recon_loss = 0.0
        total_crf_loss = 0.0
        total_perceptual_loss = 0.0
        num_batches = 0

        pbar = tqdm(dataloader, desc=f"Validation | Epoch {epoch+1}", leave=False)

        for batch in pbar:
            lq_images = batch['lq'].to(self.device, non_blocking=True)
            hq_images = batch['hq'].to(self.device, non_blocking=True)
            crf_values = batch['crf'].to(self.device, non_blocking=True)

            reconstructed_images, predicted_crf, _ = self.model(lq_images)

            loss_dict = self.criterion(
                reconstructed_images, hq_images,
                predicted_crf, crf_values
            )

            total_loss += loss_dict['total_loss'].item()
            total_recon_loss += loss_dict['reconstruction_loss'].item()
            total_crf_loss += loss_dict['crf_loss'].item()
            if 'perceptual_loss' in loss_dict:
                total_perceptual_loss += loss_dict['perceptual_loss'].item()
            num_batches += 1

        avg_metrics = {
            'val_total_loss': total_loss / num_batches,
            'val_reconstruction_loss': total_recon_loss / num_batches,
            'val_crf_loss': total_crf_loss / num_batches
        }

        if total_perceptual_loss > 0:
            avg_metrics['val_perceptual_loss'] = total_perceptual_loss / num_batches

        logger.info(f"Validation Results - Loss: {avg_metrics['val_total_loss']:.4f}")

        # Visual logging to W&B
        if self.project_cfg.get('log_to_wandb', False) and WANDB_AVAILABLE:
            sample_batch = self.validation_samples
            lq_images = sample_batch['lq']
            hq_images = sample_batch['hq']
            crf_values = sample_batch['crf']
            base_names = sample_batch.get('base_name', [f"sample_{i}" for i in range(lq_images.size(0))])

            reconstructed_images, predicted_crf, _ = self.model(lq_images)

            log_images = []
            for i in range(lq_images.size(0)):
                true_crf = crf_values[i].item()
                pred_crf = predicted_crf[i].item()

                # Denormalize if using improved loss
                if hasattr(self.criterion, 'denormalize_crf'):
                    pred_crf = self.criterion.denormalize_crf(torch.tensor(pred_crf)).item()

                caption = (
                    f"Epoch {epoch+1} | CRF: True={true_crf:.1f}, Pred={pred_crf:.1f} | "
                    f"{base_names[i]}"
                )

                # Create side-by-side comparison: LQ | Reconstructed | HQ
                image_grid = torch.cat([lq_images[i], reconstructed_images[i], hq_images[i]], dim=2)
                log_images.append(wandb.Image(image_grid.cpu(), caption=caption))

            log_dict = {
                'val/samples': log_images,
                **{f'val/{k}': v for k, v in avg_metrics.items()},
                'val/epoch': epoch,
                'val/stage': stage_idx
            }

            # Add adaptive weights if using AdaptiveDualLearningLoss
            if isinstance(self.criterion, AdaptiveDualLearningLoss):
                log_dict.update({
                    'val/weight_rec': torch.exp(-self.criterion.log_var_rec).item(),
                    'val/weight_crf': torch.exp(-self.criterion.log_var_crf).item(),
                    'val/uncertainty_rec': torch.exp(self.criterion.log_var_rec * 0.5).item(),
                    'val/uncertainty_crf': torch.exp(self.criterion.log_var_crf * 0.5).item()
                })

            wandb.log(log_dict)

        return avg_metrics

    def _save_checkpoint(
        self,
        epoch: int,
        stage_idx: int,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[Any] = None,
        is_best: bool = False,
        suffix: str = ''
    ):
        """Save complete training checkpoint."""
        checkpoint = {
            # Model state
            'model_state_dict': self.model.state_dict(),

            # Optimizer state (includes momentum buffers, etc.)
            'optimizer_state_dict': optimizer.state_dict() if optimizer else None,

            # Scheduler state (LR schedule position)
            'scheduler_state_dict': scheduler.state_dict() if scheduler else None,

            # Loss function state (for adaptive weights)
            'criterion_state_dict': self.criterion.state_dict() if isinstance(self.criterion, AdaptiveDualLearningLoss) else None,

            # Training progress
            'epoch': epoch,
            'stage_idx': stage_idx,
            'global_step': self.global_step,
            'best_val_loss': self.best_val_loss,

            # W&B integration
            'wandb_id': self.wandb_id if WANDB_AVAILABLE else None,

            # Configuration
            'config': self.config
        }

        # Determine checkpoint path
        if suffix:
            ckpt_path = self.checkpoint_dir / f"checkpoint_{suffix}.pth"
        else:
            ckpt_path = self.checkpoint_dir / f"checkpoint_stage{stage_idx}_epoch{epoch}.pth"

        torch.save(checkpoint, ckpt_path)
        logger.info(f"Checkpoint saved: {ckpt_path}")

        # Also save as latest.pth for easy resumption
        latest_path = self.checkpoint_dir / "latest.pth"
        torch.save(checkpoint, latest_path)
        logger.info(f"Latest checkpoint: {latest_path}")

        # Save best model
        if is_best:
            best_path = self.checkpoint_dir / "best_model.pth"
            torch.save(checkpoint, best_path)
            logger.info(f"✨ Best model saved: {best_path}")

    def _load_checkpoint(self, checkpoint_path: str) -> Tuple[Optional[dict], Optional[dict], Optional[dict]]:
        """
        Load checkpoint and restore training state.

        Returns:
            Tuple of (optimizer_state_dict, scheduler_state_dict, criterion_state_dict)
        """
        ckpt_path = Path(checkpoint_path).expanduser()

        if not ckpt_path.exists():
            logger.error(f"Checkpoint not found: {ckpt_path}")
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

        logger.info(f"Loading checkpoint from: {ckpt_path}")
        checkpoint = torch.load(ckpt_path, map_location=self.device)

        # Restore model state
        self.model.load_state_dict(checkpoint['model_state_dict'])
        logger.info("✓ Model state restored")

        # Restore training progress
        self.global_step = checkpoint.get('global_step', 0)
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        self.start_stage = checkpoint.get('stage_idx', 0)
        self.start_epoch = checkpoint.get('epoch', -1) + 1

        logger.info(f"✓ Training progress restored:")
        logger.info(f"  Stage: {self.start_stage}")
        logger.info(f"  Epoch: {self.start_epoch}")
        logger.info(f"  Global step: {self.global_step}")
        logger.info(f"  Best val loss: {self.best_val_loss:.4f}")

        # Restore W&B run ID if available
        if 'wandb_id' in checkpoint and checkpoint['wandb_id']:
            if not self.wandb_id:
                self.wandb_id = checkpoint['wandb_id']
                logger.info(f"✓ W&B run ID restored: {self.wandb_id}")

        # Return optimizer, scheduler, and criterion states
        optimizer_state = checkpoint.get('optimizer_state_dict')
        scheduler_state = checkpoint.get('scheduler_state_dict')
        criterion_state = checkpoint.get('criterion_state_dict')

        return optimizer_state, scheduler_state, criterion_state

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

        # Store optimizer/scheduler/criterion states from checkpoint (if resuming)
        saved_optimizer_state = None
        saved_scheduler_state = None
        saved_criterion_state = None
        if self.resume_from:
            saved_optimizer_state, saved_scheduler_state, saved_criterion_state = self._load_checkpoint(self.resume_from)

            # Restore criterion state if using AdaptiveDualLearningLoss
            if saved_criterion_state and isinstance(self.criterion, AdaptiveDualLearningLoss):
                self.criterion.load_state_dict(saved_criterion_state)
                logger.info("✓ Adaptive loss weights restored")

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

            # Create dataloaders
            train_loader = self._create_dataloader(patch_size, batch_size, is_train=True)

            val_loader = None
            if self.phase1_cfg.get('validate_every_n_epochs', 0) > 0:
                val_loader = self._create_dataloader(
                    patch_size,
                    batch_size=min(batch_size * 2, 32),
                    is_train=False
                )

            # Create optimizer and scheduler
            lr = self.phase1_cfg.get('learning_rate', 1e-4)
            optimizer = self._create_optimizer(lr)

            total_steps = epochs * len(train_loader)
            scheduler = self._create_scheduler(optimizer, total_steps)

            # Restore optimizer/scheduler state if resuming THIS stage
            if stage_idx == self.start_stage and saved_optimizer_state:
                optimizer.load_state_dict(saved_optimizer_state)
                logger.info("✓ Optimizer state restored")

                if scheduler and saved_scheduler_state:
                    scheduler.load_state_dict(saved_scheduler_state)
                    logger.info("✓ Scheduler state restored")

                # Clear saved states after loading
                saved_optimizer_state = None
                saved_scheduler_state = None
                saved_criterion_state = None

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
                        self._save_checkpoint(epoch, stage_idx, optimizer, scheduler, is_best=True)

                # Save checkpoint periodically
                if (epoch + 1) % self.phase1_cfg.get('save_every_n_epochs', 10) == 0:
                    self._save_checkpoint(epoch, stage_idx, optimizer, scheduler)

            # Save checkpoint after completing stage
            self._save_checkpoint(epochs - 1, stage_idx, optimizer, scheduler, suffix=f"stage{stage_idx}_complete")

            # Reset start_epoch for next stage
            self.start_epoch = 0

        # Save final model
        final_path = self.checkpoint_dir / "final_model.pth"
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'criterion_state_dict': self.criterion.state_dict() if isinstance(self.criterion, AdaptiveDualLearningLoss) else None,
            'config': self.config,
            'global_step': self.global_step,
            'best_val_loss': self.best_val_loss
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
        help='Path to checkpoint to resume from (use "latest" for ./checkpoints/phase1_av1_p4/latest.pth)'
    )

    parser.add_argument(
        '--wandb_id',
        type=str,
        default=None,
        help='W&B run ID to resume. If not provided, will use ID from checkpoint or create new run.'
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
    """Setup and return the appropriate device for training."""
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
    """Validate training configuration."""
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

    # Handle "latest" shortcut for resume
    resume_from = args.resume
    if resume_from == "latest":
        checkpoint_dir = Path(config['phase1']['checkpoint_save_path']).expanduser()
        resume_from = str(checkpoint_dir / "latest.pth")
        logger.info(f"Using latest checkpoint: {resume_from}")

    # Create trainer
    trainer = CurriculumTrainer(
        config=config,
        device=device,
        resume_from=resume_from,
        wandb_id=args.wandb_id
    )

    # Start training
    try:
        trainer.train()
    except KeyboardInterrupt:
        logger.warning("Training interrupted by user")
        logger.info("Saving emergency checkpoint...")

        # Save emergency checkpoint with current state
        if hasattr(trainer, 'optimizer') and trainer.optimizer:
            trainer._save_checkpoint(
                epoch=trainer.start_epoch if hasattr(trainer, 'start_epoch') else 0,
                stage_idx=trainer.start_stage if hasattr(trainer, 'start_stage') else 0,
                optimizer=trainer.optimizer,
                scheduler=trainer.scheduler if hasattr(trainer, 'scheduler') else None,
                suffix='emergency_interrupt'
            )
            logger.info("Emergency checkpoint saved. You can resume with --resume latest")
        else:
            logger.warning("No optimizer available, saving model state only")
            emergency_path = trainer.checkpoint_dir / "emergency_model_only.pth"
            torch.save({
                'model_state_dict': trainer.model.state_dict(),
                'config': trainer.config
            }, emergency_path)
            logger.info(f"✓ Model state saved: {emergency_path}")

    except Exception as e:
        logger.error(f"Training failed with error: {e}", exc_info=True)

        # Try to save emergency checkpoint
        try:
            logger.info("Attempting to save emergency checkpoint...")
            if hasattr(trainer, 'optimizer') and trainer.optimizer:
                trainer._save_checkpoint(
                    epoch=trainer.start_epoch if hasattr(trainer, 'start_epoch') else 0,
                    stage_idx=trainer.start_stage if hasattr(trainer, 'start_stage') else 0,
                    optimizer=trainer.optimizer,
                    scheduler=trainer.scheduler if hasattr(trainer, 'scheduler') else None,
                    suffix='emergency_error'
                )
                logger.info("Emergency checkpoint saved")
        except Exception as save_error:
            logger.error(f"Failed to save emergency checkpoint: {save_error}")

        raise


if __name__ == "__main__":
    main()