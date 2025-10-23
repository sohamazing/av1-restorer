# train_conditional_unet_patch_gan.py
"""
Training Script for Conditional AV1 U-Net Restorers using GAN.

Supports both standard (CRF+Preset) and CRF-Only U-Net generators.
Automatically selects generator based on config's preset_range.
Uses a PatchGAN discriminator for adversarial training.

==============================================================================
FEATURES
==============================================================================
- Generator: AV1ConditionalUNet or AV1ConditionalUNetCRF
- Discriminator: PatchGAN
- Automatic generator selection based on preset_range
- Pre-processed (.npy) and on-the-fly (.avif) dataset loading
- Curriculum learning (progressive patch sizes, CRF ranges)
- Training: GAN (LSGAN) + Reconstruction Loss (L1, Perceptual, etc.)
- SOTA training: Mixed Precision (AMP), EMA (Generator only), Warmup + Cosine Annealing (Generator only)
- W&B integration
- Robust checkpointing
- Cross-platform (CUDA, MPS, CPU) via torch.amp

==============================================================================
USAGE
==============================================================================
Basic:
    python train_conditional_unet_gan.py --config configs/unet_gan_base.yaml

Resume:
    python train_conditional_unet_gan.py --config configs/unet_gan_base.yaml --resume latest

Resume W&B:
    python train_conditional_unet_gan.py --config configs/unet_gan_base.yaml \
        --resume best --wandb_id abc123def

==============================================================================
Author: Soham Mukherjee
Version: 2.1 (Conditional U-Net GAN, PyTorch Amp Compliant, No Semicolons)
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

# --- Generator ---
# Imports the unified factory and model classes
from av1_restorer.models.av1_conditional_unet_restorer import (
    create_av1_restorer,
    AV1ConditionalUNet,
    AV1ConditionalUNetCRF
)
# --- Discriminator ---
from av1_restorer.models.patch_gan_discriminator import PatchGANDiscriminator

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
logger = logging.getLogger("ConditionalUNetGANTrainer")


# ==============================================================================
# SECTION 1: Exponential Moving Average (EMA) - Applied to Generator Only
# ==============================================================================

class EMA:
    """ Exponential Moving Average for stable model evaluation. """
    def __init__(self, model: nn.Module, decay: float = 0.9999):
        self.model = model
        self.decay = decay
        self.shadow = {
            name: param.data.clone()
            for name, param in model.named_parameters() if param.requires_grad
        }
        self.backup = {}

    def update(self):
        for name, param in self.model.named_parameters():
             if param.requires_grad:
                 self.shadow[name].data.mul_(self.decay).add_(param.data, alpha=1.0 - self.decay)

    def apply_shadow(self):
        self.backup = {
            name: param.data.clone()
            for name, param in self.model.named_parameters() if param.requires_grad
        }
        for name, param in self.model.named_parameters():
             if param.requires_grad:
                 param.data.copy_(self.shadow[name])

    def restore(self):
        for name, param in self.model.named_parameters():
             if param.requires_grad and name in self.backup: # Check if key exists before copying
                 param.data.copy_(self.backup[name])
        self.backup = {}


# ==============================================================================
# SECTION 2: Conditional GAN Trainer
# ==============================================================================

class ConditionalUNetGANTrainer:
    """ Config-driven GAN trainer for Conditional AV1 U-Net Restorers. """

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

        # System setup
        self._setup_system() # Sets self.device, self.use_amp, self.scaler (if cuda)

        # Models and Losses
        self.generator, self.discriminator, self.model_needs_preset = self._setup_models()
        self.reconstruction_loss_fn, self.adversarial_loss_fn = self._setup_losses()
        self.lambda_recon = self.config['loss'].get('reconstruction_weight', 100.0)
        self.lambda_adv = self.config['loss'].get('adversarial_weight', 1.0)

        # Optimizers and Schedulers
        self.optimizer_G, self.optimizer_D, self.scheduler_G = self._setup_optimizers()

        # EMA (Generator Only)
        self.ema_G = None
        if self.config['optimizer_G'].get('use_ema', False):
            ema_decay = self.config['optimizer_G'].get('ema_decay', 0.9999)
            self.ema_G = EMA(self.generator, decay=ema_decay)
            logger.info(f"EMA enabled for Generator (decay={ema_decay})")

        # W&B logging
        self._setup_logging()

        # Summary and validation samples
        self._print_summary()
        self.val_samples = self._get_fixed_val_samples() # Uses Generator only

        # Training state initialization
        self.start_stage = 0
        self.start_epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf') # Based on Generator validation loss
        self.saved_optimizer_G_state = None
        self.saved_optimizer_D_state = None
        self.saved_scheduler_G_state = None
        self.adv_label_shape = None # Will be determined in first training step

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
                logger.warning("Device: CPU")
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
        self.scaler = None # Initialize scaler
        if self.use_amp:
            if self.device.type == 'cuda':
                # GradScaler is implicitly for CUDA and enabled by default if instantiated
                self.scaler = torch.amp.GradScaler(enabled=True)
                logger.info("AMP: Enabled with GradScaler (CUDA)")
            else:
                # MPS/CPU: AMP works but no GradScaler
                logger.info(f"AMP: Enabled without GradScaler ({self.device.type})")
        else:
            logger.info("AMP: Disabled")

    def _setup_models(self) -> Tuple[nn.Module, nn.Module, bool]:
        """ Setup Generator (U-Net) and Discriminator (PatchGAN). """
        model_cfg = self.config['model'] # Generator config section
        dset_cfg = self.config['dataset']
        disc_cfg = self.config.get('discriminator', {}) # Discriminator config section

        # --- Generator Setup ---
        model_type = model_cfg.get('type', 'unet')
        if model_type != 'unet':
            raise ValueError(f"This script only supports generator type 'unet', found '{model_type}'.")

        size = model_cfg['size']
        crf_range = tuple(dset_cfg['crf_range'])
        preset_range = tuple(dset_cfg.get('preset_range', [0, 8]))
        norm_range = tuple(dset_cfg.get('norm_range', [-1, 1]))

        logger.info(f"Creating Generator: {model_type} (size={size})")
        model_needs_preset: bool
        if preset_range[0] == preset_range[1]:
            logger.info(f"Preset range is single value ({preset_range[0]}). Using CRF-Only Generator.")
            model_needs_preset = False
        else:
            logger.info("Using Standard Generator (CRF + Preset).")
            model_needs_preset = True

        # Use the unified factory from the model file
        generator = create_av1_restorer(
            size=size, crf_range=crf_range, preset_range=preset_range, norm_range=norm_range
        )
        generator = generator.to(self.device)
        gen_params = sum(p.numel() for p in generator.parameters() if p.requires_grad)
        logger.info(f"Generator parameters: {gen_params:,} ({gen_params/1e6:.2f}M)")

        # --- Discriminator Setup ---
        logger.info("Creating Discriminator: PatchGAN")
        disc_in_channels = disc_cfg.get('in_channels', 3)
        disc_base_channels = disc_cfg.get('base_channels', 64)
        disc_n_layers = disc_cfg.get('n_layers', 3)
        discriminator = PatchGANDiscriminator(
            in_channels=disc_in_channels,
            base_channels=disc_base_channels,
            n_layers=disc_n_layers
        )
        discriminator = discriminator.to(self.device)
        # Optional: Initialize discriminator weights
        # self._init_weights(discriminator) # Define this if needed
        disc_params = sum(p.numel() for p in discriminator.parameters() if p.requires_grad)
        logger.info(f"Discriminator parameters: {disc_params:,} ({disc_params/1e6:.2f}M)")

        return generator, discriminator, model_needs_preset

    def _setup_losses(self) -> Tuple[nn.Module, nn.Module]:
        """ Initialize Reconstruction and Adversarial losses. """
        norm_range = tuple(self.config['dataset'].get('norm_range', [-1, 1]))
        reconstruction_loss_fn = CombinedLoss(self.config['loss'], norm_range=norm_range)
        logger.info("Reconstruction loss initialized")

        # LSGAN uses MSE Loss
        adversarial_loss_fn = nn.MSELoss()
        logger.info("Adversarial loss (LSGAN/MSE) initialized")

        return reconstruction_loss_fn.to(self.device), adversarial_loss_fn.to(self.device)

    def _setup_optimizers(self) -> Tuple[torch.optim.Optimizer, torch.optim.Optimizer, Optional[Any]]:
        """ Create optimizers for G and D, scheduler for G. """
        opt_g_cfg = self.config['optimizer_G']
        opt_d_cfg = self.config['optimizer_D']
        sched_g_cfg = self.config['scheduler_G']

        # --- Optimizer G ---
        lr_g = opt_g_cfg['lr']
        if opt_g_cfg['type'].lower() == 'adamw':
            optimizer_G = AdamW(self.generator.parameters(), lr=lr_g,
                                betas=tuple(opt_g_cfg.get('betas', [0.9, 0.999])),
                                weight_decay=opt_g_cfg.get('weight_decay', 1e-4))
        else: # Default Adam
            optimizer_G = Adam(self.generator.parameters(), lr=lr_g,
                               betas=tuple(opt_g_cfg.get('betas', [0.9, 0.999])))
        logger.info(f"Optimizer G: {opt_g_cfg['type'].upper()}, lr={lr_g}")

        # --- Optimizer D ---
        lr_d = opt_d_cfg['lr']
        if opt_d_cfg['type'].lower() == 'adamw':
            optimizer_D = AdamW(self.discriminator.parameters(), lr=lr_d,
                                betas=tuple(opt_d_cfg.get('betas', [0.5, 0.999])), # Common GAN betas
                                weight_decay=opt_d_cfg.get('weight_decay', 0)) # Often no WD for D
        else: # Default Adam
            optimizer_D = Adam(self.discriminator.parameters(), lr=lr_d,
                               betas=tuple(opt_d_cfg.get('betas', [0.5, 0.999])))
        logger.info(f"Optimizer D: {opt_d_cfg['type'].upper()}, lr={lr_d}")

        # --- Scheduler G ---
        # Estimate total steps using fallback (will be updated in train())
        estimated_total_steps_G = sum(
            sum(s['epochs']) if isinstance(s['epochs'], list) else s['epochs']
            for s in self.config['curriculum']
        ) * 500

        scheduler_G = None
        scheduler_type = sched_g_cfg.get('type', 'cosine').lower()
        warmup_steps = sched_g_cfg.get('warmup_steps', 500)
        if scheduler_type == 'cosine':
            main_scheduler = CosineAnnealingLR(
                optimizer_G, T_max=max(1, estimated_total_steps_G - warmup_steps),
                eta_min=sched_g_cfg.get('min_lr', 1e-6)
            )
            warmup_scheduler = LinearLR(
                optimizer_G, start_factor=1e-5, end_factor=1.0, total_iters=warmup_steps
            )
            scheduler_G = SequentialLR(
                optimizer_G, schedulers=[warmup_scheduler, main_scheduler],
                milestones=[warmup_steps]
            )
            logger.info(f"Scheduler G: Cosine with {warmup_steps}-step warmup (T_max based on estimate)")
        else:
            logger.info("Scheduler G: None")

        return optimizer_G, optimizer_D, scheduler_G

    def _print_summary(self):
        """Print training configuration summary."""
        logger.info("=" * 80)
        logger.info(f"{'AV1 Conditional U-Net GAN - Training Configuration':^80}")
        logger.info("=" * 80)
        proj = self.config['project']
        logger.info(f"Project: {proj['name']}")
        logger.info(f"Experiment: {proj['experiment_name']}")

        model_cfg = self.config['model']
        gen_params = sum(p.numel() for p in self.generator.parameters())
        logger.info(f"Generator: {model_cfg['type']} ({model_cfg['size']})")
        logger.info(f"  Params: {gen_params:,} ({gen_params/1e6:.2f}M)")
        logger.info(f"  Conditioning: CRF{' + Preset' if self.model_needs_preset else '-Only'}")

        disc_cfg = self.config.get('discriminator', {})
        disc_params = sum(p.numel() for p in self.discriminator.parameters())
        logger.info(f"Discriminator: PatchGAN")
        logger.info(f"  Params: {disc_params:,} ({disc_params/1e6:.2f}M)")
        logger.info(f"  Layers: {disc_cfg.get('n_layers', 3)}, Base Channels: {disc_cfg.get('base_channels', 64)}")

        dset_cfg = self.config['dataset']
        logger.info(f"Dataset CRF Range: {dset_cfg['crf_range']}")
        logger.info(f"Dataset Preset Range: {dset_cfg['preset_range']}")

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
                logger.info(f"  Stage {i+1}: {epochs} epochs, {patch}px, batch {batch}, CRF {crf}")

        logger.info(f"Loss Weights: Recon λ={self.lambda_recon}, Adv λ={self.lambda_adv}")
        logger.info(f"Device: {self.device} | AMP: {self.use_amp} | EMA (G): {self.ema_G is not None}")
        logger.info("=" * 80)

    def _get_fixed_val_samples(self) -> Optional[Dict[str, Any]]:
        """Sample fixed validation images using the Generator."""
        try:
            data_cfg = self.config['data']
            dset_cfg = self.config['dataset']
            num_samples = self.config['training'].get('num_val_samples_to_log', 4)

            last_stage = self.config['curriculum'][-1]
            patch_size = last_stage['patch_size']
            patch_size = patch_size[-1] if isinstance(patch_size, list) else patch_size

            val_cache_key = f"{data_cfg['val_lq_root']}_{tuple(dset_cfg['crf_range'])}"
            cached_val = self.val_image_pairs_cache.get(val_cache_key)

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
                cached_image_pairs=cached_val
            )
            if cached_val is None and hasattr(val_dataset, 'image_pairs'):
                self.val_image_pairs_cache[val_cache_key] = val_dataset.image_pairs

            loader = DataLoader(val_dataset, batch_size=num_samples, shuffle=False)
            batch = next(iter(loader))
            result = {
                k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }
            logger.info(f"Sampled {num_samples} validation images")
            return result
        except Exception as e:
            logger.warning(f"Could not sample val images: {e}")
            return None

    def _create_dataloaders(self, stage_config: dict) -> Tuple[DataLoader, DataLoader]:
        """Create dataloaders for a curriculum stage."""
        data_cfg = self.config['data']
        dset_cfg = self.config['dataset']
        sys_cfg = self.config['system']

        patch_size = stage_config['patch_size']
        batch_size = stage_config['batch_size']
        crf_range = stage_config.get('crf_range', dset_cfg['crf_range'])

        train_key = f"{data_cfg['train_lq_root']}_{tuple(crf_range)}"
        val_key = f"{data_cfg['val_lq_root']}_{tuple(crf_range)}"
        cached_train = self.train_image_pairs_cache.get(train_key)
        cached_val = self.val_image_pairs_cache.get(val_key)

        if isinstance(patch_size, list):
            patch_size = patch_size[0]
            batch_size = batch_size[0]

        lq_ext = data_cfg.get('lq_ext', '.avif').lower()
        DatasetClass = AV1DatasetFast if lq_ext == '.npy' and 'AV1DatasetFast' in globals() else AV1Dataset

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
            self.train_image_pairs_cache[train_key] = train_dset.image_pairs
        if cached_val is None and hasattr(val_dset, 'image_pairs'):
            self.val_image_pairs_cache[val_key] = val_dset.image_pairs

        num_workers = sys_cfg.get('num_workers', 8)
        pin_memory = (self.device.type == 'cuda')

        train_loader = DataLoader(
            train_dset, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=pin_memory, drop_last=True,
            persistent_workers=(num_workers > 0),
            prefetch_factor=4 if num_workers > 0 else None
        )
        val_loader = DataLoader(
            val_dset, batch_size=batch_size * 2, shuffle=False,
            num_workers=max(2, num_workers // 2), pin_memory=pin_memory, drop_last=False,
            persistent_workers=(num_workers > 0),
            prefetch_factor=2 if num_workers > 0 else None
        )
        logger.info(f"Dataloaders: {len(train_dset):,} train, {len(val_dset):,} val")
        return train_loader, val_loader

    # --------------------------------------------------------------------------
    # Training Loop Logic
    # --------------------------------------------------------------------------

    def train(self):
        """Main training loop orchestrating curriculum stages."""
        logger.info("\n" + "=" * 80 + "\nSTARTING TRAINING\n" + "=" * 80 + "\n")
        curriculum = self.config['curriculum']

        # --- Setup optimizers and scheduler ---
        # Note: scheduler T_max is initially estimated
        optimizer_G, optimizer_D, scheduler_G = self._setup_optimizers()

        # --- Recalculate T_max for Scheduler G based on actual dataloader lengths ---
        total_steps_G = 0
        stage_loader_lengths = []
        try:
            logger.info("Calculating actual total steps for G scheduler...")
            for stage_cfg in curriculum:
                # Use a minimal config for dataloader length estimation
                temp_cfg = stage_cfg.copy()
                if isinstance(temp_cfg['patch_size'], list):
                    temp_cfg['patch_size'] = temp_cfg['patch_size'][0]
                    temp_cfg['batch_size'] = temp_cfg['batch_size'][0]

                temp_loader, _ = self._create_dataloaders(temp_cfg)
                loader_len = len(temp_loader)
                stage_loader_lengths.append(loader_len) # Store length for the stage

                stage_epochs = stage_cfg['epochs']
                epochs = sum(stage_epochs) if isinstance(stage_epochs, list) else stage_epochs
                total_steps_G += epochs * loader_len
                del temp_loader # Release resources
            logger.info(f"Actual total G steps: {total_steps_G:,}")

            # Reconfigure scheduler G with correct T_max if it's Cosine
            if scheduler_G and hasattr(scheduler_G, 'schedulers') and isinstance(scheduler_G.schedulers[-1], CosineAnnealingLR):
                 warmup_steps = self.config['scheduler_G'].get('warmup_steps', 500)
                 # Adjust T_max of the main CosineAnnealingLR part
                 scheduler_G.schedulers[-1].T_max = max(1, total_steps_G - warmup_steps)
                 logger.info(f"Scheduler G T_max updated to {scheduler_G.schedulers[-1].T_max}")

        except Exception as e:
            logger.warning(f"Could not accurately calculate total steps: {e}. Scheduler may be inaccurate.")
            # Use fallback length estimate if calculation fails
            stage_loader_lengths = [500] * len(curriculum)

        # --- Restore initial state if needed ---
        if self.start_stage == 0 and self.start_epoch == 0 and self.saved_optimizer_G_state:
            logger.info("Restoring initial optimizer/scheduler state...")
            try:
                optimizer_G.load_state_dict(self.saved_optimizer_G_state)
                logger.info("Optimizer G state restored.")
                if self.saved_optimizer_D_state:
                    optimizer_D.load_state_dict(self.saved_optimizer_D_state)
                    logger.info("Optimizer D state restored.")
                if scheduler_G and self.saved_scheduler_G_state:
                    scheduler_G.load_state_dict(self.saved_scheduler_G_state)
                    logger.info("Scheduler G state restored.")
            except Exception as e:
                logger.error(f"Failed to restore initial state: {e}")
            finally:
                self.saved_optimizer_G_state = None
                self.saved_optimizer_D_state = None
                self.saved_scheduler_G_state = None

        # --- Curriculum loop ---
        for stage_idx in range(self.start_stage, len(curriculum)):
            stage_cfg = curriculum[stage_idx]
            logger.info(f"\n▶ Starting Stage {stage_idx + 1}/{len(curriculum)}")
            # Use pre-calculated loader length for this stage
            current_stage_loader_len = stage_loader_lengths[stage_idx]

            # Pass all optimizers/schedulers to the stage runner
            self._run_stage(
                stage_idx, stage_cfg, optimizer_G, optimizer_D, scheduler_G,
                current_stage_loader_len # Pass estimated length (unused inside)
            )

            # Save stage completion checkpoint
            epochs_in_stage = stage_cfg['epochs']
            last_epoch = (sum(epochs_in_stage) - 1) if isinstance(epochs_in_stage, list) else (epochs_in_stage - 1)
            self._save_checkpoint(
                last_epoch, stage_idx, optimizer_G, optimizer_D, scheduler_G,
                is_stage_complete=True
            )
            self.start_epoch = 0 # Reset epoch count for the next stage

        # --- Training complete ---
        logger.info("\n" + "=" * 80)
        logger.info("🎉 TRAINING COMPLETE!")
        logger.info(f"Best Gen validation loss: {self.best_val_loss:.6f}")
        logger.info("=" * 80 + "\n")
        if WANDB_AVAILABLE and wandb.run:
            try:
                wandb.finish()
                logger.info("W&B finished.")
            except Exception as e:
                logger.error(f"Error finishing W&B: {e}")

    def _run_stage(
        self, stage_idx, stage_cfg, optimizer_G, optimizer_D, scheduler_G,
        loader_len_estimate # This estimate is not used within the function currently
    ):
        """Runs a single curriculum stage."""
        is_progressive = isinstance(stage_cfg.get('patch_size'), list)

        # Restore states if resuming mid-stage
        if stage_idx == self.start_stage and self.start_epoch > 0 and self.saved_optimizer_G_state:
            logger.info(f"Restoring state for Stage {stage_idx+1} resume...")
            try:
                optimizer_G.load_state_dict(self.saved_optimizer_G_state)
                logger.info("Optimizer G state restored.")
                if self.saved_optimizer_D_state:
                    optimizer_D.load_state_dict(self.saved_optimizer_D_state)
                    logger.info("Optimizer D state restored.")
                if scheduler_G and self.saved_scheduler_G_state:
                    scheduler_G.load_state_dict(self.saved_scheduler_G_state)
                    logger.info("Scheduler G state restored.")
            except Exception as e:
                logger.error(f"Failed to restore state: {e}")
            finally:
                self.saved_optimizer_G_state = None
                self.saved_optimizer_D_state = None
                self.saved_scheduler_G_state = None

        if is_progressive:
            patch_sizes = stage_cfg['patch_size']
            batch_sizes = stage_cfg['batch_size']
            epochs_per_sub = stage_cfg['epochs']
            total_epochs = sum(epochs_per_sub)

            if not (len(patch_sizes) == len(batch_sizes) == len(epochs_per_sub)):
                raise ValueError("Progressive stage lists have mismatched lengths.")

            logger.info(f"Progressive stage: {len(patch_sizes)} sub-stages")
            cumulative_epochs = 0

            for sub_idx in range(len(patch_sizes)):
                sub_patch = patch_sizes[sub_idx]
                sub_batch = batch_sizes[sub_idx]
                sub_epochs = epochs_per_sub[sub_idx]

                # Skip completed sub-stages
                if self.start_epoch >= cumulative_epochs + sub_epochs:
                    logger.info(f"  Skipping Sub-stage {stage_idx+1}.{sub_idx+1} (completed).")
                    cumulative_epochs += sub_epochs
                    continue

                logger.info(f"  → Sub-stage {stage_idx+1}.{sub_idx+1}: {sub_epochs} epochs @ {sub_patch}px, batch {sub_batch}")

                sub_cfg = stage_cfg.copy()
                sub_cfg['patch_size'] = sub_patch
                sub_cfg['batch_size'] = sub_batch
                train_loader, val_loader = self._create_dataloaders(sub_cfg)

                start_epoch_in_sub = max(0, self.start_epoch - cumulative_epochs)

                self._run_epochs(
                    stage_idx, start_epoch_in_sub, sub_epochs,
                    cumulative_epochs, total_epochs,
                    train_loader, val_loader, optimizer_G, optimizer_D, scheduler_G,
                    sub_idx
                )
                cumulative_epochs += sub_epochs
                self.start_epoch = 0 # Reset relative epoch counter for next sub-stage

        else: # Simple stage
            epochs = stage_cfg['epochs']
            logger.info(f"Simple stage: {epochs} epochs")
            train_loader, val_loader = self._create_dataloaders(stage_cfg)
            self._run_epochs(
                stage_idx, self.start_epoch, epochs,
                0, epochs, # epoch_offset is 0, total_epochs is just stage epochs
                train_loader, val_loader, optimizer_G, optimizer_D, scheduler_G,
                -1 # sub_stage_idx is -1 for simple stage
            )

    def _run_epochs(
        self,
        stage_idx: int,
        start_epoch: int,
        num_epochs: int,
        epoch_offset: int,
        total_epochs: int, # Total epochs in the *entire* stage (for logging)
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer_G: torch.optim.Optimizer,
        optimizer_D: torch.optim.Optimizer,
        scheduler_G: Optional[Any],
        sub_stage_idx: int = -1
    ):
        """Run training and validation loop for specified epochs within a stage/sub-stage."""
        for epoch_in_segment in range(start_epoch, num_epochs):
            # Calculate the absolute epoch number within the stage
            current_epoch = epoch_offset + epoch_in_segment

            # Train one epoch
            self._train_epoch(
                current_epoch, total_epochs, stage_idx,
                train_loader, optimizer_G, optimizer_D, scheduler_G,
                sub_stage_idx
            )

            # Perform validation and checkpointing
            self._validate_and_checkpoint(
                current_epoch, total_epochs, stage_idx,
                val_loader, optimizer_G, optimizer_D, scheduler_G
            )

    def _train_epoch(
        self,
        epoch: int, # Absolute epoch index within the current stage (0-based)
        total_epochs: int, # Total epochs for the current stage
        stage_idx: int,
        train_loader: DataLoader,
        optimizer_G: torch.optim.Optimizer,
        optimizer_D: torch.optim.Optimizer,
        scheduler_G: Optional[Any],
        sub_stage_idx: int = -1
    ):
        """Train Generator and Discriminator for one epoch."""
        self.generator.train()
        self.discriminator.train()

        # Setup progress bar description
        stage_desc = f"Stage {stage_idx+1}"
        if sub_stage_idx != -1:
            stage_desc += f".{sub_stage_idx+1}"
        pbar_desc = f"{stage_desc} | Epoch {epoch+1}/{total_epochs}" # Display 1-based epoch
        pbar = tqdm(train_loader, desc=pbar_desc, leave=True, dynamic_ncols=True)

        # Initialize epoch loss accumulators
        epoch_loss_G_total = 0.0
        epoch_loss_G_recon = 0.0
        epoch_loss_G_adv = 0.0
        epoch_loss_D_total = 0.0
        epoch_loss_D_real = 0.0
        epoch_loss_D_fake = 0.0
        epoch_recon_components = defaultdict(float)

        for batch_idx, batch in enumerate(pbar):
            # --- 1. Load Data ---
            try:
                lq = batch['lq'].to(self.device, non_blocking=True)
                hq = batch['hq'].to(self.device, non_blocking=True)
                crf = batch['crf'].to(self.device, non_blocking=True)
                preset = None
                if self.model_needs_preset:
                    preset = batch['preset'].to(self.device, non_blocking=True)
            except Exception as e:
                logger.error(f"Error loading batch {batch_idx}: {e}")
                continue # Skip batch on error

            # --- Determine label shapes once ---
            # Needed because discriminator output shape depends on architecture and input size
            if self.adv_label_shape is None:
                try:
                    with torch.no_grad(), torch.amp.autocast(device_type=str(self.device.type), enabled=self.use_amp):
                         # Use a small slice to determine output shape efficiently
                         dummy_pred = self.discriminator(hq[:1])
                         # Store shape excluding batch dimension
                         self.adv_label_shape = dummy_pred.shape[1:]
                    logger.info(f"Determined Discriminator output patch shape: {self.adv_label_shape}")
                except Exception as e:
                    logger.error(f"Could not determine Discriminator output shape: {e}", exc_info=True)
                    raise # Cannot proceed without label shape

            # Create labels for adversarial loss (LSGAN: real=1, fake=0) matching D output shape
            current_batch_size = lq.size(0)
            real_labels = torch.ones(current_batch_size, *self.adv_label_shape, device=self.device)
            fake_labels = torch.zeros(current_batch_size, *self.adv_label_shape, device=self.device)

            # --- 2. Train Discriminator ---
            # Enable grads for D, disable for G
            for p in self.discriminator.parameters():
                p.requires_grad = True
            for p in self.generator.parameters():
                p.requires_grad = False

            optimizer_D.zero_grad(set_to_none=True)

            # Generate fake image within autocast context and detach
            with torch.amp.autocast(device_type=str(self.device.type), enabled=self.use_amp):
                if self.model_needs_preset:
                    fake_hq = self.generator(lq, crf, preset).detach()
                else:
                    fake_hq = self.generator(lq, crf).detach()

            # Real forward + loss
            with torch.amp.autocast(device_type=str(self.device.type), enabled=self.use_amp):
                pred_real = self.discriminator(hq)
                loss_D_real = self.adversarial_loss_fn(pred_real, real_labels)

            # Fake forward + loss
            with torch.amp.autocast(device_type=str(self.device.type), enabled=self.use_amp):
                pred_fake = self.discriminator(fake_hq) # Use detached fake
                loss_D_fake = self.adversarial_loss_fn(pred_fake, fake_labels)

            # Combine, backward, and step for D
            loss_D = (loss_D_real + loss_D_fake) * 0.5

            if self.scaler: # CUDA AMP path for Discriminator backward
                self.scaler.scale(loss_D).backward()
                # Optional: Clip D grads if needed (usually not necessary)
                # self.scaler.unscale_(optimizer_D)
                # torch.nn.utils.clip_grad_norm_(self.discriminator.parameters(), max_norm=1.0)
                self.scaler.step(optimizer_D)
                # Scaler update happens after G step
            else: # Non-CUDA or no AMP path for Discriminator backward
                loss_D.backward()
                # Optional: Clip D grads if needed
                # torch.nn.utils.clip_grad_norm_(self.discriminator.parameters(), max_norm=1.0)
                optimizer_D.step()

            # --- 3. Train Generator ---
            # Enable grads for G, disable for D
            for p in self.discriminator.parameters():
                p.requires_grad = False
            for p in self.generator.parameters():
                p.requires_grad = True

            optimizer_G.zero_grad(set_to_none=True)

            # Forward pass G + D and loss calculation within autocast
            with torch.amp.autocast(device_type=str(self.device.type), enabled=self.use_amp):
                # Regenerate fake image to keep it in G's computation graph
                if self.model_needs_preset:
                    fake_hq_for_G = self.generator(lq, crf, preset)
                else:
                    fake_hq_for_G = self.generator(lq, crf)

                # Adversarial loss (G tries to make D output 1 for fake images)
                pred_fake_for_G = self.discriminator(fake_hq_for_G)
                loss_G_adv = self.adversarial_loss_fn(pred_fake_for_G, real_labels)

                # Reconstruction loss
                loss_G_recon, recon_loss_dict = self.reconstruction_loss_fn(fake_hq_for_G, hq)

                # Combined Generator loss
                loss_G = (loss_G_recon * self.lambda_recon) + (loss_G_adv * self.lambda_adv)

            # Backward and step for G
            if self.scaler: # CUDA AMP path for Generator backward
                self.scaler.scale(loss_G).backward()
                # Clip grads for G before step
                grad_clip_G = self.config['optimizer_G'].get('grad_clip_norm', 0.0)
                if grad_clip_G > 0:
                    self.scaler.unscale_(optimizer_G) # Unscale before clipping
                    torch.nn.utils.clip_grad_norm_(self.generator.parameters(), grad_clip_G)
                self.scaler.step(optimizer_G)
                # Update scaler ONCE per iteration, after both G and D steps
                self.scaler.update()
            else: # Non-CUDA or no AMP path for Generator backward
                loss_G.backward()
                # Clip grads for G before step
                grad_clip_G = self.config['optimizer_G'].get('grad_clip_norm', 0.0)
                if grad_clip_G > 0:
                    torch.nn.utils.clip_grad_norm_(self.generator.parameters(), grad_clip_G)
                optimizer_G.step()

            # --- 4. Updates & Logging ---
            if scheduler_G:
                scheduler_G.step()
            if self.ema_G:
                self.ema_G.update()

            # Accumulate losses for epoch average
            epoch_loss_D_real += loss_D_real.item()
            epoch_loss_D_fake += loss_D_fake.item()
            epoch_loss_D_total += loss_D.item()
            epoch_loss_G_adv += loss_G_adv.item()
            epoch_loss_G_recon += loss_G_recon.item()
            epoch_loss_G_total += loss_G.item() # Combined G loss
            for k, v in recon_loss_dict.items():
                epoch_recon_components[k] += v.item()

            # Update progress bar
            lr_g = optimizer_G.param_groups[0]['lr']
            pbar.set_postfix(
                G_Loss=f"{loss_G.item():.3f}",
                D_Loss=f"{loss_D.item():.3f}",
                Adv=f"{loss_G_adv.item():.3f}",
                Recon=f"{loss_G_recon.item():.3f}",
                LR_G=f"{lr_g:.2e}",
                refresh=(batch_idx % 10 == 0) # Reduce refresh rate
            )

            # W&B step logging
            if WANDB_AVAILABLE and wandb.run:
                log_freq = self.config['training'].get('log_every_n_steps', 50)
                if self.global_step % log_freq == 0:
                    wandb_log = {
                        'train/loss_G_total': loss_G.item(),
                        'train/loss_G_recon': loss_G_recon.item(),
                        'train/loss_G_adv': loss_G_adv.item(),
                        'train/loss_D_total': loss_D.item(),
                        'train/loss_D_real': loss_D_real.item(),
                        'train/loss_D_fake': loss_D_fake.item(),
                        'train/lr_G': lr_g,
                        **{f'train/recon_{k.replace("loss_", "")}': v.item()
                           for k, v in recon_loss_dict.items()}
                    }
                    wandb.log(wandb_log, step=self.global_step)

            self.global_step += 1

        # --- Epoch End Logging ---
        num_batches = len(train_loader)
        if num_batches > 0:
            avg_loss_G = epoch_loss_G_total / num_batches
            avg_loss_D = epoch_loss_D_total / num_batches
            avg_recon_comps = {k: v / num_batches for k,v in epoch_recon_components.items()}

            logger.info(f"Epoch {epoch+1}/{total_epochs} | Avg G Loss: {avg_loss_G:.4f} | Avg D Loss: {avg_loss_D:.4f}")

            if WANDB_AVAILABLE and wandb.run:
                wandb_log = {
                    'epoch/avg_loss_G_total': avg_loss_G,
                    'epoch/avg_loss_G_recon': epoch_loss_G_recon / num_batches,
                    'epoch/avg_loss_G_adv': epoch_loss_G_adv / num_batches,
                    'epoch/avg_loss_D_total': avg_loss_D,
                    'epoch/avg_loss_D_real': epoch_loss_D_real / num_batches,
                    'epoch/avg_loss_D_fake': epoch_loss_D_fake / num_batches,
                    **{f'epoch/avg_recon_{k.replace("loss_", "")}': v
                       for k, v in avg_recon_comps.items()}
                }
                wandb.log(wandb_log, step=self.global_step)
        else:
            logger.warning("Train loader was empty for this epoch.")

    def _validate_and_checkpoint(
        self,
        epoch: int,
        total_epochs: int,
        stage_idx: int,
        val_loader: DataLoader,
        optimizer_G: torch.optim.Optimizer,
        optimizer_D: torch.optim.Optimizer,
        scheduler_G: Optional[Any]
    ):
        """Run validation (on Generator) and save checkpoints based on config."""
        val_freq = self.config['training'].get('validate_every_n_epochs', 1)
        is_last_epoch = (epoch == total_epochs - 1)
        is_best = False

        # Run validation if frequency met or last epoch
        if val_freq > 0 and ((epoch + 1) % val_freq == 0 or is_last_epoch):
            # Validation only evaluates the generator's reconstruction performance
            val_loss = self._validate(epoch, total_epochs, stage_idx, val_loader)

            # Check if new best based on validation (reconstruction) loss
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                is_best = True
                logger.info(f"✨ New best validation loss (Generator): {val_loss:.6f}")
                self._save_checkpoint(
                    epoch, stage_idx, optimizer_G, optimizer_D, scheduler_G,
                    is_best=True
                )

            # Save latest checkpoint after every validation run (if not already saved as best)
            if not is_best:
                self._save_checkpoint(
                    epoch, stage_idx, optimizer_G, optimizer_D, scheduler_G
                )

        # Periodic checkpoint (independent of validation frequency)
        save_freq = self.config['checkpoint'].get('save_every_n_epochs', 0)
        if save_freq > 0 and (epoch + 1) % save_freq == 0:
            # Avoid saving 'latest' again if validation just ran and saved it
            validation_ran_this_epoch = (val_freq > 0 and ((epoch + 1) % val_freq == 0 or is_last_epoch))
            if not validation_ran_this_epoch:
                self._save_checkpoint(
                    epoch, stage_idx, optimizer_G, optimizer_D, scheduler_G
                )

    @torch.no_grad()
    def _validate(
        self,
        epoch: int,
        total_epochs: int,
        stage_idx: int,
        val_loader: DataLoader
    ) -> float:
        """ Run validation on the Generator only, evaluating reconstruction performance. """
        self.generator.eval() # Set generator to eval mode
        if self.ema_G:
            self.ema_G.apply_shadow() # Apply EMA weights for generator

        # Accumulators for validation metrics
        total_recon_loss = 0.0 # Primary metric for validation
        recon_loss_components = defaultdict(float)
        baseline_l1, baseline_l2 = 0.0, 0.0
        restored_l1, restored_l2 = 0.0, 0.0

        pbar = tqdm(
            val_loader,
            desc=f"Validating Gen Epoch {epoch+1}/{total_epochs}",
            leave=False, # Don't leave progress bar after completion
            dynamic_ncols=True
        )

        for batch_idx, batch in enumerate(pbar):
            try:
                # Load data
                lq = batch['lq'].to(self.device, non_blocking=True)
                hq = batch['hq'].to(self.device, non_blocking=True)
                crf = batch['crf'].to(self.device, non_blocking=True)
                preset = None
                if self.model_needs_preset:
                    preset = batch['preset'].to(self.device, non_blocking=True)

                # Use generator's inference method if available, else forward
                inference_fn = getattr(self.generator, 'inference', self.generator.forward)

                # Run generator inference with autocast
                with torch.amp.autocast(device_type=str(self.device.type), enabled=self.use_amp):
                    if self.model_needs_preset:
                        restored = inference_fn(lq, crf, preset)
                    else:
                        restored = inference_fn(lq, crf)

                    # Calculate reconstruction loss only
                    recon_loss, recon_loss_dict = self.reconstruction_loss_fn(restored, hq)

                # Accumulate reconstruction loss and components
                total_recon_loss += recon_loss.item()
                for k, v in recon_loss_dict.items():
                    recon_loss_components[k] += v.item()

                # Accumulate baseline (L1/L2 vs LQ) and restored (L1/L2 vs HQ) metrics
                baseline_l1 += F.l1_loss(lq, hq).item()
                baseline_l2 += F.mse_loss(lq, hq).item()
                restored_l1 += F.l1_loss(restored, hq).item()
                restored_l2 += F.mse_loss(restored, hq).item()

            except Exception as e:
                logger.error(f"Error during validation batch {batch_idx}: {e}", exc_info=True)
                continue # Skip batch on error

        # Calculate averages
        num_batches = len(val_loader)
        if num_batches == 0:
            logger.warning("Validation loader was empty.")
            # Ensure EMA is restored even if validation fails early
            if self.ema_G:
                self.ema_G.restore()
            return float('inf') # Return infinity to indicate validation failure

        avg_loss = total_recon_loss / num_batches # Use reconstruction loss as the primary validation metric
        avg_comps = {k: v / num_batches for k, v in recon_loss_components.items()}
        avg_base_l1 = baseline_l1 / num_batches
        avg_base_l2 = baseline_l2 / num_batches
        avg_rest_l1 = restored_l1 / num_batches
        avg_rest_l2 = restored_l2 / num_batches

        # Calculate improvement percentages
        epsilon = 1e-9
        l1_imp = ((avg_base_l1 - avg_rest_l1) / (avg_base_l1 + epsilon)) * 100
        l2_imp = ((avg_base_l2 - avg_rest_l2) / (avg_base_l2 + epsilon)) * 100

        # Console logging focuses on Generator performance
        logger.info(f"📊 Validation Gen Epoch {epoch+1}/{total_epochs}:")
        logger.info(f"    Avg Recon Loss: {avg_loss:.6f}")
        logger.info(f"    L1: {avg_rest_l1:.6f} (Baseline: {avg_base_l1:.6f}, ↓ {l1_imp:.2f}%)")
        logger.info(f"    L2: {avg_rest_l2:.6f} (Baseline: {avg_base_l2:.6f}, ↓ {l2_imp:.2f}%)")

        # W&B logging focuses on Generator performance
        if WANDB_AVAILABLE and wandb.run:
            wandb_dict = {
                'val/epoch': epoch + 1,
                'val/stage': stage_idx + 1,
                'val/avg_recon_loss': avg_loss,
                **{f'val/avg_recon_{k.replace("loss_", "")}': v for k, v in avg_comps.items()},
                'val/baseline_l1': avg_base_l1,
                'val/baseline_l2': avg_base_l2,
                'val/restored_l1': avg_rest_l1,
                'val/restored_l2': avg_rest_l2,
                'val/improvement_l1': l1_imp,
                'val/improvement_l2': l2_imp
            }
            # Log visual samples if available
            if self.val_samples:
                try:
                    self._log_visual_samples(epoch + 1, stage_idx + 1)
                except Exception as e:
                    logger.warning(f"Failed to log visual samples: {e}")

            wandb.log(wandb_dict, step=self.global_step)

        # Restore original generator weights if EMA was applied
        if self.ema_G:
            self.ema_G.restore()

        return avg_loss # Return reconstruction loss for best model tracking

    @torch.no_grad()
    def _log_visual_samples(self, epoch_num: int, stage_num: int):
        """Generate and log visual comparison samples using the Generator to W&B."""
        if not self.val_samples:
            logger.debug("No validation samples available to log.")
            return

        self.generator.eval() # Ensure generator is in eval mode
        if self.ema_G:
            self.ema_G.apply_shadow() # Use EMA weights for visuals if enabled

        lq = self.val_samples['lq']
        hq = self.val_samples['hq']
        num_samples = lq.size(0)

        inference_fn = getattr(self.generator, 'inference', self.generator.forward)

        restored_list = []
        for i in range(num_samples):
            lq_i = lq[i:i+1] # Keep batch dim
            try:
                # Run inference with autocast
                with torch.amp.autocast(device_type=str(self.device.type), enabled=self.use_amp):
                    crf_i = self.val_samples['crf'][i:i+1]
                    if self.model_needs_preset:
                        preset_i = self.val_samples['preset'][i:i+1]
                        restored_i = inference_fn(lq_i, crf_i, preset_i)
                    else:
                        restored_i = inference_fn(lq_i, crf_i)
                restored_list.append(restored_i)
            except Exception as e:
                logger.warning(f"Error during inference for visual sample {i}: {e}")
                continue # Skip failed samples

        if not restored_list:
            logger.warning("No visual samples were successfully generated.")
            if self.ema_G: self.ema_G.restore() # Ensure EMA restored if we exit early
            return

        restored = torch.cat(restored_list, dim=0)

        # Denormalize images for display (assuming [-1, 1] or [0, 1])
        norm_range = tuple(self.config['dataset'].get('norm_range', [-1, 1]))
        if norm_range == (-1, 1):
            denorm = lambda t: (t.float() + 1.0) / 2.0
            lq_display = denorm(lq[:restored.size(0)]) # Match batch size if some failed
            restored_display = denorm(restored)
            hq_display = denorm(hq[:restored.size(0)])
        else: # Assume [0, 1]
            lq_display = lq[:restored.size(0)].float()
            restored_display = restored.float()
            hq_display = hq[:restored.size(0)].float()

        # Clamp results to [0, 1] after denormalization
        lq_display.clamp_(0, 1)
        restored_display.clamp_(0, 1)
        hq_display.clamp_(0, 1)

        # Create list of wandb.Image objects
        images_to_log = []
        actual_samples_logged = restored_display.size(0)
        for i in range(actual_samples_logged):
            base_name = self.val_samples.get('base_name', [""] * actual_samples_logged)[i]
            crf_str, preset_str = "", ""
            try: crf_str = f"CRF{int(self.val_samples['crf'][i].item())}"
            except: pass # Ignore errors if CRF is not scalar

            if self.model_needs_preset and 'preset' in self.val_samples:
                try: preset_str = f" P{int(self.val_samples['preset'][i].item())}"
                except: pass

            caption = f"S{stage_num} E{epoch_num} | {base_name} {crf_str}{preset_str} | LQ-Restored-HQ"

            # Create horizontal strip: LQ | Restored | HQ
            try:
                img_strip = torch.cat([lq_display[i], restored_display[i], hq_display[i]], dim=2) # Concat width-wise
                images_to_log.append(wandb.Image(img_strip, caption=caption))
            except Exception as e:
                logger.warning(f"Error creating image strip for sample {i}: {e}")
                continue

        # Log to W&B
        if images_to_log:
            wandb.log({"Validation Samples": images_to_log}, step=self.global_step)
        else:
            logger.warning("No visual samples were successfully processed for logging.")

        # Restore original generator weights if EMA was applied
        if self.ema_G:
            self.ema_G.restore()

    # --------------------------------------------------------------------------
    # Checkpointing Logic (Handles G and D states)
    # --------------------------------------------------------------------------
    def _get_checkpoint_path(self, key: Optional[str]) -> Optional[Path]:
        """Resolve checkpoint key to absolute file path."""
        if not key:
            return None

        key_path = Path(key).expanduser()
        if key_path.is_file():
            return key_path.resolve()

        if not self.checkpoint_dir:
            logger.error("Checkpoint directory not configured.")
            return None

        key_lower = key.lower()
        if key_lower == 'latest':
            return (self.checkpoint_dir / 'latest.pth').resolve()
        if key_lower == 'best':
            return (self.checkpoint_dir / 'best.pth').resolve()

        # Assume filename within checkpoint directory
        filename = key_path.name
        if not filename.endswith('.pth'):
            filename += '.pth'
        potential_path = (self.checkpoint_dir / filename).resolve()

        if potential_path.is_file():
            return potential_path
        else:
            logger.warning(f"Checkpoint key '{key}' resolved to non-existent file: {potential_path}")
            return None

    def _save_checkpoint(
        self,
        epoch, stage_idx, optimizer_G, optimizer_D, scheduler_G,
        is_best=False, is_stage_complete=False, is_emergency=False
    ):
        """Save checkpoint with G, D, optimizers, scheduler_G, EMA_G."""
        if not self.checkpoint_dir:
            logger.error("Checkpoint directory not set, cannot save.")
            return

        state = {
            'config': self.config,
            'generator_state_dict': self.generator.state_dict(),
            'discriminator_state_dict': self.discriminator.state_dict(),
            'stage_idx': stage_idx, 'epoch': epoch, 'global_step': self.global_step,
            'best_val_loss': self.best_val_loss, 'wandb_id': self.wandb_id,
            # Only save optimizer state if available (might be None in emergency)
            'optimizer_G_state_dict': optimizer_G.state_dict() if optimizer_G else None,
            'optimizer_D_state_dict': optimizer_D.state_dict() if optimizer_D else None,
        }
        if scheduler_G:
            state['scheduler_G_state_dict'] = scheduler_G.state_dict()
        if self.ema_G:
            state['ema_G_state_dict'] = self.ema_G.shadow

        # Determine filename based on flags
        save_path, log_prefix = None, ""
        if is_emergency:
            save_path, log_prefix = self.checkpoint_dir / "emergency_interrupt.pth", "🚨 Emergency"
        elif is_best:
            save_path, log_prefix = self.checkpoint_dir / "best.pth", "✨ Best (Gen)"
        elif is_stage_complete:
            save_path, log_prefix = self.checkpoint_dir / f"stage_{stage_idx+1:02d}_complete.pth", f"🏁 Stage {stage_idx+1}"
        else: # Default is latest
            save_path, log_prefix = self.checkpoint_dir / "latest.pth", "Latest"

        # Save the primary checkpoint file
        try:
            save_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(state, save_path)
            if is_best or is_stage_complete or is_emergency:
                logger.info(f"{log_prefix} checkpoint saved: {save_path.name}")
            else: # Log latest saves at debug level to reduce noise
                logger.debug(f"Saved latest checkpoint: {save_path.name}")
        except Exception as e:
            logger.error(f"Failed to save {log_prefix} checkpoint: {e}", exc_info=True)
            return # Don't try to update latest if primary save failed

        # Update latest.pth to point to this save (unless it was an emergency save)
        if not is_emergency and save_path != (self.checkpoint_dir / "latest.pth"):
            latest_path = self.checkpoint_dir / "latest.pth"
            try:
                # Overwrite latest.pth with the current state (safer than symlink)
                torch.save(state, latest_path)
                logger.debug(f"Updated latest checkpoint pointer to {save_path.name}")
            except Exception as e:
                logger.error(f"Failed to update latest checkpoint file: {e}", exc_info=True)

    def _load_checkpoint(self):
        """Load G, D, optimizers, scheduler_G, EMA_G from checkpoint."""
        if not self.resume_from:
            return

        ckpt_path = None
        try:
            ckpt_path = self._get_checkpoint_path(self.resume_from)
            if not ckpt_path or not ckpt_path.exists():
                logger.error(f"Resume checkpoint not found: {ckpt_path}")
                logger.warning("Starting training from scratch.")
                self.resume_from = None
                return

            logger.info(f"Loading checkpoint: {ckpt_path}")
            # Load onto the correct device, allow loading full pickle objects
            ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=False)

            # --- Restore Generator ---
            if 'generator_state_dict' not in ckpt:
                logger.error("Checkpoint missing generator_state_dict")
                raise ValueError("Invalid checkpoint")
            try:
                missing, unexpected = self.generator.load_state_dict(ckpt['generator_state_dict'], strict=False)
                if unexpected: logger.warning(f"Generator - Unexpected keys: {unexpected}")
                if missing: logger.warning(f"Generator - Missing keys: {missing}")
                logger.info("Generator state restored")
            except Exception as e:
                logger.error(f"Error loading generator state: {e}")
                raise

            # --- Restore Discriminator ---
            if 'discriminator_state_dict' in ckpt:
                try:
                    missing, unexpected = self.discriminator.load_state_dict(ckpt['discriminator_state_dict'], strict=False)
                    if unexpected: logger.warning(f"Discriminator - Unexpected keys: {unexpected}")
                    if missing: logger.warning(f"Discriminator - Missing keys: {missing}")
                    logger.info("Discriminator state restored")
                except Exception as e:
                    logger.warning(f"Could not load discriminator state: {e}")
            else:
                logger.warning("Checkpoint missing discriminator_state_dict. Initializing discriminator from scratch.")

            # --- Restore Training State ---
            saved_stage = ckpt.get('stage_idx', 0)
            saved_epoch = ckpt.get('epoch', -1) # Epoch index completed (0-based)
            self.global_step = ckpt.get('global_step', 0)
            self.best_val_loss = ckpt.get('best_val_loss', float('inf'))

            self.start_stage = saved_stage
            self.start_epoch = saved_epoch + 1 # Resume from the next epoch

            # Check if stage was completed
            try:
                current_stage_config = self.config['curriculum'][self.start_stage]
                epochs_config = current_stage_config['epochs']
                total_epochs_in_stage = sum(epochs_config) if isinstance(epochs_config, list) else epochs_config
                if self.start_epoch >= total_epochs_in_stage:
                    logger.info(f"Checkpoint indicates Stage {self.start_stage + 1} completed.")
                    self.start_stage += 1
                    self.start_epoch = 0
                    logger.info(f"Resuming from the beginning of Stage {self.start_stage + 1}.")
            except IndexError:
                logger.error(f"Checkpoint stage index {saved_stage} out of bounds for curriculum.")
                raise ValueError("Checkpoint incompatible with current curriculum.")
            except Exception as e:
                logger.error(f"Error processing stage/epoch for resume: {e}", exc_info=True)
                # Continue with loaded start_stage/start_epoch as best guess

            # Check if training already finished
            if self.start_stage >= len(self.config['curriculum']):
                logger.warning(f"Checkpoint indicates training already completed after Stage {self.start_stage}.")
                sys.exit(0)

            # --- Restore W&B ID ---
            ckpt_wandb = ckpt.get('wandb_id')
            if ckpt_wandb and not self.wandb_id:
                self.wandb_id = ckpt_wandb
                logger.info(f"Resuming W&B run ID from checkpoint: {self.wandb_id}")
            elif ckpt_wandb and self.wandb_id and ckpt_wandb != self.wandb_id:
                logger.warning(f"W&B ID mismatch! CLI: {self.wandb_id}, Checkpoint: {ckpt_wandb}. Using CLI ID.")

            # --- Store Optimizer/Scheduler States ---
            self.saved_optimizer_G_state = ckpt.get('optimizer_G_state_dict')
            self.saved_optimizer_D_state = ckpt.get('optimizer_D_state_dict')
            self.saved_scheduler_G_state = ckpt.get('scheduler_G_state_dict')
            if self.saved_optimizer_G_state: logger.info("Optimizer G state found.")
            if self.saved_optimizer_D_state: logger.info("Optimizer D state found.")
            if self.saved_scheduler_G_state: logger.info("Scheduler G state found.")

            # --- Restore EMA G ---
            if self.ema_G:
                ema_state = ckpt.get('ema_G_state_dict')
                if ema_state:
                    try:
                        if len(ema_state) == len(self.ema_G.shadow):
                            self.ema_G.shadow = ema_state
                            logger.info("EMA G state restored.")
                        else:
                            logger.warning("EMA G state keys mismatch. Resetting EMA G.")
                            self.ema_G = EMA(self.generator, decay=self.ema_G.decay) # Reinitialize
                    except Exception as e:
                        logger.warning(f"Failed to restore EMA G state: {e}. EMA G will start fresh.")
                else:
                    logger.warning("EMA G enabled but no state found in checkpoint.")

            logger.info(f"Checkpoint loaded successfully. Resuming at Stage {self.start_stage + 1}, Epoch {self.start_epoch + 1}")
            logger.info(f"Global Step: {self.global_step}, Best Val Loss: {self.best_val_loss:.6f}")

        except FileNotFoundError:
             # Already handled by _get_checkpoint_path returning None
             pass
        except Exception as e:
            logger.error(f"Failed to load checkpoint '{ckpt_path}': {e}", exc_info=True)
            logger.warning("Starting training from scratch due to load error.")
            # Reset state variables
            self.start_stage, self.start_epoch, self.global_step, self.best_val_loss = 0, 0, 0, float('inf')
            self.saved_optimizer_G_state, self.saved_optimizer_D_state, self.saved_scheduler_G_state = None, None, None
            self.resume_from = None # Prevent re-attempting load

# ==============================================================================
# SECTION 3: Entry Point & Argument Parsing
# ==============================================================================
def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train Conditional AV1 U-Net GAN",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '--config', type=str, required=True,
        help="Path to YAML configuration file."
    )
    parser.add_argument(
        '--resume', type=str, default=None,
        help="Resume checkpoint (latest, best, or path)."
    )
    parser.add_argument(
        '--wandb_id', type=str, default=None,
        help="W&B run ID to resume logging."
    )
    return parser.parse_args()


def main():
    """Main execution function with error handling and emergency save."""
    args = parse_args()
    trainer = None # Initialize trainer variable

    try:
        # Initialize trainer (loads config, sets up system, models, etc.)
        trainer = ConditionalUNetGANTrainer(
            config_path=args.config,
            resume_from=args.resume,
            wandb_id=args.wandb_id
        )
        # Start the training loop
        trainer.train()

    except KeyboardInterrupt:
        logger.warning("\nTraining interrupted by user (KeyboardInterrupt).")
        if trainer: # Check if trainer initialized successfully
            logger.info("Attempting to save emergency checkpoint...")
            try:
                # Determine last completed epoch (0-based)
                epoch_completed = trainer.start_epoch - 1 if trainer.start_epoch > 0 else -1
                # Retrieve optimizers/scheduler safely
                opt_g = getattr(trainer, 'optimizer_G', None)
                opt_d = getattr(trainer, 'optimizer_D', None)
                sched_g = getattr(trainer, 'scheduler_G', None)
                # Save state, excluding optimizers/scheduler in emergency
                trainer._save_checkpoint(
                    epoch_completed, trainer.start_stage,
                    None, None, None, # Pass None for optimizers/scheduler
                    is_emergency=True
                )
            except Exception as e:
                logger.error(f"Failed to save emergency checkpoint during interrupt: {e}", exc_info=True)
        else:
            logger.warning("Trainer object not available, cannot save emergency checkpoint.")

        # Attempt clean W&B finish
        if WANDB_AVAILABLE and wandb.run:
            logger.info("Finishing W&B run (interrupted)...")
            wandb.finish(exit_code=130) # Use standard interrupt exit code
        sys.exit(130) # Exit script with interrupt code

    except Exception as e:
        # Catch any other unexpected error during training
        logger.error(f"Training failed due to unexpected error: {e}", exc_info=True)

        # Attempt emergency save if trainer and generator exist
        if trainer and hasattr(trainer, 'generator'):
            logger.info("Attempting to save emergency checkpoint due to error...")
            try:
                epoch_completed = trainer.start_epoch - 1 if trainer.start_epoch > 0 else -1
                opt_g = getattr(trainer, 'optimizer_G', None)
                opt_d = getattr(trainer, 'optimizer_D', None)
                sched_g = getattr(trainer, 'scheduler_G', None)
                trainer._save_checkpoint(
                    epoch_completed, trainer.start_stage,
                    None, None, None, # Pass None for optimizers/scheduler
                    is_emergency=True
                )
            except Exception as save_e:
                logger.error(f"Failed to save emergency checkpoint during error handling: {save_e}", exc_info=True)

        # Attempt clean W&B finish
        if WANDB_AVAILABLE and wandb.run:
            logger.info("Finishing W&B run (error)...")
            wandb.finish(exit_code=1) # Indicate general error
        sys.exit(1) # Exit script with non-zero code


if __name__ == "__main__":
    main()
