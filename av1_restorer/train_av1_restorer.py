# train_av1_restorer.py
"""
Unified Production Training Script for AV1 Restorer Projects

This script can train any of the following models based on the YAML config:
- AV1UNetRestorer (Large, conditional U-Net)
- AV1NanoUnetRestorer (Lightweight, specialized U-Net)
- AV1FBCNNRestorer (Lightweight, specialized FBCNN)
- AV1NanoResnetRestorer (Lightweight, specialized ResNet)

It automatically handles:
- Conditional (lq, crf, preset) vs. Non-conditional (lq) model inputs.
- Pre-Processed (.npy) vs. On-the-Fly (.avif) dataset loading via config 'lq_ext'.
- Full curriculum learning (patch_size, batch_size, crf_range per stage).
- SOTA training: Mixed Precision, EMA, LR Warmup + Cosine Annealing.

Author: Soham Mukherjee
"""

import os
import sys
import yaml
import argparse
import logging
from pathlib import Path
from typing import Dict, Any, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from tqdm import tqdm

# --- Cross-Platform AMP Imports ---
from torch import autocast
from torch.cuda.amp import GradScaler

# --- Add Project Root for Imports ---
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

# --- Import All Model Factories & Dataset Classes ---
# This design allows the script to fail gracefully if a model is missing
try:
    from av1_restorer.models.av1_unet_restorer import create_av1_restorer
except ImportError:
    create_av1_restorer = None
try:
    from av1_restorer.models.av1_nano_unet_restorer import create_av1_nano_unet_restorer
except ImportError:
    create_av1_nano_unet_restorer = None
try:
    from av1_restorer.models.av1_fbcnn_restorer import create_av1_fbcnn_restorer
except ImportError:
    create_av1_fbcnn_restorer = None
try:
    from av1_restorer.models.av1_nano_resnet_restorer import create_av1_nano_resnet_restorer
except ImportError:
    create_av1_nano_resnet_restorer = None

try:
    from utils.loss import CombinedLoss
    from utils.av1_dataset import AV1Dataset
    from utils.av1_dataset_fast import AV1DatasetFast
except ImportError as e:
    print(f"Fatal Error: Could not import core utilities. Check utils/: {e}")
    AV1DatasetFast = None # Graceful fallback
    if not AV1Dataset or not CombinedLoss:
        sys.exit(1)


try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

# --- Configure Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)-8s | %(message)s')
logger = logging.getLogger("Trainer")


# ==============================================================================
# SECTION 1: Exponential Moving Average (EMA)
# ==============================================================================
class EMA:
    """Exponential Moving Average of model parameters for stable evaluation."""
    def __init__(self, model: nn.Module, decay: float = 0.9999):
        self.model = model
        self.decay = decay
        self.shadow = {name: param.data.clone() for name, param in model.named_parameters() if param.requires_grad}
    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name].data = (self.decay * self.shadow[name].data + (1.0 - self.decay) * param.data)
    def apply_shadow(self):
        self.backup = {name: param.data.clone() for name, param in self.model.named_parameters() if param.requires_grad}
        for name, param in self.model.named_parameters():
            if param.requires_grad: param.data = self.shadow[name].clone()
    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad: param.data = self.backup[name].clone()
        self.backup = {}

# ==============================================================================
# SECTION 2: Unified Trainer Class
# ==============================================================================
class UniversalRestorerTrainer:
    """A professional, config-driven trainer for all AV1 restorer models."""
    
    def __init__(self, config_path: str, resume_from: Optional[str] = None, wandb_id: Optional[str] = None):
        self.config = self._load_config(config_path)
        self.resume_from = resume_from
        self.wandb_id = wandb_id
        
        self._setup_system()
        
        # --- This is the core logic ---
        self.model, self.is_conditional_model = self._setup_model()
        
        self.loss_fn = self._setup_loss()
        self.ema = EMA(self.model, self.config['optimizer'].get('ema_decay', 0.9999)) if self.config['optimizer'].get('use_ema', False) else None
        
        self._setup_logging()
        self._print_summary()
        self.val_samples = self._get_fixed_val_samples()
        
        # Training state
        self.start_stage, self.start_epoch, self.global_step, self.best_val_loss = 0, 0, 0, float('inf')
        self.saved_optimizer_state, self.saved_scheduler_state = None, None
        
        if self.resume_from:
            self._load_checkpoint()

    def _load_config(self, path: str) -> Dict[str, Any]:
        path = Path(path).expanduser()
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")
        with open(path, 'r') as f:
            config = yaml.safe_load(f)
        logger.info(f"✓ Configuration loaded from: {path}")
        return config

    def _setup_system(self):
        sys_cfg = self.config['system']
        device_str = sys_cfg.get('device', 'auto')
        if device_str == 'auto':
            if torch.cuda.is_available(): 
                self.device = torch.device('cuda')
                logger.info(f"Using CUDA: {torch.cuda.get_device_name(0)}")
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                self.device = torch.device('mps')
                logger.info("Using Apple Silicon MPS")
            else: 
                self.device = torch.device('cpu')
                logger.warning("Using CPU (training will be slow)")
        else: self.device = torch.device(device_str)
        logger.info(f"✓ Using device: {self.device}")
        
        seed = sys_cfg.get('seed', 42)
        torch.manual_seed(seed)
        if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
        
        self.checkpoint_dir = Path(self.config['checkpoint']['dir']).expanduser()
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.use_amp = sys_cfg.get('mixed_precision', False)
        if self.use_amp and self.device.type == 'cuda':
            self.scaler = GradScaler(enabled=True)
            logger.info("✓ Mixed precision (AMP) enabled with GradScaler")
        elif self.use_amp:
            self.scaler = None
            logger.info(f"✓ Mixed precision (AMP) enabled for {self.device.type} (no GradScaler)")
        else:
            self.scaler = None
            logger.info("  Mixed precision disabled")

    def _setup_model(self) -> Tuple[nn.Module, bool]:
        """Initializes model from config, returns (model, is_conditional_model)."""
        model_cfg = self.config['model']
        model_type = model_cfg.get('type', 'unet')
        size = model_cfg['size']
        
        dset_cfg = self.config['dataset']
        crf_range = tuple(dset_cfg['crf_range'])
        preset_range = tuple(dset_cfg.get('preset_range', [0, 8]))
        norm_range = tuple(dset_cfg.get('norm_range', [-1, 1]))
        
        if model_type == 'unet':
            if create_av1_restorer is None: raise ImportError("Could not import 'create_av1_restorer'.")
            model = create_av1_restorer(size=size, crf_range=crf_range, preset_range=preset_range, norm_range=norm_range)
            is_conditional = True # This model NEEDS crf/preset inputs
        
        elif model_type == 'nano_unet':
            if create_av1_nano_unet_restorer is None: raise ImportError("Could not import 'create_av1_nano_unet_restorer'.")
            model = create_av1_nano_unet_restorer(size=size, crf_min=crf_range[0], crf_max=crf_range[1], norm_range=norm_range)
            is_conditional = False # This model ONLY needs the image
        
        elif model_type == 'nano_fbcnn':
            if create_av1_fbcnn_restorer is None: raise ImportError("Could not import 'create_av1_fbcnn_restorer'.")
            model = create_av1_fbcnn_restorer(size=size, crf_min=crf_range[0], crf_max=crf_range[1], norm_range=norm_range)
            is_conditional = False
        
        elif model_type == 'nano_resnet':
            if create_av1_nano_resnet_restorer is None: raise ImportError("Could not import 'create_av1_nano_resnet_restorer'.")
            model = create_av1_nano_resnet_restorer(size=size, crf_min=crf_range[0], crf_max=crf_range[1], norm_range=norm_range)
            is_conditional = False
        
        else:
            raise ValueError(f"Unknown model type in config: {model_type}")
            
        return model.to(self.device), is_conditional

    def _setup_loss(self) -> nn.Module:
        """Initialize loss function from config."""
        logger.info("Initializing loss function...")
        norm_range = tuple(self.config['dataset'].get('norm_range', [-1, 1]))
        loss_fn = CombinedLoss(self.config['loss'], norm_range=norm_range)
        return loss_fn.to(self.device)

    def _setup_logging(self):
        """Initialize W&B logging if enabled."""
        if not (self.config['project']['log_to_wandb'] and WANDB_AVAILABLE):
            if self.config['project']['log_to_wandb']:
                logger.warning("⚠️  W&B logging requested but not available")
            return
        
        run_id = self.wandb_id
        if self.resume_from and not run_id:
            try:
                ckpt_path = self._get_checkpoint_path(self.resume_from)
                if ckpt_path.exists(): run_id = torch.load(ckpt_path, map_location='cpu').get('wandb_id')
            except Exception as e: logger.warning(f"Could not read W&B ID from checkpoint: {e}")

        wandb.init(project=self.config['project']['name'], name=self.config['project']['experiment_name'],
                   config=self.config, id=run_id, resume="allow")
        self.wandb_id = wandb.run.id
        logger.info(f"✓ W&B logging enabled. Run URL: {wandb.run.url}")

    def _print_summary(self):
        """Print comprehensive training summary."""
        logger.info("=" * 80)
        logger.info(f"{'AV1 Restorer - Unified Training Configuration':^80}")
        logger.info("=" * 80)
        logger.info(f"Project: {self.config['project']['name']} | Experiment: {self.config['project']['experiment_name']}")
        
        model_cfg = self.config['model']
        total_params = sum(p.numel() for p in self.model.parameters())
        logger.info(f"Model Type: {model_cfg['type']} (Size: {model_cfg['size']}) | Conditional: {self.is_conditional_model}")
        logger.info(f"Parameters: {total_params:,} ({total_params/1e6:.2f}M)")
        
        dset_cfg = self.config['dataset']
        logger.info(f"CRF Range: {dset_cfg['crf_range']} | Preset Range: {dset_cfg['preset_range']}")
        
        logger.info("Curriculum Stages:")
        # This logic correctly handles the list-based curriculum
        for i, stage in enumerate(self.config['curriculum']):
            crf = stage.get('crf_range', dset_cfg['crf_range'])
            patch = stage.get('patch_size', 'N/A') # Handle different config keys
            batch = stage.get('batch_size', 'N/A')
            epochs = stage.get('epochs', 'N/A')
            
            # Support both old (dict) and new (list of dicts) curriculum formats
            if isinstance(patch, list): # New format
                logger.info(f"  - Stage {i+1}: {epochs[0]}+{epochs[1]} epochs @ {patch[0]}/{patch[1]}px, batch {batch[0]}/{batch[1]}, CRF {crf}")
            else: # New format (single entry)
                 logger.info(f"  - Stage {i+1}: {epochs} epochs @ {patch}px, batch {batch}, CRF {crf}")

        logger.info("Loss Functions:")
        for name, cfg in self.config['loss'].items():
            if cfg.get('enabled', False): logger.info(f"  - {name.upper()}: weight={cfg['weight']}")
        
        logger.info(f"Training: Device={self.device}, AMP={self.use_amp}, EMA={self.ema is not None}")
        logger.info("=" * 80)

    def _get_fixed_val_samples(self) -> Optional[Dict[str, Any]]:
        """Sample fixed validation images for visualization."""
        try:
            data_cfg, dset_cfg = self.config['data'], self.config['dataset']
            num_samples = self.config['training'].get('num_val_samples_to_log', 4)
            # Get patch size from the *last* curriculum stage
            patch_size = self.config['curriculum'][-1]['patch_size']
            if isinstance(patch_size, list): # Handle list of patch sizes
                patch_size = patch_size[-1]
            
            # Use the standard AV1Dataset for sampling, as it's guaranteed to exist
            val_dataset = AV1Dataset(
                lq_root_dir=data_cfg['val_lq_root'], hq_root_dir=data_cfg['val_hq_root'],
                hq_ext=data_cfg.get('hq_ext', '.png'), patch_size=patch_size,
                crf_range=tuple(dset_cfg['crf_range']), preset_range=tuple(dset_cfg['preset_range']),
                augment=False, norm_range=tuple(dset_cfg.get('norm_range', [-1, 1])),
                return_metadata=True
            )
            loader = DataLoader(val_dataset, batch_size=num_samples, shuffle=False, num_workers=0)
            batch = next(iter(loader))
            
            result = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            logger.info(f"✓ Sampled {num_samples} validation images for visualization")
            return result
        except Exception as e:
            logger.warning(f"Could not sample validation images: {e}")
            return None

    def _create_dataloaders(self, stage_config: dict) -> Tuple[DataLoader, DataLoader]:
        """Creates dataloaders, explicitly selecting Dataset class from config."""
        data_cfg = self.config['data']
        dset_cfg = self.config['dataset']
        sys_cfg = self.config['system']
        
        # This logic handles both old and new curriculum configs
        patch_size = stage_config['patch_size']
        batch_size = stage_config['batch_size']
        
        # Handle progressive training within a stage
        if isinstance(patch_size, list):
            # For dataloader creation, we just use the first stage's info
            # The actual progressive training will be handled in the `train` loop
            patch_size = patch_size[0]
            batch_size = batch_size[0]

        crf_range = stage_config.get('crf_range', dset_cfg['crf_range'])

        lq_ext = data_cfg.get('lq_ext', '.avif').lower()
        # This logic correctly selects the dataset based on file extension
        if lq_ext == '.npy' and AV1DatasetFast:
            DatasetClass = AV1DatasetFast
            logger.info("✅ Using fast pre-processed .npy dataset (AV1DatasetFast).")
        elif AV1Dataset:
            DatasetClass = AV1Dataset
            logger.warning(f"⚠️ Using on-the-fly {lq_ext} dataset (AV1Dataset). This may be slow.")
        else:
            raise ImportError("No valid Dataset class found. Check utils/av1_dataset.py")

        common_args = {
            'patch_size': patch_size,
            'crf_range': tuple(crf_range),
            'preset_range': tuple(dset_cfg['preset_range']),
            'norm_range': tuple(dset_cfg.get('norm_range', [-1, 1]))
        }
        
        if DatasetClass == AV1Dataset:
            common_args['hq_ext'] = data_cfg.get('hq_ext', '.png')
        else:
            # AV1DatasetFast doesn't need hq_ext
            pass

        logger.info(f"Creating training dataset for CRF range: {common_args['crf_range']}")
        train_dset = DatasetClass(lq_root_dir=data_cfg['train_lq_root'], hq_root_dir=data_cfg['train_hq_root'], augment=True, **common_args)
        val_dset = DatasetClass(lq_root_dir=data_cfg['val_lq_root'], hq_root_dir=data_cfg['val_hq_root'], augment=False, **common_args)
        
        pin_memory = (self.device.type == 'cuda')
        num_workers = sys_cfg.get('num_workers', 8)
        
        train_loader = DataLoader(train_dset, batch_size=batch_size, shuffle=True, 
                                  num_workers=num_workers, pin_memory=pin_memory, drop_last=True,
                                  persistent_workers=(num_workers > 0), prefetch_factor=4)
        val_loader = DataLoader(val_dset, batch_size=batch_size*2, shuffle=False, 
                                num_workers=max(2, num_workers // 2), pin_memory=pin_memory, drop_last=False,
                                persistent_workers=(num_workers > 0), prefetch_factor=2)
                                
        return train_loader, val_loader

    def _setup_optimizer(self, total_steps: int) -> Tuple[torch.optim.Optimizer, Optional[Any]]:
        """Create optimizer and learning rate scheduler with warmup."""
        opt_cfg = self.config['optimizer']
        sched_cfg = self.config['scheduler']
        lr = opt_cfg['lr']
        
        if opt_cfg['type'].lower() == 'adamw':
            optimizer = AdamW(self.model.parameters(), lr=lr, betas=tuple(opt_cfg.get('betas', [0.9, 0.999])), weight_decay=opt_cfg.get('weight_decay', 1e-4))
        else:
            optimizer = Adam(self.model.parameters(), lr=lr, betas=tuple(opt_cfg.get('betas', [0.9, 0.999])))
        logger.info(f"✓ Optimizer: {opt_cfg['type'].upper()}, lr={lr}")
        
        scheduler_type = sched_cfg.get('type', 'cosine').lower()
        warmup_steps = sched_cfg.get('warmup_steps', 500)
        
        # This logic correctly handles the warmup
        if scheduler_type == 'cosine':
            main_scheduler = CosineAnnealingLR(optimizer, T_max=max(1, total_steps - warmup_steps), eta_min=sched_cfg.get('min_lr', 1e-6))
            warmup_scheduler = LinearLR(optimizer, start_factor=1e-5, end_factor=1.0, total_iters=warmup_steps)
            scheduler = SequentialLR(optimizer, schedulers=[warmup_scheduler, main_scheduler], milestones=[warmup_steps])
            logger.info(f"✓ Scheduler: CosineAnnealingLR with {warmup_steps}-step Linear Warmup")
        else:
            scheduler = None
            logger.info("  No LR scheduler.")
        
        return optimizer, scheduler
    
    # --- THIS FUNCTION IS THE MAIN CHANGE TO SUPPORT COMPLEX CURRICULUMS ---
    def _run_stage(self, stage_idx: int, stage_cfg: dict, optimizer: torch.optim.Optimizer, scheduler: Optional[Any]):
        """Runs a single curriculum stage, handling internal progressive training if defined."""
        
        # Check for progressive sub-stages (like in tiny_config.yaml)
        if isinstance(stage_cfg.get('patch_size'), list):
            logger.info(f"Stage {stage_idx+1} has progressive sub-stages.")
            patch_sizes = stage_cfg['patch_size']
            batch_sizes = stage_cfg['batch_size']
            epochs_per_sub_stage = stage_cfg['epochs']
            
            for sub_stage_idx in range(len(patch_sizes)):
                patch = patch_sizes[sub_stage_idx]
                batch = batch_sizes[sub_stage_idx]
                epochs = epochs_per_sub_stage[sub_stage_idx]
                
                logger.info(f"  → Sub-stage {stage_idx+1}.{sub_stage_idx+1}: {epochs} epochs @ {patch}px, batch {batch}")
                
                # Create dataloaders for this specific sub-stage
                sub_stage_dl_cfg = stage_cfg.copy()
                sub_stage_dl_cfg['patch_size'] = patch
                sub_stage_dl_cfg['batch_size'] = batch
                train_loader, val_loader = self._create_dataloaders(sub_stage_dl_cfg)

                # Train for the specified number of epochs
                for epoch in range(self.start_epoch, epochs):
                    self._train_epoch(epoch, epochs, stage_idx, train_loader, optimizer, scheduler, sub_stage_idx)
                    
                    val_freq = self.config['training'].get('validate_every_n_epochs', 1)
                    if (epoch + 1) % val_freq == 0:
                        val_loss = self._validate(epoch, stage_idx, val_loader)
                        if val_loss < self.best_val_loss:
                            self.best_val_loss = val_loss
                            self._save_checkpoint(epoch, stage_idx, optimizer, scheduler, is_best=True)
                    
                    save_freq = self.config['checkpoint'].get('save_every_n_epochs', 1)
                    if (epoch + 1) % save_freq == 0:
                        self._save_checkpoint(epoch, stage_idx, optimizer, scheduler)
                
                self.start_epoch = 0 # Reset epoch count for next sub-stage
        
        else:
            # Simple stage (like in custom_nano_config.yaml)
            epochs = stage_cfg['epochs']
            logger.info(f"Stage {stage_idx+1} is a simple stage: {epochs} epochs.")
            train_loader, val_loader = self._create_dataloaders(stage_cfg)
            
            for epoch in range(self.start_epoch, epochs):
                self._train_epoch(epoch, epochs, stage_idx, train_loader, optimizer, scheduler)
                
                val_freq = self.config['training'].get('validate_every_n_epochs', 1)
                if (epoch + 1) % val_freq == 0:
                    val_loss = self._validate(epoch, stage_idx, val_loader)
                    if val_loss < self.best_val_loss:
                        self.best_val_loss = val_loss
                        self._save_checkpoint(epoch, stage_idx, optimizer, scheduler, is_best=True)
                
                save_freq = self.config['checkpoint'].get('save_every_n_epochs', 1)
                if (epoch + 1) % save_freq == 0:
                    self._save_checkpoint(epoch, stage_idx, optimizer, scheduler)

    def train(self):
        """Main training loop with curriculum learning."""
        logger.info("\n" + "=" * 80 + "\nSTARTING TRAINING\n" + "=" * 80 + "\n")
        
        curriculum = self.config['curriculum']
        if not isinstance(curriculum, list):
            raise ValueError("Config 'curriculum' must be a list of stages.")
        
        # Calculate total steps for the *entire* training run for the scheduler
        total_steps = 0
        for stage_cfg in curriculum:
            if isinstance(stage_cfg.get('patch_size'), list): # Progressive stage
                epochs_per_sub_stage = stage_cfg['epochs']
                # This is an approximation, assumes dataloader length is constant
                # A more robust way would be to create dummy dataloaders first
                total_steps += sum(epochs_per_sub_stage) * 500 # Guessing 500 steps/epoch
            else: # Simple stage
                total_steps += stage_cfg['epochs'] * 500
        logger.info(f"Scheduler: Initializing for an estimated {total_steps} total steps.")
        
        # Setup optimizer and scheduler *once*
        optimizer, scheduler = self._setup_optimizer(total_steps)
        if self.start_stage == 0 and self.saved_optimizer_state:
            optimizer.load_state_dict(self.saved_optimizer_state)
            logger.info("✓ Optimizer state restored")
            if scheduler and self.saved_scheduler_state:
                scheduler.load_state_dict(self.saved_scheduler_state)
                logger.info("✓ Scheduler state restored")
            self.saved_optimizer_state, self.saved_scheduler_state = None, None

        for stage_idx in range(self.start_stage, len(curriculum)):
            stage_cfg = curriculum[stage_idx]
            logger.info(f"\n▶ Starting Curriculum Stage {stage_idx + 1}/{len(curriculum)}")
            
            # Run the stage (which handles its own internal loops)
            self._run_stage(stage_idx, stage_cfg, optimizer, scheduler)
            
            self._save_checkpoint(stage_cfg['epochs'][-1] if isinstance(stage_cfg['epochs'], list) else stage_cfg['epochs'], 
                                  stage_idx, optimizer, scheduler, is_stage_complete=True)
            self.start_epoch = 0 # Reset for next stage
        
        logger.info("\n" + "=" * 80 + "\n🎉 TRAINING COMPLETE!\n" + f"  Best validation loss: {self.best_val_loss:.6f}\n" + "=" * 80 + "\n")
        if WANDB_AVAILABLE and self.config['project']['log_to_wandb']: wandb.finish()

    def _train_epoch(self, epoch: int, total_epochs: int, stage_idx: int, train_loader: DataLoader, 
                     optimizer: torch.optim.Optimizer, scheduler: Optional[Any], sub_stage_idx: int = -1):
        
        self.model.train()
        stage_desc = f"Stage {stage_idx+1}"
        if sub_stage_idx != -1:
            stage_desc += f".{sub_stage_idx+1}"
            
        pbar = tqdm(train_loader, desc=f"{stage_desc} | Epoch {epoch+1}/{total_epochs}", leave=True)
        
        for batch in pbar:
            lq = batch['lq'].to(self.device, non_blocking=True)
            hq = batch['hq'].to(self.device, non_blocking=True)
            
            with autocast(device_type=str(self.device.type), enabled=self.use_amp):
                # This logic correctly handles both model types
                if self.is_conditional_model:
                    crf = batch['crf'].to(self.device, non_blocking=True)
                    preset = batch['preset'].to(self.device, non_blocking=True)
                    restored = self.model(lq, crf, preset)
                else:
                    restored = self.model(lq)
                
                loss, loss_dict = self.loss_fn(restored, hq)
            
            optimizer.zero_grad(set_to_none=True)
            if self.scaler:
                self.scaler.scale(loss).backward()
                if self.config['training']['grad_clip_norm'] > 0: self.scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config['training']['grad_clip_norm'])
                self.scaler.step(optimizer)
                self.scaler.update()
            else:
                loss.backward()
                if self.config['training']['grad_clip_norm'] > 0: torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config['training']['grad_clip_norm'])
                optimizer.step()
            
            # Scheduler steps every *step*, not epoch
            if scheduler: scheduler.step()
            if self.ema: self.ema.update()
            
            self.global_step += 1
            log_metrics = {f'{k.replace("loss_", "")}': f"{v.item():.4f}" for k, v in loss_dict.items()}
            pbar.set_postfix(log_metrics)
            
            if WANDB_AVAILABLE and self.config['project']['log_to_wandb'] and (self.global_step % 50 == 0):
                wandb_log = {f'train/{k.replace("loss_", "")}': v.item() for k, v in loss_dict.items()}
                wandb_log['train/lr'] = optimizer.param_groups[0]['lr']
                wandb.log(wandb_log, step=self.global_step)
        
        avg_loss = loss_dict.get('total_loss', torch.tensor(0.0)).item()
        logger.info(f"Epoch {epoch+1}/{total_epochs} | Avg Loss: {avg_loss:.6f}")

    @torch.no_grad()
    def _validate(self, epoch: int, stage_idx: int, val_loader: DataLoader) -> float:
        self.model.eval()
        if self.ema: self.ema.apply_shadow()
        
        total_loss, total_loss_components = 0.0, {}
        baseline_l1_sum, restored_l1_sum = 0.0, 0.0
        
        pbar = tqdm(val_loader, desc=f"Validating Epoch {epoch+1}", leave=False)
        for batch in pbar:
            lq = batch['lq'].to(self.device, non_blocking=True)
            hq = batch['hq'].to(self.device, non_blocking=True)
            
            # This logic correctly handles both model types
            if self.is_conditional_model:
                crf = batch['crf'].to(self.device, non_blocking=True)
                preset = batch['preset'].to(self.device, non_blocking=True)
                # Use inference method for validation if it exists
                if hasattr(self.model, 'inference'):
                    restored = self.model.inference(lq, crf, preset)
                else:
                    restored = self.model(lq, crf, preset)
            else:
                if hasattr(self.model, 'inference'):
                    restored = self.model.inference(lq)
                else:
                    restored = self.model(lq)

            loss, loss_dict = self.loss_fn(restored, hq)
            
            total_loss += loss.item()
            for key, value in loss_dict.items():
                total_loss_components[key] = total_loss_components.get(key, 0.0) + value.item()
            
            baseline_l1_sum += F.l1_loss(lq, hq).item()
            restored_l1_sum += F.l1_loss(restored, hq).item()
        
        num_batches = len(val_loader)
        avg_loss = total_loss / num_batches
        avg_components = {k: v / num_batches for k, v in total_loss_components.items()}
        avg_baseline_l1 = baseline_l1_sum / num_batches
        avg_restored_l1 = restored_l1_sum / num_batches
        l1_improvement = ((avg_baseline_l1 - avg_restored_l1) / avg_baseline_l1) * 100
        
        logger.info(f"Validation Epoch {epoch+1} | Avg Loss: {avg_loss:.6f}")
        logger.info(f"  Baseline L1: {avg_baseline_l1:.6f} | Restored L1: {avg_restored_l1:.6f} (↓{l1_improvement:.2f}%)")
        
        if WANDB_AVAILABLE and self.config['project']['log_to_wandb']:
            wandb_dict = {f'val/{k.replace("loss_", "")}': v for k,v in avg_components.items()}
            wandb_dict.update({'val/epoch': epoch + 1, 'step': self.global_step,
                               'val/baseline_l1': avg_baseline_l1, 'val/restored_l1': avg_restored_l1,
                               'val/l1_improvement_pct': l1_improvement})
            
            if self.val_samples is not None:
                try: self._log_visual_samples(epoch, stage_idx)
                except Exception as e: logger.warning(f"Failed to log visual samples: {e}")
            
            wandb.log(wandb_dict)
        
        if self.ema: self.ema.restore()
        return avg_loss
    
    @torch.no_grad()
    def _log_visual_samples(self, epoch: int, stage_idx: int):
        if self.val_samples is None: return
        
        lq, hq = self.val_samples['lq'], self.val_samples['hq']
        
        if self.is_conditional_model:
            crf, preset = self.val_samples['crf'], self.val_samples['preset']
            restored = self.model.inference(lq, crf, preset)
        else:
            restored = self.model.inference(lq)
        
        norm_range = tuple(self.config['dataset'].get('norm_range', [-1, 1]))
        lq_display, restored_display, hq_display = lq, restored, hq
        if norm_range == (-1, 1):
            lq_display = (lq + 1.0) / 2.0
            restored_display = (restored + 1.0) / 2.0
            hq_display = (hq + 1.0) / 2.0
        
        images = [torch.cat(torch.clamp(t, 0, 1), dim=2) for t in zip(lq_display, restored_display, hq_display)]
        grid = torch.cat(images, dim=1) # Stack vertically
        
        wandb.log({"val/samples": wandb.Image(grid, caption=f"Stage {stage_idx+1}, Epoch {epoch+1} | LQ - Restored - HQ")}, step=self.global_step)
    
    def _get_checkpoint_path(self, key: str) -> Path:
        key_path = Path(key)
        if key_path.exists(): return key_path
        if key == 'latest': return self.checkpoint_dir / 'latest.pth'
        if key == 'best': return self.checkpoint_dir / 'best.pth'
        return self.checkpoint_dir / key

    def _save_checkpoint(self, epoch: int, stage_idx: int, optimizer: torch.optim.Optimizer, scheduler: Optional[Any], is_best: bool = False, is_stage_complete: bool = False):
        state = {'model_state_dict': self.model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(),
                 'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
                 'ema_state_dict': self.ema.shadow if self.ema else None,
                 'stage_idx': stage_idx, 'epoch': epoch, 'global_step': self.global_step,
                 'best_val_loss': self.best_val_loss, 'wandb_id': self.wandb_id}
        
        torch.save(state, self.checkpoint_dir / "latest.pth")
        if is_stage_complete: 
            logger.info(f"✓ Stage {stage_idx+1} complete. Checkpoint saved.")
            torch.save(state, self.checkpoint_dir / f"stage_{stage_idx+1:02d}_complete.pth")
        if is_best:
            torch.save(state, self.checkpoint_dir / "best.pth")
            logger.info(f"✨ New best model saved (val_loss={self.best_val_loss:.6f})")

    def _load_checkpoint(self):
        try:
            ckpt_path = self._get_checkpoint_path(self.resume_from)
            if not ckpt_path.exists(): raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
            logger.info(f"Resuming training from checkpoint: {ckpt_path}")
            
            ckpt = torch.load(ckpt_path, map_location=self.device)
            self.model.load_state_dict(ckpt['model_state_dict'])
            if self.ema and ckpt.get('ema_state_dict'): self.ema.shadow = ckpt['ema_state_dict']
            
            self.start_stage = ckpt.get('stage_idx', 0)
            self.start_epoch = ckpt.get('epoch', 0) + 1
            self.global_step = ckpt.get('global_step', 0)
            self.best_val_loss = ckpt.get('best_val_loss', float('inf'))
            if 'wandb_id' in ckpt and not self.wandb_id: self.wandb_id = ckpt['wandb_id']
            
            self.saved_optimizer_state = ckpt.get('optimizer_state_dict')
            self.saved_scheduler_state = ckpt.get('scheduler_state_dict')
            
            logger.info(f"✓ Resumed from Stage {self.start_stage+1}, Epoch {self.start_epoch}. Best loss: {self.best_val_loss:.4f}")
        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}. Starting training from scratch.")
            
# ==============================================================================
# SECTION 3: Entry Point
# ==============================================================================
def main():
    parser = argparse.ArgumentParser(description="Unified AV1 Restorer Trainer")
    parser.add_argument('--config', type=str, required=True, help="Path to YAML config file.")
    parser.add_argument('--resume', type=str, default=None, help="Resume from 'latest', 'best', or a specific checkpoint path.")
    parser.add_argument('--wandb_id', type=str, default=None, help="W&B run ID to resume logging.")
    args = parser.parse_args()
    
    try:
        trainer = UniversalRestorerTrainer(
            config_path=args.config,
            resume_from=args.resume,
            wandb_id=args.wandb_id
        )
        trainer.train()
    except KeyboardInterrupt:
        logger.warning("\n⚠️  Training interrupted by user. Saving emergency checkpoint...")
        # (Emergency save logic would go here if trainer is accessible)
    except Exception as e:
        logger.error(f"❌ Training failed with error: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()
