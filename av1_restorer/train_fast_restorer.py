# av1_restorer/train_fast_restorer.py
"""
Professional Training Script for the AURA-Net FAST Restorer.
Tailored for the small, non-conditional model.
"""
import sys
import yaml
import argparse
import logging
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.amp import autocast, GradScaler
from tqdm import tqdm

# Add project root to path for clean imports
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from av1_restorer.models.av1_fast_restorer import AV1FastRestorer # <-- Import fast model
from utils.losses import CombinedLoss
from utils.av1_dataset import AV1Dataset
from av1_restorer.train_av1_restorer import EMA, print_training_summary # <-- Reuse from main trainer

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

class FastTrainer:
    """A tailored, modular training class for the AV1FastRestorer."""
    def __init__(self, config_path: str, resume_from: str = None, wandb_id: str = None):
        # (Most of the __init__ and helper methods are identical to the main Trainer)
        # ... The code below is a simplified version focusing on the key changes ...
        self.config = yaml.safe_load(open(config_path, 'r'))
        self.resume_from = resume_from
        self.wandb_id = wandb_id
        
        # System Setup
        sys_cfg = self.config['system']
        self.device = torch.device(sys_cfg['device'])
        torch.manual_seed(sys_cfg['seed'])
        self.checkpoint_dir = Path(self.config['checkpoint']['dir'])
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.use_amp = sys_cfg['mixed_precision']
        self.scaler = GradScaler(enabled=self.use_amp)
        logger.info(f"System setup complete. Device: {self.device}")

        # Model, Loss, EMA
        self.model = self._setup_model()
        self.loss_fn = CombinedLoss(self.config['loss'])
        self.ema = EMA(self.model) if self.config['optimizer'].get('use_ema', False) else None

        # Logging and Summary
        self._setup_logging()
        print_training_summary(self.config, self.model)
        
        # State
        self.start_stage, self.start_epoch, self.global_step, self.best_val_loss = 0, 0, 0, float('inf')
        self.optimizer_state, self.scheduler_state = None, None
        
        if self.resume_from:
            self._load_checkpoint()

    def _setup_model(self) -> torch.nn.Module:
        """Instantiates the AV1FastRestorer based on the config."""
        model_params = self.config['model'].get('params', {})
        model = AV1FastRestorer(**model_params).to(self.device)
        logger.info(f"Model: AV1FastRestorer | Parameters: {sum(p.numel() for p in model.parameters()):,}")
        return model

    def _train_epoch(self, epoch, total_epochs, stage_idx, loader, optimizer, scheduler):
        self.model.train()
        pbar = tqdm(loader, desc=f"Stage {stage_idx+1} | Epoch {epoch+1}/{total_epochs}", leave=True)
        
        for batch in pbar:
            # We only need the LQ and HQ images for this model
            lq, hq = batch['lq'].to(self.device), batch['hq'].to(self.device)
            
            with autocast(device_type=str(self.device).split(':')[0], enabled=self.use_amp):
                # The model call is simpler: no CRF or preset needed
                restored_image = self.model(lq)
                loss, loss_dict = self.loss_fn(restored_image, hq)

            optimizer.zero_grad(set_to_none=True)
            self.scaler.scale(loss).backward()
            if self.config['training']['grad_clip_norm'] > 0:
                self.scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config['training']['grad_clip_norm'])
            self.scaler.step(optimizer)
            self.scaler.update()
            
            if scheduler: scheduler.step()
            if self.ema: self.ema.update()
            self.global_step += 1
            
            log_metrics = {f'{k.replace("loss_", "")}': f"{v.item():.4f}" for k, v in loss_dict.items()}
            pbar.set_postfix(log_metrics)
            
            if WANDB_AVAILABLE and self.config['project']['log_to_wandb']:
                wandb_log = {f'train/{k.replace("loss_", "")}': v.item() for k, v in loss_dict.items()}
                wandb_log['train/lr'] = optimizer.param_groups[0]['lr']
                wandb.log(wandb_log, step=self.global_step)

    @torch.no_grad()
    def _validate(self, epoch, loader):
        self.model.eval()
        if self.ema: self.ema.apply_shadow()
        total_loss = 0
        
        for batch in tqdm(loader, desc=f"Validating Epoch {epoch+1}", leave=False):
            lq, hq = batch['lq'].to(self.device), batch['hq'].to(self.device)
            restored_image = self.model(lq)
            loss, _ = self.loss_fn(restored_image, hq)
            total_loss += loss.item()

        avg_loss = total_loss / len(loader)
        logger.info(f"Validation Loss: {avg_loss:.4f}")

        if WANDB_AVAILABLE and self.config['project']['log_to_wandb']:
            wandb.log({'val/total_loss': avg_loss}, step=self.global_step)
            
        if self.ema: self.ema.restore()
        return avg_loss

    # ... The rest of the Trainer methods (_setup_logging, _create_dataloaders, etc.)
    # can be copied from `train_aura_net.py` as they are highly reusable. For brevity,
    # this example only shows the methods that required changes. A full implementation
    # would include all the robust checkpointing and setup logic from the main trainer.

def main():
    parser = argparse.ArgumentParser(description="Train AURA-Net FAST Model")
    parser.add_argument('--config', type=str, required=True, help="Path to the training configuration YAML file.")
    # ... add resume, wandb_id args ...
    args = parser.parse_args()
    
    # This is a simplified main loop. A full implementation would use the complete Trainer class.
    trainer = FastTrainer(config_path=args.config)
    # trainer.train() # This would start the full training process

if __name__ == "__main__":
    logger.info("This is a simplified example. For a full training run, please integrate the logic into the complete Trainer class from `train_aura_net.py`.")
    main()