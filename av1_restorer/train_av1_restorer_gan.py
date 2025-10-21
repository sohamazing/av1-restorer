# av1_restorer/train_sv1_restorer_gan.py

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from pathlib import Path
import yaml
import argparse
from tqdm import tqdm
import wandb
import sys
import torchvision

# Add project root for clean imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from data.dataset import AV1Dataset
from models.fast_restorer import FastPathRestorer
from models.discriminator import PatchGANDiscriminator
from losses.restoration_loss import RestorationLoss

class EMA:
    """Exponential Moving Average of model weights for improved validation performance."""
    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.model = model
        self.decay = decay
        self.shadow = {name: param.data.clone() for name, param in model.named_parameters()}
        self.backup = {}

    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name].data = self.decay * self.shadow[name].data + (1 - self.decay) * param.data

    def apply_shadow(self):
        self.backup = {name: param.data.clone() for name, param in self.model.named_parameters()}
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data = self.shadow[name].clone()

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data = self.backup[name].clone()
        self.backup = {}


class GANTrainer:
    """Orchestrates the SOTA GAN training process for the AV1 Restorer."""
    def __init__(self, config: dict, resume_path: str = None, wandb_id: str = None):
        self.config = config
        self.device = self._setup_device()
        self.wandb_id = wandb_id
        
        # Init logging
        if config['log_to_wandb']:
            wandb.init(project=config['project_name'], config=config, id=wandb_id, resume="must" if wandb_id else None)
            self.wandb_id = wandb.run.id
        
        # Init Components
        self.generator = FastPathRestorer(**config['generator']).to(self.device)
        self.discriminator = PatchGANDiscriminator(**config['discriminator']).to(self.device)
        self.criterion_recon = RestorationLoss(device=self.device, **config['loss'])
        self.criterion_gan = nn.BCEWithLogitsLoss()
        self.criterion_crf = nn.MSELoss()
        self.optimizer_g = AdamW(self.generator.parameters(), lr=config['lr_g'], betas=(0.5, 0.999), weight_decay=config['weight_decay'])
        self.optimizer_d = AdamW(self.discriminator.parameters(), lr=config['lr_d'], betas=(0.5, 0.999))
        self.ema = EMA(self.generator)
        self.scaler = torch.cuda.amp.GradScaler(enabled=config['mixed_precision'] and self.device.type == 'cuda')
        
        self.checkpoint_dir = Path(config['checkpoint_save_path'])
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.start_epoch, self.start_stage, self.best_val_loss, self.global_step = 0, 0, float('inf'), 0
        if resume_path:
            self._load_checkpoint(resume_path)

    def _setup_device(self) -> torch.device:
        """Auto-detects and sets up the training device."""
        device_str = self.config.get('accelerator', 'auto').lower()
        if device_str == 'auto':
            if torch.cuda.is_available(): device_str = 'cuda'
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available(): device_str = 'mps'
            else: device_str = 'cpu'
        print(f"Using device: {device_str}")
        return torch.device(device_str)

    def _load_checkpoint(self, ckpt_path: str):
        """Loads a full training state from a checkpoint file."""
        print(f"Resuming from checkpoint: {ckpt_path}")
        checkpoint = torch.load(ckpt_path, map_location=self.device)
        self.generator.load_state_dict(checkpoint['generator'])
        self.discriminator.load_state_dict(checkpoint['discriminator'])
        self.optimizer_g.load_state_dict(checkpoint['optimizer_g'])
        self.optimizer_d.load_state_dict(checkpoint['optimizer_d'])
        self.ema.shadow = checkpoint['ema']
        self.start_epoch = checkpoint['epoch'] + 1
        self.start_stage = checkpoint['stage']
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        self.global_step = checkpoint.get('global_step', 0)
        print(f"Resumed from Stage {self.start_stage + 1}, Epoch {self.start_epoch + 1}")

    def _save_checkpoint(self, epoch: int, stage_idx: int, is_best: bool = False):
        """Saves a complete checkpoint of the current training state."""
        state = {
            'epoch': epoch, 'stage': stage_idx, 'global_step': self.global_step,
            'best_val_loss': self.best_val_loss, 'wandb_id': self.wandb_id,
            'generator': self.generator.state_dict(),
            'discriminator': self.discriminator.state_dict(),
            'optimizer_g': self.optimizer_g.state_dict(),
            'optimizer_d': self.optimizer_d.state_dict(),
            'ema': self.ema.shadow
        }
        torch.save(state, self.checkpoint_dir / "latest.pth")
        if is_best:
            print(f"✨ New best model found! Saving to best_model.pth")
            torch.save(state, self.checkpoint_dir / "best_model.pth")

    def train(self):
        """Executes the full curriculum learning and GAN training loop."""
        curriculum = self.config['curriculum']
        
        for stage_idx in range(self.start_stage, len(curriculum['patch_sizes'])):
            patch_size, batch_size, epochs = (
                curriculum['patch_sizes'][stage_idx], curriculum['batch_sizes'][stage_idx], curriculum['epochs_per_stage'][stage_idx]
            )
            print(f"\n--- Starting Curriculum Stage {stage_idx+1}: {patch_size}x{patch_size} patches ---")
            
            dataset = AV1Dataset(
                lq_root_dir=self.config['lq_data_root'], hq_root_dir=self.config['hq_data_root'],
                hq_ext=self.config['hq_file_extension'], patch_size=patch_size,
                preset_to_train=self.config['preset_to_train']
            )
            loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=self.config['num_workers'], pin_memory=True, drop_last=True)
            
            scheduler_g = CosineAnnealingLR(self.optimizer_g, T_max=len(loader) * epochs, eta_min=self.config['min_lr'])
            scheduler_d = CosineAnnealingLR(self.optimizer_d, T_max=len(loader) * epochs, eta_min=self.config['min_lr'])

            for epoch in range(self.start_epoch, epochs):
                self.train_epoch(loader, epoch, pbar_desc=f"Epoch {epoch+1}/{epochs} [Patch {patch_size}]")
                
                if (epoch + 1) % self.config['validate_every_n_epochs'] == 0:
                    val_loss = self.validate(loader, epoch) # Simplified: validate on a random train batch
                    is_best = val_loss < self.best_val_loss
                    self.best_val_loss = min(self.best_val_loss, val_loss)
                    self._save_checkpoint(epoch, stage_idx, is_best=is_best)

            self.start_epoch = 0 # Reset for next curriculum stage

    def train_epoch(self, loader: DataLoader, epoch: int, pbar_desc: str):
        """Runs a single training epoch."""
        self.generator.train(); self.discriminator.train()
        pbar = tqdm(loader, desc=pbar_desc)

        for batch in pbar:
            lq, hq, crf_gt = batch['lq'].to(self.device), batch['hq'].to(self.device), batch['crf'].to(self.device)
            autocast_device = 'cuda' if self.device.type == 'cuda' else 'cpu'

            # --- Train Discriminator ---
            self.optimizer_d.zero_grad()
            with torch.autocast(device_type=autocast_device, enabled=self.config['mixed_precision']):
                fake, _ = self.generator(lq)
                loss_d = (self.criterion_gan(self.discriminator(hq), torch.ones_like(self.discriminator(hq))) +
                          self.criterion_gan(self.discriminator(fake.detach()), torch.zeros_like(self.discriminator(fake.detach())))) * 0.5
            self.scaler.scale(loss_d).backward()
            self.scaler.step(self.optimizer_d)
            
            # --- Train Generator ---
            self.optimizer_g.zero_grad()
            with torch.autocast(device_type=autocast_device, enabled=self.config['mixed_precision']):
                fake, pred_crf = self.generator(lq)
                loss_g_gan = self.criterion_gan(self.discriminator(fake), torch.ones_like(self.discriminator(fake)))
                loss_g_recon_dict = self.criterion_recon(fake, hq)
                loss_g_recon = loss_g_recon_dict['total_loss']
                loss_g_crf = self.criterion_crf(pred_crf, crf_gt)
                loss_g = loss_g_recon + self.config['loss']['w_gan'] * loss_g_gan + self.config['loss']['w_crf'] * loss_g_crf
            self.scaler.scale(loss_g).backward()
            if self.config['grad_clip'] > 0: self.scaler.unscale_(self.optimizer_g); torch.nn.utils.clip_grad_norm_(self.generator.parameters(), self.config['grad_clip'])
            self.scaler.step(self.optimizer_g)
            self.scaler.update()

            self.ema.update()
            
            # --- Logging ---
            if self.config['log_to_wandb'] and self.global_step % 50 == 0:
                 log_dict = {"train/loss_g": loss_g.item(), "train/loss_d": loss_d.item(), **{f"train/{k}": v.item() for k,v in loss_g_recon_dict.items()},
                             "train/loss_g_gan": loss_g_gan.item(), "train/loss_g_crf": loss_g_crf.item(), "train/lr_g": self.optimizer_g.param_groups[0]['lr']}
                 wandb.log(log_dict, step=self.global_step)
            self.global_step += 1

    @torch.no_grad()
    def validate(self, loader: DataLoader, epoch: int) -> float:
        """Runs validation, logs samples, and returns the primary validation metric."""
        self.ema.apply_shadow()
        self.generator.eval()
        
        batch = next(iter(loader)) # Use a batch from the training loader for simplicity
        lq, hq = batch['lq'].to(self.device), batch['hq'].to(self.device)
        
        restored, _ = self.generator(lq)
        loss_dict = self.criterion_recon(restored, hq)
        val_loss = loss_dict['total_loss'].item()

        if self.config['log_to_wandb']:
            lq_vis, restored_vis, hq_vis = ((img * 0.5 + 0.5).clamp(0, 1) for img in [lq, restored, hq])
            samples_to_log = min(lq.size(0), self.config['num_val_samples_to_log'])
            grid = torchvision.utils.make_grid(torch.cat([lq_vis[:samples_to_log], restored_vis[:samples_to_log], hq_vis[:samples_to_log]]), nrow=samples_to_log)
            wandb.log({"val/samples": wandb.Image(grid, caption=f"Epoch {epoch+1} | Top: LQ, Mid: Restored, Btm: HQ"), "val/loss": val_loss}, commit=False)

        self.ema.restore()
        return val_loss

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train FastPathRestorer with GAN")
    parser.add_argument("--config", type=str, default="configs/train_restorer.yaml")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint ('latest' for latest.pth).")
    parser.add_argument("--wandb_id", type=str, default=None, help="W&B run ID to resume logging.")
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    resume_path = args.resume
    if resume_path == 'latest':
        resume_path = str(Path(config['checkpoint_save_path']) / 'latest.pth')

    trainer = GANTrainer(config, resume_path=resume_path, wandb_id=args.wandb_id)
    trainer.train()