# AURA-Net: AV1 Artifact Removal via Compression-Aware Diffusion

**AURA-Net** (AV1 Upscaling and Restoration with Awareness) is a deep learning project for removing compression artifacts from AV1-encoded images, inspired by [CODiff](https://github.com/jp-guo/CODiff).

## 🎯 Project Overview

AURA-Net extends CODiff's compression-aware one-step diffusion approach from JPEG to AV1, the modern open-source video codec. The project follows a two-phase training strategy:

### Phase 1: Compression Prior Embedder (CaPE) 🎓
Train a model to understand AV1 compression through **dual learning**:
- **Explicit Learning**: Predict CRF values to distinguish compression levels
- **Implicit Learning**: Reconstruct clean images to understand artifacts

### Phase 2: One-Step Diffusion Model (Coming Soon) 🚀
Integrate CaPE with a diffusion model for real-time artifact removal.

## 📊 Key Features

- ✅ **Dual Learning Strategy**: Comprehensive compression prior extraction
- ✅ **Curriculum Learning**: Progressive training from 128×128 to 512×512 patches
- ✅ **Multi-Scale Architecture**: CNN (local artifacts) + Swin Transformer (global context)
- ✅ **Production Ready**: Comprehensive logging, checkpointing, and validation
- ✅ **Extensible**: Designed for future multi-preset and multi-codec support
- ✅ **Well Documented**: Extensive docstrings and inline comments

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                  Input: LQ AV1 Image [B,3,H,W]              │
└──────────────────────┬──────────────────────────────────────┘
                       │
        ┌──────────────┴──────────────┐
        │                             │
┌───────▼────────┐          ┌────────▼────────┐
│  CaPE Encoder  │          │ Swin Transformer│
│ (Local Detect) │          │ (Global Context)│
└───────┬────────┘          └────────┬────────┘
        │                             │
        └──────────────┬──────────────┘
                       │
                ┌──────▼──────┐
                │   Fusion    │
                └──────┬──────┘
                       │
        ┌──────────────┴──────────────┐
        │                             │
┌───────▼────────┐          ┌────────▼────────┐
│  UNet Decoder  │          │ CRF Predictor   │
│   (Implicit)   │          │   (Explicit)    │
└───────┬────────┘          └────────┬────────┘
        │                             │
 Reconstructed                   Predicted
    Image                          CRF
```

## 📁 Project Structure

```
aura/
├── configs/
│   └── train_av1_p4.yaml          # Training configuration
├── core/
│   ├── dataset.py                 # AV1 dataset loader
│   ├── av1_prior_embedder.py      # CaPE model architecture
│   ├── losses.py                  # Dual learning losses
│   └── train_av1_prior_embedder.py # Phase 1 training script
├── av1_divk2k_dataset/
│   └── train/
│       ├── crf_23/preset_4/       # Organized by CRF and preset
│       ├── crf_24/preset_4/
│       └── ...
├── degrade_av1.py                 # Dataset generation script
├── create_patches.py              # Patch extraction (optional)
├── requirements.txt
└── setup_env.sh                   # Environment setup
```

## 🚀 Quick Start

### 1. Environment Setup

```bash
# Clone the repository
git clone <your-repo-url>
cd aura

# Setup conda environment (handles macOS MPS and Linux CUDA automatically)
chmod +x setup_env.sh
./setup_env.sh

# Activate environment
conda activate aura
```

### 2. Dataset Preparation

#### Option A: Generate AV1 Dataset from High-Quality Images

```bash
# Download DIV2K dataset (example)
wget http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_HR.zip
unzip DIV2K_train_HR.zip

# Generate AV1-compressed versions
python degrade_av1.py \
    --input_dir ./DIV2K_train_HR \
    --output_dir ./av1_divk2k_dataset/train \
    --crf_range 23 63 \
    --preset_range 4 4 \
    --num_workers 8
```

This creates a directory structure:
```
av1_divk2k_dataset/train/
├── crf_23/preset_4/0001_crf23_p4.avif
├── crf_24/preset_4/0001_crf24_p4.avif
└── ...
```

#### Option B: Use Pre-Generated Dataset

If you already have AV1-compressed images, organize them following the structure above.

### 3. Configure Training

Edit `configs/train_av1_p4.yaml`:

```yaml
# Update these paths to match your system
lq_data_root: "/path/to/av1_divk2k_dataset/train"
hq_data_root: "/path/to/DIV2K_train_HR"

# Adjust training parameters
phase1:
  patch_sizes: [128, 256]      # Start small, grow larger
  batch_sizes: [64, 32]        # Adjust based on GPU memory
  epochs_per_stage: [50, 75]   # Training epochs per stage
```

### 4. Train Phase 1

```bash
# Start training
python core/train_av1_prior_embedder.py --config configs/train_av1_p4.yaml

# Resume from checkpoint
python core/train_av1_prior_embedder.py \
    --config configs/train_av1_p4.yaml \
    --resume checkpoints/phase1_av1_p4/checkpoint_stage0_epoch49.pth

# Enable debug mode
python core/train_av1_prior_embedder.py \
    --config configs/train_av1_p4.yaml \
    --debug
```

### 5. Monitor Training

Training progress is logged to:
- **Console**: Real-time progress bars and metrics
- **Weights & Biases**: Comprehensive experiment tracking (if enabled)
- **Checkpoints**: Saved in `checkpoints/phase1_av1_p4/`

## 📊 Training Configuration Guide

### Curriculum Learning

The model trains progressively on larger patches:

| Stage | Patch Size | Batch Size | Purpose |
|-------|------------|------------|---------|
| 1 | 128×128 | 64 | Learn local compression patterns |
| 2 | 256×256 | 32 | Learn broader context |
| 3 (opt) | 512×512 | 16 | Refine full-context understanding |

### Loss Weights

Based on CODiff paper recommendations:

```yaml
reconstruction_loss_weight: 1.0   # Implicit learning
crf_prediction_loss_weight: 0.1   # Explicit learning
```

Adjust if:
- **Model can't predict CRF**: Increase `crf_prediction_loss_weight`
- **Reconstruction quality poor**: Increase `reconstruction_loss_weight`

### Learning Rate

Default: `1e-4` with cosine annealing to `1e-6`

Adjust if:
- **Training unstable**: Decrease to `5e-5`
- **Training too slow**: Increase to `2e-4`

## 💡 Usage Tips

### Memory Optimization

If you encounter OOM (Out of Memory) errors:

```yaml
# Reduce batch size
batch_sizes: [32, 16]  # Instead of [64, 32]

# Reduce patch sizes
patch_sizes: [128, 192]  # Instead of [128, 256]

# Reduce number of workers
num_workers: 4  # Instead of 8

# Enable mixed precision (CUDA only)
mixed_precision: true
```

### Training Time Estimates

On NVIDIA RTX 4090:
- Stage 1 (128×128, 50 epochs): ~4 hours
- Stage 2 (256×256, 75 epochs): ~12 hours
- **Total**: ~16 hours

On Apple M4 Max (MPS):
- Stage 1: ~8 hours
- Stage 2: ~24 hours
- **Total**: ~32 hours

### Validation Strategy

```yaml
validate_every_n_epochs: 5  # Validate every 5 epochs
```

Validation helps monitor:
- Overfitting (train loss ↓, val loss ↑)
- Convergence (both losses plateau)
- Best model selection

## 📈 Future Roadmap

### Short Term
- [ ] Complete Phase 1 training on preset 4
- [ ] Add validation metrics (PSNR, SSIM, LPIPS)
- [ ] Implement frequency domain loss
- [ ] Add visualization tools

### Medium Term
- [ ] Phase 2: Integrate with Stable Diffusion
- [ ] Multi-preset training (presets 2-8)
- [ ] Perceptual loss implementation
- [ ] Model compression and optimization

### Long Term
- [ ] Multi-codec support (HEVC, VP9, AV2)
- [ ] Real-time inference optimization
- [ ] Web demo and API
- [ ] Pre-trained model zoo

## 🔬 Experimental Results (Coming Soon)

Metrics to track:
- **PSNR/SSIM**: Traditional quality metrics
- **LPIPS/DISTS**: Perceptual quality
- **CRF Prediction MAE**: Explicit learning effectiveness
- **Inference Time**: Speed benchmarks

## 🤝 Contributing

Contributions welcome! Areas of interest:
- Phase 2 diffusion model implementation
- Additional loss functions
- Optimization improvements
- Documentation and tutorials

## 📝 Citation

If you use this work, please cite:

```bibtex
@article{guo2025codiff,
  title={Compression-Aware One-Step Diffusion Model for JPEG Artifact Removal},
  author={Guo, Jinpei and Chen, Zheng and Li, Wenbo and Guo, Yong and Zhang, Yulun},
  journal={arXiv preprint arXiv:2502.09873},
  year={2025}
}
```

## 📄 License

[Your chosen license]

## 🙏 Acknowledgments

- [CODiff](https://github.com/jp-guo/CODiff) for the compression-aware diffusion framework
- [Stable Diffusion](https://github.com/Stability-AI/stablediffusion) for base diffusion architecture
- [timm](https://github.com/huggingface/pytorch-image-models) for Swin Transformer implementation
- [DIV2K](https://data.vision.ee.ethz.ch/cvl/DIV2K/) for high-quality training data

## 📧 Contact

For questions or collaboration:
- GitHub Issues: [Create an issue]
- Email: [your-email]

---

**Status**: 🚧 Phase 1 in active development | Phase 2 coming soon
