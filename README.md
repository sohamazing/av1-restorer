# **AV1 Artifact Restoration \- SOTA Conditional & Nano Models**

**High-performance deep learning models for removing AV1 compression artifacts from images and video frames.**

## **🎯 Project Overview**

This project provides production-ready models for AV1 artifact removal using two complementary approaches:

* **Conditional U-Net (av1\_conditional\_unet\_restorer\_v2.py)**: A single, universal model that handles the entire CRF quality range (23-63) by accepting the CRF value as a condition. This is the recommended high-quality, flexible solution.  
* **Nano Models** (av1\_nano\_\*.py): A suite of ultra-lightweight, specialized models, each trained for a narrow CRF "bucket" (e.g., CRF 34-43). These are optimized for maximum speed and real-time video processing.

Both architectures are built on SOTA principles, including GroupNorm, GELU, efficient blocks, and modern upsampling techniques to ensure high-quality, artifact-free restoration.

## **💾 Dataset Structure**

Your AV1 dataset should be organized as follows for the training and testing scripts:

av1\_data/  
├── train/        \# 80% of master dataset  
│   ├── lq/         \# AV1-compressed images (.avif)  
│   │   ├── crf\_23/preset\_4/  
│   │   ├── crf\_24/preset\_4/  
│   │   └── ... (up to crf\_63)  
│   └── hq/         \# High-quality reference images (.png)  
│  
├── val/          \# 20% of master dataset  
│   ├── lq/         \# AV1-compressed images  
│   └── hq/         \# High-quality reference images  
│  
└── test/         \# Separate test set (e.g., DIV2K\_valid)  
    ├── lq/         \# AV1-compressed images  
    └── hq/         \# High-quality reference images

**Purpose of each split:**

* train/: Used for model training.  
* val/: Used for validation during training (for early stopping and selecting the best model).  
* test/: Used for final inference evaluation with scripts/restore\_av1.py.

## **🏗️ Architecture Overview**

### **1\. Conditional U-Net (Universal Model)**

**Purpose**: A single model handles the entire CRF range (23-63) with compression-aware adaptive restoration.

#### **Key Features**

* **Dual Conditioning Modes**:  
  * **CRF-Only** (128-dim): Fast, efficient, and the primary mode.  
  * **CRF+Preset** (192-dim): Optional mode for maximum quality.  
* **5-Level U-Net**: Deep multi-scale feature extraction for robust artifact removal.  
* **FiLM Conditioning**: Adaptive feature modulation based on compression parameters.  
* **Bottleneck Attention**: Global context modeling via efficient channel-wise self-attention.  
* **SOTA Upsampling**: Uses Bilinear Upsample \+ Conv to eliminate checkerboard artifacts.  
* **Residual Learning**: Predicts the artifact correction, preserving original image details.  
* **Memory-Efficient Tiling**: Built-in inference logic handles images of any size.

\<details\>  
\<summary\>\<b\>View Detailed Architecture Flow\</b\>\</summary\>  
Input \[B,3,H,W\] \+ CRF \[B,1\] (+ Preset \[B,1\])  
    │  
    ▼  
Conditioning Embedder (128/192-dim)  
    │  
    ▼  
┌─────────────────────────────────────────┐  
│         5-Level U-Net Backbone          │  
├─────────────────────────────────────────┤  
│ Head (ch\[0\]) → EfficientResBlocks       │  
│   │                                     │  
│   ▼ Skip 0                              │  
│ Enc1 (ch\[1\]) → FiLM → Blocks  ↓2×       │  
│   │                                     │  
│   ▼ Skip 1                              │  
│ Enc2 (ch\[2\]) → FiLM → Blocks  ↓2×       │  
│   │                                     │  
│   ▼ Skip 2                              │  
│ Enc3 (ch\[3\]) → FiLM → Blocks  ↓2×       │  
│   │                                     │  
│   ▼ Skip 3                              │  
│ Bottleneck (ch\[4\]) ↓2×                 │  
│   → Pre-Attn (Blocks)                   │  
│   → SimpleSelfAttention (Channel-wise)  │  
│   → FiLM Conditioning                   │  
│   → Post-Attn (Blocks)                  │  
│   │                                     │  
│   ▼ ↑2× (Bilinear \+ Conv)               │  
│ Dec3 (ch\[3\]) ← Skip 3 → Blocks          │  
│   │                                     │  
│   ▼ ↑2× (Bilinear \+ Conv)               │  
│ Dec2 (ch\[2\]) ← Skip 2 → Blocks          │  
│   │                                     │  
│   ▼ ↑2× (Bilinear \+ Conv)               │  
│ Dec1 (ch\[1\]) ← Skip 1 → Blocks          │  
│   │                                     │  
│   ▼ ↑2× (Bilinear \+ Conv)               │  
│ Tail (ch\[0\]) ← Skip 0 → Predict Residual│  
└─────────────────────────────────────────┘  
    │  
    ▼  
Output \= Input \+ Residual (clamped)

\</details\>

#### **Model Sizes (Empirically Calibrated, CRF-Only Mode)**

These configurations are precisely engineered to match target parameter counts.

| Size | Target | Actual Params | Use Case |
| :---- | :---- | :---- | :---- |
| nano | \~2M | **2.30M** | Minimal viable conditional model |
| tiny | \~5M | **4.96M** | Lightweight, fast iteration |
| small | \~10M | **9.21M** | Standard balanced |
| base | \~12M | **12.79M** | Enhanced standard |
| **large** | **\~20M** | **19.69M** | **RECOMMENDED DEFAULT** ⭐ |
| huge | \~32M | **32.34M** | High quality |
| pro | \~50M | **50.31M** | Maximum quality / Research |

*(Note: CRF+Preset mode adds \< 0.3M parameters)*

### **2\. Nano Models (CRF-Specialized)**

**Purpose**: Ultra-lightweight models trained on narrow CRF ranges for real-time video processing.

#### **Strategy: CRF Bucket Specialization**

Train separate specialized models for each compression tier:

| CRF Range | Compression | Strategy |
| :---- | :---- | :---- |
| 23-33 | Light | Texture preservation |
| 34-43 | Moderate | Balanced restoration |
| 44-53 | Heavy | Aggressive correction |
| 54-63 | Extreme | Full reconstruction |

At inference, a router selects the appropriate model based on the input CRF, resulting in faster and more accurate restoration for that specific range.

#### **Available Architectures**

**A. Nano U-Net (av1\_nano\_unet\_restorer.py)**

* **Best balance of quality and speed**  
* 3-Level shallow U-Net (vs 5 in full U-Net)  
* Depthwise separable convolutions \+ ECA attention  
* Sizes: nano (0.2M), tiny (0.5M), small (1.2M), base (2.5M), large (6.0M), huge (11.2M)

Input Image  
    │  
    ▼  
Head \+ ResBlocks  
    │  
    ▼ \[skip0\]  
Encoder-1 (downsample 2×)  
    │  
    ▼ \[skip1\]  
Encoder-2 (downsample 2×)  
    │  
    ▼ \[skip2\]  
Encoder-3 (downsample 2×)  
    │  
    ▼  
Decoder-3 (upsample 2×) ← \[skip2\]  
    │  
    ▼  
Decoder-2 (upsample 2×) ← \[skip1\]  
    │  
    ▼  
Decoder-1 (upsample 2×) ← \[skip0\]  
    │  
    ▼  
Tail → Residual  
    │  
    ▼  
Restored \= Input \+ Residual

**B. Nano ResNet (av1\_nano\_resnet\_restorer.py)**

* **Maximum speed (no downsampling)**  
* Processes at native resolution  
* Multi-scale feature extraction head  
* Sizes: nano (0.7M), tiny (1.2M), small (2.1M), base (3.3M), huge (6.5M)

Input Image  
    │  
    ▼  
Head Conv  
    │  
    ▼  
Multi-Scale Feature Extractor  
(parallel 3x3, 5x5, 7x7 paths)  
    │  
    ▼  
N × Residual Blocks  
(with long skip connection)  
    │  
    ▼  
Tail → Residual  
    │  
    ▼  
Restored \= Input \+ Residual

**C. Nano FBCNN (av1\_nano\_fbcnn\_restorer.py)**

* FBCNN architecture adapted for AV1  
* Single-scale processing, quality-focused  
* \~1.8M params

**D. Nano Mamba (av1\_nano\_mamba\_restorer.py)**

* Experimental Hybrid CNN \+ State Space Model (SSM)  
* Global receptive field  
* \~2.0M params

## **🚀 Setup & Installation**

### **1\. Clone Repository**

git clone \<your-repo-url\>  
cd aura

### **2\. Setup Environment**

The setup script automatically detects and configures for CUDA (NVIDIA), MPS (Apple Silicon), or CPU.

chmod \+x setup\_env.sh  
./setup\_env.sh

\# Activate the environment  
conda activate aura

## **💾 Dataset Preparation Workflow**

This is the complete workflow to generate the train, val, and test splits.

### **1\. Download Source Data**

\# Download DIV2K training data (for train/val)  
wget \[http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K\_train\_HR.zip\](http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K\_train\_HR.zip)  
unzip DIV2K\_train\_HR.zip

\# Download DIV2K validation data (for test)  
wget \[http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K\_valid\_HR.zip\](http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K\_valid\_HR.zip)  
unzip DIV2K\_valid\_HR.zip

### **2\. Generate Master Training Dataset**

python scripts/degrade\_av1.py \\  
    \--input\_dir ./DIV2K\_train\_HR \\  
    \--output\_dir ./av1\_data/master/lq \\  
    \--crf\_range 23 63 \\  
    \--preset\_range 4 4 \\  
    \--num\_workers 8

\# Create symlink to HQ images  
ln \-s $(pwd)/DIV2K\_train\_HR ./av1\_data/master/hq

### **3\. Generate Test Dataset**

python scripts/degrade\_av1.py \\  
    \--input\_dir ./DIV2K\_valid\_HR \\  
    \--output\_dir ./av1\_data/test/lq \\  
    \--crf\_range 23 63 \\  
    \--preset\_range 4 4 \\  
    \--num\_workers 8

\# Create symlink to HQ images  
ln \-s $(pwd)/DIV2K\_valid\_HR ./av1\_data/test/hq

### **4\. Split Master into Train/Val (80/20)**

python scripts/split\_av1\_dataset.py \\  
    \--input\_lq ./av1\_data/master/lq \\  
    \--input\_hq ./av1\_data/master/hq \\  
    \--output\_train\_lq ./av1\_data/train/lq \\  
    \--output\_train\_hq ./av1\_data/train/hq \\  
    \--output\_val\_lq ./av1\_data/val/lq \\  
    \--output\_val\_hq ./av1\_data/val/hq \\  
    \--split\_ratio 0.8

## **🎓 Training Workflows**

### **Workflow 1: Conditional U-Net (Universal Model)**

This workflow trains a single, high-quality model (e.g., large @ 19.7M params) on the full CRF range.

#### **Step 1: Create Configuration**

Create configs/conditional\_unet/unet\_large\_crf23-63.yaml:

\# \============================================================  
\# Conditional U-Net Configuration (CRF-Only Mode)  
\# \============================================================  
project:  
  name: "AV1-Restorer"  
  experiment\_name: "unet\_large\_crf23-63\_full"  
  log\_to\_wandb: true

system:  
  device: "auto"  \# cuda/mps/cpu  
  seed: 42  
  mixed\_precision: true  \# True for CUDA, False for MPS  
  num\_workers: 8

model:  
  type: "unet"  
  size: "large"  \# Select: nano, tiny, small, base, large, huge, pro

dataset:  
  crf\_range: \[23, 63\]     \# Full CRF spectrum  
  preset\_range: \[4, 4\]    \# Single value \= CRF-Only mode  
  norm\_range: \[-1, 1\]     \# Image normalization

data:  
  train\_lq\_root: "./av1\_data/train/lq"  
  train\_hq\_root: "./av1\_data/train/hq"  
  val\_lq\_root: "./av1\_data/val/lq"  
  val\_hq\_root: "./av1\_data/val/hq"  
  lq\_ext: ".avif"  
  hq\_ext: ".png"

\# \============================================================  
\# Curriculum Learning (Progressive Training)  
\# \============================================================  
curriculum:  
  \- \# Stage 1: Small patches (learn local patterns)  
    patch\_size: 128  
    batch\_size: 32 \# Adjust based on VRAM  
    epochs: 50  
      
  \- \# Stage 2: Medium patches (learn broader context)  
    patch\_size: 256  
    batch\_size: 16  
    epochs: 75  
      
  \- \# Stage 3: Large patches (refine full context)  
    patch\_size: 512  
    batch\_size: 8  
    epochs: 50

\# \============================================================  
\# Loss Configuration (SOTA Balanced)  
\# \============================================================  
loss:  
  charbonnier: {enabled: true, weight: 1.0}  
  perceptual: {enabled: true, weight: 0.05} \# Start low  
  ms\_ssim: {enabled: true, weight: 0.15}    \# pip install pytorch-msssim  
  frequency: {enabled: true, weight: 0.01}    \# Low weight

\# \============================================================  
\# Optimizer & Scheduler  
\# \============================================================  
optimizer:  
  type: "adamw"  
  lr: 0.0001  
  use\_ema: true  
  ema\_decay: 0.9999

scheduler:  
  type: "cosine"  
  warmup\_steps: 1000  
  min\_lr: 1.0e-7

\# \============================================================  
\# Training Settings  
\# \============================================================  
training:  
  grad\_clip\_norm: 1.0  
  validate\_every\_n\_epochs: 1  
  log\_every\_n\_steps: 50

checkpoint:  
  dir: "./checkpoints/unet\_large\_crf23-63\_full"  
  save\_every\_n\_epochs: 1

#### **Step 2: Start Training**

\# Train from scratch  
python av1\_restorer/train\_av1\_conditional\_restorer.py \\  
    \--config configs/conditional\_unet/unet\_large\_crf23-63.yaml

\# Resume from latest checkpoint  
python av1\_restorer/train\_av1\_conditional\_restorer.py \\  
    \--config configs/conditional\_unet/unet\_large\_crf23-63.yaml \\  
    \--resume latest

\# Resume with W\&B tracking  
python av1\_restorer/train\_av1\_conditional\_restorer.py \\  
    \--config configs/conditional\_unet/unet\_large\_crf23-63.yaml \\  
    \--resume best \\  
    \--wandb\_id \<your-wandb-run-id\>

#### **Step 3: Monitor Training**

* **Console**: Real-time loss, LR, and validation metrics.  
* **Weights & Biases**: Comprehensive experiment tracking.  
* **Checkpoints**: Auto-saved to the checkpoint: dir.

**Key Metrics to Watch**:

* loss\_charbonnier: Main reconstruction loss (should decrease).  
* metric\_ms\_ssim: Structural similarity (0-1, higher is better).  
* val/improvement\_l1: L1 improvement over baseline (should be positive and increasing).  
* val/restored\_l1 vs. val/baseline\_l1: Restored L1 should become lower than baseline.

### **Workflow 2: Nano Models (CRF-Specialized)**

This workflow trains multiple lightweight models, each specialized for a specific CRF range.

#### **Step 1: Create Configurations**

Create a config file for *each* CRF bucket (e.g., configs/nano\_models/nano\_unet\_small\_crf34-43.yaml).

\# ... (project, system, data sections as above) ...  
model:  
  type: "nano\_unet"  \# or nano\_resnet  
  size: "small"      \# nano, tiny, small, etc.

dataset:  
  crf\_range: \[34, 43\]     \# \<\<\< NARROW CRF BUCKET  
\# ... (rest of config) ...

#### **Step 2: Train Each CRF Bucket**

Launch a separate training run for each config file.

\# Train Model A (CRF 23-33)  
python av1\_restorer/train\_av1\_nano\_restorer.py \\  
    \--config configs/nano\_models/nano\_unet\_small\_crf23-33.yaml

\# Train Model B (CRF 34-43)  
python av1\_restorer/train\_av1\_nano\_restorer.py \\  
    \--config configs/nano\_models/nano\_unet\_small\_crf34-43.yaml

## **🖥️ Inference (Usage)**

Use scripts/restore\_av1.py to run your trained models.

### **Key Arguments**

| Argument | Short | Description |
| :---- | :---- | :---- |
| \--checkpoint \<path\> | \-c | **Required.** Path to the .pth checkpoint file. |
| \--input\_dir \<path\> | \-d | **Required.** Path to a directory of LQ images. |
| \--output\_dir \<path\> |  | **Required.** Path to save restored images. |
| \--auto |  | Auto-detect CRF/Preset from filenames (e.g., ...\_crf30\_p4.avif). |
| \--test |  | Enable test mode: runs metrics against an HQ directory. |
| \--hq\_dir \<path\> |  | (Test Mode) Path to the corresponding HQ images. |
| \--device \<name\> |  | auto, cuda, mps, or cpu. Defaults to auto. |
| \--tile \<size\> |  | Tile size for large images (e.g., 512). |
| \--overwrite |  | Overwrite existing files in the output directory. |
| \--dry\_run |  | Log actions without processing. |

### **Example 1: Test Full Validation Set (with Metrics)**

This is the recommended command for evaluating a model's performance.

python scripts/restore\_av1.py \\  
    \--checkpoint checkpoints/conditional\_unet\_large/best.pth \\  
    \--input\_dir ./av1\_data/test/lq \\  
    \--output\_dir ./results/large\_model\_test\_metrics \\  
    \--hq\_dir ./av1\_data/test/hq \\  
    \--test \\  
    \--auto

### **Example 2: Restore a Directory (Auto-Detect)**

This is the standard use case for batch-processing a folder of images.

python scripts/restore\_av1.py \\  
    \--checkpoint checkpoints/conditional\_unet\_large/best.pth \\  
    \--input\_dir /path/to/my\_compressed\_images \\  
    \--output\_dir /path/to/my\_restored\_images \\  
    \--auto

### **Example 3: Restore a Single Image (Manual Params)**

python scripts/restore\_av1.py \\  
    \--checkpoint checkpoints/conditional\_unet\_large/best.pth \\  
    \--input /path/to/my\_image\_crf45\_p4.avif \\  
    \--output /path/to/restored\_image.png \\  
    \--crf 45 \\  
    \--preset 4

## **📊 Performance Benchmarking**

This project will compare training and inference performance across two distinct hardware setups:

* **Nvidia L4 VM (Cloud Engine)**: Representing a typical cloud-based GPU environment.  
* **M4 Max Macbook Pro (Local Machine)**: Representing high-end local ARM-based hardware (MPS).

Metrics will be gathered during the training and testing phases to evaluate real-world speed and efficiency.

## **🔧 Troubleshooting & Training Tips**

### **Common Issues & Solutions**

* **CUDA out of memory / OOM**  
  * **1\. Set Env Var:** export PYTORCH\_CUDA\_ALLOC\_CONF=expandable\_segments:True  
  * **2\. Reduce batch\_size** in your YAML config.  
  * **3\. Enable Gradient Checkpointing:** In av1\_conditional\_unet\_restorer\_v2.py, import from torch.utils.checkpoint import checkpoint and wrap body calls in your forward pass (e.g., e1 \= checkpoint(self.encoder1\['body'\], e1, use\_reentrant=False)).  
* **NaN/Inf Loss**  
  * **Disable AMP on MPS:** If on Apple Silicon, set mixed\_precision: false in your YAML.  
  * **Check Loss Weights:** A perceptual weight \> 0.2 can cause instability. Start low (0.05).  
  * **Lower Learning Rate:** Change lr: 1.0e-4 to 5.0e-5.  
* **Slow/Negative L1 Improvement**  
  * **Balance Losses:** Your perceptual loss weight is likely too high. **Reduce perceptual.weight** (e.g., to 0.05).  
  * **Add Structural Loss:** **Install and enable ms\_ssim** (pip install pytorch-msssim). This is critical for structural integrity.  
* **Checkerboard Artifacts in Output**  
  * Ensure you are using av1\_conditional\_unet\_restorer\_v2.py, which uses Bilinear Upsample \+ Conv to fix this.

### **Debug Tools**

* **Test Dataset Loading:**  
  python \-c "  
  from utils.av1\_dataset import AV1Dataset  
  ds \= AV1Dataset(  
  lq\_root\_dir='av1\_data/train/lq',  
  hq\_root\_dir='av1\_data/train/hq',  
  hq\_ext='.png',  
  patch\_size=128,  
  crf\_range=(23, 63),  
  preset\_range=(4, 4),  
  norm\_range=(-1, 1\)  
  )  
  print(f'Dataset size: {len(ds)}')  
  ds.print\_statistics()  
  "

* **Dry Run Training (2 epochs):**  
  \# Create a dry\_run.yaml config with epochs: 2  
  python av1\_restorer/train\_av1\_conditional\_restorer.py \\  
      \--config configs/dry\_run.yaml

## **📁 Project Structure**

aura/  
├── av1\_restorer/  
│   ├── models/  
│   │   ├── av1\_conditional\_unet\_restorer\_v2.py  \# SOTA Conditional U-Net  
│   │   ├── av1\_nano\_unet\_restorer.py            \# Nano U-Net  
│   │   ├── av1\_nano\_resnet\_restorer.py          \# Nano ResNet (Fastest)  
│   │   ├── av1\_nano\_fbcnn\_restorer.py  
│   │   ├── av1\_nano\_mamba\_restorer.py  
│   │   └── blocks.py                            \# Shared building blocks  
│   │  
│   ├── train\_av1\_conditional\_restorer.py        \# Trainer for Conditional U-Net  
│   └── train\_av1\_nano\_restorer.py               \# (Hypothetical) Trainer for Nanos  
│  
├── utils/  
│   ├── av1\_dataset.py                  \# Dataloader  
│   └── loss.py                         \# CombinedLoss function  
│  
├── scripts/  
│   ├── degrade\_av1.py                  \# Creates LQ dataset  
│   ├── split\_av1\_dataset.py            \# Splits train/val  
│   └── restore\_av1.py                  \# Inference script  
│  
├── configs/  
│   ├── conditional\_unet/               \# Configs for Conditional U-Net  
│   └── nano\_models/                    \# Configs for Nano models  
│  
├── av1\_data/                           \# master dataset (Div2K \+ Flickr2K)  
├── train/                              \# 80% of master dataset  
│   ├── lq/                             \# AV1-compressed images (.avif)  
│   │   ├── crf\_23/preset\_4/  
│   │   ├── crf\_24/preset\_4/  
│   │   └── ... (up to crf\_63)  
│   └── hq/                             \# High-quality reference images (.png)  
│  
├── val/                                \# 20% of master dataset  
│   ├── lq/                             \# AV1-compressed images  
│   └── hq/                             \# High-quality reference images  
│  
├── test/                               \# Separate test set (e.g., DIV2K\_valid)  
│   ├── lq/                             \# AV1-compressed images  
│   └── hq/                             \# High-quality reference images  
│  
└── checkpoints/                        \# Saved models  
