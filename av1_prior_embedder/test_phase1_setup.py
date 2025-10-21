"""
AURA-Net Setup Validation Script

This script verifies that your environment is correctly configured for training.
It checks:
- Python version
- Required packages
- PyTorch and device availability
- Dataset paths and structure
- Configuration files

Usage:
    python test_phase1_setup.py --config configs/train_av1_p4.yaml
"""

import sys
import argparse
from pathlib import Path
from typing import Tuple, List

import yaml


def print_header(text: str):
    """Print a formatted header."""
    print("\n" + "="*80)
    print(f"  {text}")
    print("="*80)


def print_check(description: str, status: bool, details: str = ""):
    """Print a check result."""
    symbol = "✅" if status else "❌"
    print(f"{symbol} {description}")
    if details:
        print(f"   {details}")


def check_python_version() -> Tuple[bool, str]:
    """Check if Python version is compatible."""
    version = sys.version_info
    required = (3, 8)
    
    is_compatible = version >= required
    details = f"Python {version.major}.{version.minor}.{version.micro}"
    if not is_compatible:
        details += f" (Required: >= {required[0]}.{required[1]})"
    
    return is_compatible, details


def check_packages() -> List[Tuple[str, bool, str]]:
    """Check if required packages are installed."""
    packages = [
        ('torch', 'PyTorch'),
        ('torchvision', 'TorchVision'),
        ('PIL', 'Pillow'),
        ('timm', 'TIMM'),
        ('yaml', 'PyYAML'),
        ('tqdm', 'tqdm'),
        ('ffmpeg', 'ffmpeg-python'),
    ]
    
    results = []
    for module_name, display_name in packages:
        try:
            __import__(module_name)
            results.append((display_name, True, "Installed"))
        except ImportError:
            results.append((display_name, False, "Not installed"))
    
    return results


def check_pytorch_device() -> List[Tuple[str, bool, str]]:
    """Check PyTorch device availability."""
    try:
        import torch
    except ImportError:
        return [("PyTorch", False, "Not installed")]
    
    results = []
    
    # CUDA
    cuda_available = torch.cuda.is_available()
    cuda_details = ""
    if cuda_available:
        cuda_details = f"Device: {torch.cuda.get_device_name(0)}"
    results.append(("CUDA", cuda_available, cuda_details))
    
    # MPS (Apple Silicon)
    mps_available = hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()
    mps_details = "Apple Silicon GPU" if mps_available else ""
    results.append(("MPS", mps_available, mps_details))
    
    # CPU (always available)
    results.append(("CPU", True, "Available (slow)"))
    
    return results


def check_config_file(config_path: str) -> Tuple[bool, str, dict]:
    """Check if config file exists and is valid."""
    config_file = Path(config_path)
    
    if not config_file.exists():
        return False, f"File not found: {config_path}", {}
    
    try:
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
        
        # Check required keys
        required_keys = ['project_name', 'lq_data_root', 'hq_data_root', 'phase1']
        missing = [k for k in required_keys if k not in config]
        
        if missing:
            return False, f"Missing keys: {missing}", config
        
        return True, "Valid configuration", config
        
    except Exception as e:
        return False, f"Error loading config: {e}", {}


def check_dataset_structure(config: dict) -> List[Tuple[str, bool, str]]:
    """Check dataset paths and structure."""
    results = []
    
    # Check LQ directory
    lq_path = Path(config.get('lq_data_root', '')).expanduser()
    lq_exists = lq_path.exists()
    lq_details = str(lq_path) if lq_exists else f"Not found: {lq_path}"
    results.append(("LQ directory", lq_exists, lq_details))
    
    if lq_exists:
        avif_files = list(lq_path.glob("**/*.avif"))
        results.append(("LQ AVIF files", len(avif_files) > 0, f"Found {len(avif_files)} files"))
    
    # Check HQ directory
    hq_path = Path(config.get('hq_data_root', '')).expanduser()
    hq_exists = hq_path.exists()
    hq_details = str(hq_path) if hq_exists else f"Not found: {hq_path}"
    results.append(("HQ directory", hq_exists, hq_details))
    
    if hq_exists:
        hq_ext = config.get('hq_file_extension', '.png')
        hq_files = list(hq_path.glob(f"**/*{hq_ext}"))
        results.append(("HQ source files", len(hq_files) > 0, f"Found {len(hq_files)} files"))
    
    # Check checkpoint directory
    ckpt_path = Path(config.get('phase1', {}).get('checkpoint_save_path', './checkpoints'))
    results.append(("Checkpoint dir", True, f"Will create: {ckpt_path}"))
    
    return results


def check_dataset_sample(config: dict) -> Tuple[bool, str]:
    """Try to load a sample from the dataset."""
    try:
        import sys
        from pathlib import Path
        
        # Add project root to path
        project_root = Path(__file__).parent
        sys.path.insert(0, str(project_root))
        
        from core.dataset import AV1Dataset
        
        dataset = AV1Dataset(
            lq_root_dir=config['lq_data_root'],
            hq_root_dir=config['hq_data_root'],
            hq_ext=config['hq_file_extension'],
            patch_size=128,
            crf_range=tuple(config.get('crf_range', [23, 63])),
            preset_range=tuple(config.get('preset_range', [4, 4]))
        )
        
        if len(dataset) == 0:
            return False, "Dataset is empty"
        
        # Try to load first sample
        sample = dataset[0]
        
        return True, f"Loaded sample successfully ({len(dataset)} total samples)"
        
    except Exception as e:
        return False, f"Error loading dataset: {e}"


def check_model_instantiation() -> Tuple[bool, str]:
    """Try to instantiate the model."""
    try:
        import sys
        from pathlib import Path
        
        project_root = Path(__file__).parent
        sys.path.insert(0, str(project_root))
        
        from core.av1_prior_embedder import AV1PriorEmbedder
        
        model = AV1PriorEmbedder(embed_dim=256)
        param_count = model.get_num_params()
        
        return True, f"Model created ({param_count['total']:,} parameters)"
        
    except Exception as e:
        return False, f"Error creating model: {e}"


def main():
    parser = argparse.ArgumentParser(description="Validate AURA-Net setup")
    parser.add_argument(
        '--config',
        type=str,
        default='configs/train_av1_p4.yaml',
        help='Path to training configuration file'
    )
    parser.add_argument(
        '--skip-dataset',
        action='store_true',
        help='Skip dataset checks (useful if data not downloaded yet)'
    )
    
    args = parser.parse_args()
    
    print_header("AURA-Net Setup Validation")
    
    all_passed = True
    
    # 1. Python Version
    print_header("Python Environment")
    status, details = check_python_version()
    print_check("Python version", status, details)
    all_passed &= status
    
    # 2. Required Packages
    print("\nRequired Packages:")
    package_results = check_packages()
    for name, status, details in package_results:
        print_check(name, status, details)
        all_passed &= status
    
    # 3. PyTorch Devices
    print_header("PyTorch Device Support")
    device_results = check_pytorch_device()
    has_accelerator = False
    for name, status, details in device_results:
        print_check(name, status, details)
        if name in ['CUDA', 'MPS'] and status:
            has_accelerator = True
    
    if not has_accelerator:
        print("\n⚠️  Warning: No GPU acceleration available. Training will be slow.")
        print("   Consider using a machine with CUDA or Apple Silicon MPS support.")
    
    # 4. Configuration File
    print_header("Configuration")
    config_status, config_details, config = check_config_file(args.config)
    print_check("Config file", config_status, config_details)
    all_passed &= config_status
    
    if not config_status:
        print("\n❌ Configuration check failed. Cannot proceed with further checks.")
        sys.exit(1)
    
    # 5. Dataset Structure
    if not args.skip_dataset:
        print_header("Dataset Structure")
        dataset_results = check_dataset_structure(config)
        for name, status, details in dataset_results:
            print_check(name, status, details)
            if "directory" in name.lower():
                all_passed &= status
        
        # 6. Dataset Loading
        print("\nDataset Loading Test:")
        status, details = check_dataset_sample(config)
        print_check("Load dataset sample", status, details)
        if not status:
            all_passed = False
            print("\n⚠️  Dataset loading failed. Check paths and file structure.")
    else:
        print_header("Dataset Structure")
        print("⏭️  Skipped (--skip-dataset flag)")
    
    # 7. Model Instantiation
    print_header("Model Instantiation")
    status, details = check_model_instantiation()
    print_check("Create model", status, details)
    all_passed &= status
    
    # Summary
    print_header("Summary")
    if all_passed:
        print("✅ All checks passed! Your environment is ready for training.")
        print("\nNext steps:")
        print("  1. Review your config: configs/train_av1_p4.yaml")
        print("  2. Start training: python core/train_av1_prior_embedder.py")
        print("  3. Monitor progress with Weights & Biases or tensorboard")
    else:
        print("❌ Some checks failed. Please fix the issues above before training.")
        sys.exit(1)
    
    print("="*80 + "\n")


if __name__ == "__main__":
    main()