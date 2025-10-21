"""
Dataset Analysis Tool for AV1 dataset

This script validates your AV1 dataset and provides statistics about:
- Number of images per CRF level
- Number of images per preset
- File size distributions
- Coverage analysis
- Data quality checks

Usage:
    python validate_av1_dataset.py --lq_dir ./av1_dataset/train --hq_dir ~/Desktop/Photos/Div2K/DIV2K_train_HR
"""

import argparse
import re
from pathlib import Path
from collections import defaultdict, Counter
from typing import Dict, List, Tuple
import json

from PIL import Image
from tqdm import tqdm


class DatasetAnalyzer:
    """Comprehensive dataset analyzer for AV1 training data."""
    
    FILENAME_PATTERN = re.compile(r"^(.+?)_crf(\d+)_p(\d+)\.avif$")
    
    def __init__(self, lq_dir: str, hq_dir: str, hq_ext: str = ".png"):
        self.lq_dir = Path(lq_dir).expanduser().resolve()
        self.hq_dir = Path(hq_dir).expanduser().resolve()
        self.hq_ext = hq_ext
        
        # Statistics storage
        self.stats = {
            'total_lq_files': 0,
            'total_hq_files': 0,
            'valid_pairs': 0,
            'missing_hq': 0,
            'invalid_names': 0,
            'crf_distribution': defaultdict(int),
            'preset_distribution': defaultdict(int),
            'crf_per_preset': defaultdict(lambda: defaultdict(int)),
            'file_sizes': [],
            'image_dimensions': defaultdict(int),
            'base_names': set()
        }
    
    def analyze(self) -> Dict:
        """Run complete dataset analysis."""
        print("="*80)
        print("AURA-Net Dataset Analysis")
        print("="*80)
        print(f"LQ Directory: {self.lq_dir}")
        print(f"HQ Directory: {self.hq_dir}")
        print()
        
        # Find all files
        print("Scanning for AVIF files...")
        lq_files = list(self.lq_dir.glob("**/*.avif"))
        self.stats['total_lq_files'] = len(lq_files)
        
        hq_files = list(self.hq_dir.glob(f"**/*{self.hq_ext}"))
        self.stats['total_hq_files'] = len(hq_files)
        
        print(f"Found {self.stats['total_lq_files']} LQ AVIF files")
        print(f"Found {self.stats['total_hq_files']} HQ {self.hq_ext} files")
        print()
        
        # Analyze LQ files
        print("Analyzing LQ files and finding pairs...")
        for lq_path in tqdm(lq_files, desc="Processing"):
            match = self.FILENAME_PATTERN.match(lq_path.name)
            
            if not match:
                self.stats['invalid_names'] += 1
                continue
            
            base_name, crf_str, preset_str = match.groups()
            crf_value = int(crf_str)
            preset_value = int(preset_str)
            
            # Track statistics
            self.stats['base_names'].add(base_name)
            self.stats['crf_distribution'][crf_value] += 1
            self.stats['preset_distribution'][preset_value] += 1
            self.stats['crf_per_preset'][preset_value][crf_value] += 1
            
            # Check for HQ pair
            hq_path = self.hq_dir / f"{base_name}{self.hq_ext}"
            if hq_path.exists():
                self.stats['valid_pairs'] += 1
                
                # Track file sizes
                lq_size = lq_path.stat().st_size
                hq_size = hq_path.stat().st_size
                self.stats['file_sizes'].append({
                    'lq_size': lq_size,
                    'hq_size': hq_size,
                    'compression_ratio': hq_size / lq_size if lq_size > 0 else 0,
                    'crf': crf_value,
                    'preset': preset_value
                })
                
                # Track dimensions (sample only first occurrence of each base)
                if base_name not in [d['base'] for d in self.stats.get('dimension_samples', [])]:
                    try:
                        with Image.open(lq_path) as img:
                            dim = f"{img.size[0]}x{img.size[1]}"
                            self.stats['image_dimensions'][dim] += 1
                            if not hasattr(self.stats, 'dimension_samples'):
                                self.stats['dimension_samples'] = []
                            self.stats['dimension_samples'].append({
                                'base': base_name,
                                'dimension': dim
                            })
                    except Exception as e:
                        print(f"Warning: Could not read {lq_path.name}: {e}")
            else:
                self.stats['missing_hq'] += 1
        
        return self.stats
    
    def print_report(self):
        """Print comprehensive analysis report."""
        print()
        print("="*80)
        print("DATASET ANALYSIS REPORT")
        print("="*80)
        print()
        
        # Overview
        print("OVERVIEW")
        print("-"*80)
        print(f"Total LQ files:        {self.stats['total_lq_files']:,}")
        print(f"Total HQ files:        {self.stats['total_hq_files']:,}")
        print(f"Valid pairs:           {self.stats['valid_pairs']:,}")
        print(f"Missing HQ pairs:      {self.stats['missing_hq']:,}")
        print(f"Invalid filenames:     {self.stats['invalid_names']:,}")
        print(f"Unique base images:    {len(self.stats['base_names']):,}")
        print()
        
        # CRF Distribution
        print("CRF DISTRIBUTION")
        print("-"*80)
        crf_dist = dict(sorted(self.stats['crf_distribution'].items()))
        for crf, count in crf_dist.items():
            bar = "█" * (count // 10)
            print(f"CRF {crf:2d}: {count:5,} images {bar}")
        print()
        
        # Preset Distribution
        print("PRESET DISTRIBUTION")
        print("-"*80)
        preset_dist = dict(sorted(self.stats['preset_distribution'].items()))
        for preset, count in preset_dist.items():
            bar = "█" * (count // 10)
            print(f"Preset {preset}: {count:5,} images {bar}")
        print()

        print("CRF DISTRIBUTION PER PRESET")
        print("-"*80)
        crf_per_preset_sorted = dict(sorted(self.stats['crf_per_preset'].items()))
        for preset, crf_counts in crf_per_preset_sorted.items():
            print(f"\n--- Preset {preset} ---")
            crf_counts_sorted = dict(sorted(crf_counts.items()))
            if not crf_counts_sorted: continue
            
            max_count = max(crf_counts_sorted.values())
            scale = max(1, max_count // 40) # Scale bar for readability
            
            for crf, count in crf_counts_sorted.items():
                bar = "█" * (count // scale)
                print(f"  CRF {crf:2d}: {count:5,} images {bar}")
        print()
        
        # Image Dimensions
        if self.stats['image_dimensions']:
            print("IMAGE DIMENSIONS")
            print("-"*80)
            for dim, count in sorted(self.stats['image_dimensions'].items()):
                print(f"{dim}: {count:,} unique images")
            print()
        
        # File Size Analysis
        if self.stats['file_sizes']:
            print("FILE SIZE ANALYSIS")
            print("-"*80)
            
            avg_lq = sum(f['lq_size'] for f in self.stats['file_sizes']) / len(self.stats['file_sizes'])
            avg_hq = sum(f['hq_size'] for f in self.stats['file_sizes']) / len(self.stats['file_sizes'])
            avg_ratio = sum(f['compression_ratio'] for f in self.stats['file_sizes']) / len(self.stats['file_sizes'])
            
            print(f"Average LQ file size:      {avg_lq / 1024:.2f} KB")
            print(f"Average HQ file size:      {avg_hq / 1024:.2f} KB")
            print(f"Average compression ratio: {avg_ratio:.2f}x")
            print()
            
            # Compression ratio by CRF
            crf_ratios = defaultdict(list)
            for f in self.stats['file_sizes']:
                crf_ratios[f['crf']].append(f['compression_ratio'])
            
            print("Compression ratio by CRF:")
            for crf in sorted(crf_ratios.keys()):
                avg_r = sum(crf_ratios[crf]) / len(crf_ratios[crf])
                print(f"  CRF {crf:2d}: {avg_r:.2f}x")
            print()
        
        # Training Recommendations
        print("TRAINING RECOMMENDATIONS")
        print("-"*80)
        
        total_samples = self.stats['valid_pairs']
        if total_samples == 0:
            print("❌ No valid training pairs found!")
            return
        
        print(f"Dataset ready for training with {total_samples:,} samples")
        print()
        
        # Batch size recommendations
        print("Recommended batch sizes:")
        for patch_size in [128, 256, 512]:
            # Rough estimate: 3 channels * patch_size^2 * 4 bytes (float32) * 2 (LQ + HQ)
            mem_per_sample = 3 * patch_size * patch_size * 4 * 2 / (1024**3)  # GB
            
            # Assume 48GB GPU memory, 50% available for activations
            recommended_batch = int(24 / mem_per_sample)
            # recommended_batch = max(4, min(128, recommended_batch))
            
            print(f"  {patch_size}×{patch_size}: batch_size={recommended_batch}")
        print()
        
        # Check for imbalances
        warnings = []
        
        # Check CRF coverage
        crf_min = min(self.stats['crf_distribution'].keys())
        crf_max = max(self.stats['crf_distribution'].keys())
        expected_crfs = set(range(crf_min, crf_max + 1))
        missing_crfs = expected_crfs - set(self.stats['crf_distribution'].keys())
        
        if missing_crfs:
            warnings.append(f"Missing CRF values: {sorted(missing_crfs)}")
        
        # Check for imbalanced distribution
        crf_counts = list(self.stats['crf_distribution'].values())
        if crf_counts:
            max_count = max(crf_counts)
            min_count = min(crf_counts)
            if max_count / min_count > 5:
                warnings.append(f"Imbalanced CRF distribution (ratio: {max_count/min_count:.1f}:1)")

        # Check for imbalanced distribution WITHIN EACH PRESET
        for preset, crf_counts_dict in self.stats['crf_per_preset'].items():
            crf_counts = list(crf_counts_dict.values())
            # Only check for imbalance if there are multiple CRF groups for the preset
            if len(crf_counts) > 1:
                max_count = max(crf_counts)
                min_count = min(crf_counts)
                if min_count > 0 and max_count / min_count > 5:
                    warnings.append(
                        f"Imbalanced CRF distribution for Preset {preset} (ratio: {max_count/min_count:.1f}:1)"
                    )
        
        if warnings:
            print("⚠️ WARNINGS")
            print("-"*80)
            for warning in warnings:
                print(f"  • {warning}")
            print()
        
        print("="*80)
    
    def save_report(self, output_path: str):
        """Save analysis report to JSON file."""
        # Convert sets to lists for JSON serialization
        report = {
            **self.stats,
            'base_names': list(self.stats['base_names']),
            'crf_distribution': dict(self.stats['crf_distribution']),
            'preset_distribution': dict(self.stats['preset_distribution']),
            'crf_per_preset': {p: dict(c) for p, c in self.stats['crf_per_preset'].items()},
            'image_dimensions': dict(self.stats['image_dimensions'])
        }
        
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"Detailed report saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze AURA-Net AV1 dataset",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        '--lq_dir',
        type=str,
        required=True,
        help='Directory containing LQ AVIF files'
    )
    
    parser.add_argument(
        '--hq_dir',
        type=str,
        required=True,
        help='Directory containing HQ source images'
    )
    
    parser.add_argument(
        '--hq_ext',
        type=str,
        default='.png',
        help='Extension of HQ images'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default='dataset_analysis.json',
        help='Path to save detailed JSON report'
    )
    
    args = parser.parse_args()
    
    # Run analysis
    analyzer = DatasetAnalyzer(args.lq_dir, args.hq_dir, args.hq_ext)
    analyzer.analyze()
    analyzer.print_report()
    analyzer.save_report(args.output)


if __name__ == "__main__":
    main()