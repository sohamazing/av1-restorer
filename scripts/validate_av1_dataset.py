"""
Dataset Analysis Tool for AV1 dataset

This script validates your AV1 dataset after it has been created by
the 'degrade_av1' and 'split_dataset' scripts.

It analyzes the 'train' and 'val' sets by default.
If the --test flag is provided, it will also analyze the 'test' set.

It provides statistics about:
- Number of images per CRF level
- Number of images per preset
- File size distributions
- Coverage analysis
- Data quality checks

Usage:
    # Analyze train and val sets
    python validate_av1_dataset.py --data_dir ./av1_data

    # Analyze train, val, and test sets
    python validate_av1_dataset.py --data_dir ./av1_data --test
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
    
    # Matches filenames like: 0001_crf23_p4.avif
    FILENAME_PATTERN = re.compile(r"^(.+?)_crf(\d+)_p(\d+)\.avif$")
    
    def __init__(self, base_dir: str, set_name: str, hq_ext: str = ".png"):
        self.base_dir = Path(base_dir).expanduser().resolve()
        self.set_name = set_name  # "train", "val", or "test"
        self.lq_dir = self.base_dir / self.set_name / "lq"
        self.hq_dir = self.base_dir / self.set_name / "hq"
        self.hq_ext = hq_ext
        
        if not self.lq_dir.exists():
            raise FileNotFoundError(f"LQ directory not found: {self.lq_dir}")
        if not self.hq_dir.exists():
            raise FileNotFoundError(f"HQ directory not found: {self.hq_dir}")

        # Statistics storage
        self.stats = {
            'set_name': self.set_name,
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
        """Run complete dataset analysis for the specified set."""
        print(f"Analyzing {self.set_name.upper()} set...")
        print(f"  LQ Directory: {self.lq_dir}")
        print(f"  HQ Directory: {self.hq_dir}")
        
        # Find all files
        lq_files = list(self.lq_dir.glob("**/*.avif"))
        self.stats['total_lq_files'] = len(lq_files)
        
        # We can count HQ files, but it's more useful to count found pairs
        hq_file_count = len(list(self.hq_dir.glob(f"**/*{self.hq_ext}")))
        self.stats['total_hq_files'] = hq_file_count
        
        print(f"  Found {self.stats['total_lq_files']} LQ AVIF files")
        print(f"  Found {self.stats['total_hq_files']} HQ {self.hq_ext} files")
        
        # Analyze LQ files
        for lq_path in tqdm(lq_files, desc=f"Processing {self.set_name}"):
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
                        with Image.open(hq_path) as img: # Get dims from HQ
                            dim = f"{img.size[0]}x{img.size[1]}"
                            self.stats['image_dimensions'][dim] += 1
                            if not hasattr(self.stats, 'dimension_samples'):
                                self.stats['dimension_samples'] = []
                            self.stats['dimension_samples'].append({
                                'base': base_name,
                                'dimension': dim
                            })
                    except Exception as e:
                        print(f"Warning: Could not read {hq_path.name}: {e}")
            else:
                self.stats['missing_hq'] += 1
        
        return self.stats
    
    def print_report(self):
        """Print comprehensive analysis report."""
        print()
        print("="*80)
        print(f"DATASET ANALYSIS REPORT: {self.set_name.upper()}")
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
        if not crf_dist:
            print("  No CRF data found.")
        else:
            max_crf_count = max(crf_dist.values()) if crf_dist else 1
            scale = max(1, max_crf_count // 50) # Scale bar
            for crf, count in crf_dist.items():
                bar = "█" * (count // scale)
                print(f"  CRF {crf:2d}: {count:5,} images {bar}")
        print()
        
        # Preset Distribution
        print("PRESET DISTRIBUTION")
        print("-"*80)
        preset_dist = dict(sorted(self.stats['preset_distribution'].items()))
        if not preset_dist:
            print("  No preset data found.")
        else:
            max_preset_count = max(preset_dist.values()) if preset_dist else 1
            scale = max(1, max_preset_count // 50) # Scale bar
            for preset, count in preset_dist.items():
                bar = "█" * (count // scale)
                print(f"  Preset {preset}: {count:5,} images {bar}")
        print()

        print("CRF DISTRIBUTION PER PRESET")
        print("-"*80)
        crf_per_preset_sorted = dict(sorted(self.stats['crf_per_preset'].items()))
        if not crf_per_preset_sorted:
             print("  No per-preset data found.")
        
        for preset, crf_counts in crf_per_preset_sorted.items():
            print(f"\n--- Preset {preset} ---")
            crf_counts_sorted = dict(sorted(crf_counts.items()))
            if not crf_counts_sorted: continue
            
            max_count = max(crf_counts_sorted.values())
            scale = max(1, max_count // 40) # Scale bar for readability
            
            for crf, count in crf_counts_sorted.items():
                bar = "█" * (count // scale)
                print(f"    CRF {crf:2d}: {count:5,} images {bar}")
        print()
        
        # Image Dimensions
        if self.stats['image_dimensions']:
            print("IMAGE DIMENSIONS (FROM HQ)")
            print("-"*80)
            for dim, count in sorted(self.stats['image_dimensions'].items()):
                print(f"  {dim}: {count:,} unique images")
            print()
        
        # File Size Analysis
        if self.stats['file_sizes']:
            print("FILE SIZE ANALYSIS")
            print("-"*80)
            
            avg_lq = sum(f['lq_size'] for f in self.stats['file_sizes']) / len(self.stats['file_sizes'])
            avg_hq = sum(f['hq_size'] for f in self.stats['file_sizes']) / len(self.stats['file_sizes'])
            avg_ratio = sum(f['compression_ratio'] for f in self.stats['file_sizes']) / len(self.stats['file_sizes'])
            
            print(f"  Average LQ file size:      {avg_lq / 1024:.2f} KB")
            print(f"  Average HQ file size:      {avg_hq / (1024*1024):.2f} MB")
            print(f"  Average compression ratio: {avg_ratio:.2f}x")
            print()
            
            # Compression ratio by CRF
            crf_ratios = defaultdict(list)
            for f in self.stats['file_sizes']:
                crf_ratios[f['crf']].append(f['compression_ratio'])
            
            print("  Compression ratio by CRF:")
            for crf in sorted(crf_ratios.keys()):
                avg_r = sum(crf_ratios[crf]) / len(crf_ratios[crf])
                print(f"    CRF {crf:2d}: {avg_r:.2f}x")
            print()
        
        # Training Recommendations
        if self.set_name == 'train':
            print("TRAINING RECOMMENDATIONS")
            print("-"*80)
            
            total_samples = self.stats['valid_pairs']
            if total_samples == 0:
                print("❌ No valid training pairs found!")
                return
            
            print(f"Dataset ready for training with {total_samples:,} samples")
            print()
            
            # Check for imbalances
            warnings = []
            
            # Check CRF coverage
            if self.stats['crf_distribution']:
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
                    if min_count > 0 and max_count / min_count > 5:
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
            'crf_distribution': dict(sorted(self.stats['crf_distribution'].items())),
            'preset_distribution': dict(sorted(self.stats['preset_distribution'].items())),
            'crf_per_preset': {p: dict(sorted(c.items())) for p, c in sorted(self.stats['crf_per_preset'].items())},
            'image_dimensions': dict(sorted(self.stats['image_dimensions'].items()))
        }
        
        output_path_obj = Path(output_path)
        output_path_obj.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path_obj, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"Detailed report for '{self.set_name}' saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze AURA-Net AV1 dataset",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        '--data_dir',
        type=str,
        required=True,
        help='Base data directory (e.g., ./av1_data) containing train/, val/, and optionally test/ subfolders'
    )
    
    parser.add_argument(
        '--test',
        action='store_true',
        help='Also analyze the "test" subdirectory'
    )
    
    parser.add_argument(
        '--hq_ext',
        type=str,
        default='.png',
        help='Extension of HQ images (e.g., .png, .jpg)'
    )
    
    parser.add_argument(
        '--output_dir',
        type=str,
        default='.',
        help='Directory to save detailed JSON report(s)'
    )
    
    args = parser.parse_args()
    
    sets_to_analyze = ["train", "val"]
    if args.test:
        sets_to_analyze.append("test")
        
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*80)
    print(f"AURA-Net Dataset Analysis")
    print(f"Base Directory: {args.data_dir}")
    print(f"Sets to Analyze: {', '.join(sets_to_analyze)}")
    print("="*80)

    for set_name in sets_to_analyze:
        print(f"\n--- Processing Set: {set_name.upper()} ---")
        try:
            # Run analysis
            analyzer = DatasetAnalyzer(args.data_dir, set_name, args.hq_ext)
            analyzer.analyze()
            analyzer.print_report()
            analyzer.save_report(output_dir / f"{set_name}_analysis.json")
        
        except FileNotFoundError as e:
            logger.warning(f"Skipping '{set_name}': {e}")
        except Exception as e:
            logger.error(f"Failed to analyze '{set_name}': {e}", exc_info=True)

    print("\nAnalysis complete.")


if __name__ == "__main__":
    main()