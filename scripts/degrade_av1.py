import ffmpeg
import os
from pathlib import Path
import argparse
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

"""
Example Usage
python scripts/degrade_av1.py \
    --input_dir av1_data/test/hq \
    --output_dir av1_data/test/lq \
    --crf_range 23 63 \
    --preset_range 4 4
"""

def process_image(args_tuple):
    """
    Worker function to process a single image with a specific set of AV1 parameters.
    Takes a tuple of arguments to be compatible with multiprocessing.Pool.
    """
    input_path, output_path, crf, preset = args_tuple
    
    # Ensure the output directory for this specific preset exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        # Define the encoder options as a dictionary to ensure correct flag formatting.
        # Forces the use of '-cpu-used' instead of '-cpu_used'.
        encoder_options = {
            'vcodec': 'libaom-av1',
            'crf': crf,
            'cpu-used': preset, # The key here is the exact command-line flag.
            'loglevel': 'error'
        }

        # Use ffmpeg-python to build the command
        (
            ffmpeg
            .input(str(input_path))
            .output(str(output_path), **encoder_options) # Unpack the options dictionary
            .overwrite_output()
            .run(capture_stdout=True, capture_stderr=True) # Capture streams
        )
        return None # Return None on success
    except ffmpeg.Error as e:
        # Robustly decode stderr, handling the case where it might be None.
        error_message = e.stderr.decode('utf-8') if e.stderr else "FFmpeg error with no stderr output."
        return f"Error processing {input_path} with CRF {crf}, Preset {preset}: {error_message}"
    except Exception as e:
        # Catch any other unexpected errors
        return f"A non-ffmpeg error occurred on {input_path}: {str(e)}"

def main(args):
    """
    Main function to find images, create processing tasks, and run them in parallel.
    """
    input_dir = Path(args.input_dir).expanduser() # Expand ~ to home directory
    output_dir = Path(args.output_dir).expanduser()
    
    # Find all image files in the input directory
    print(f"Searching for images in: {input_dir}")
    image_extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.tiff']
    source_images = [p for p in input_dir.glob('**/*') if p.suffix.lower() in image_extensions]
    
    if not source_images:
        print(f"Error: No images found in {input_dir}. Please check the path.")
        return

    print(f"Found {len(source_images)} source images.")

    # Create a list of tasks, skipping files that already exist.
    print("Scanning for existing files to implement resume functionality...")
    tasks = []
    skipped_count = 0
    # Ensure preset range is valid for libaom-av1 (0-8)
    valid_preset_range = (max(0, args.preset_range[0]), min(8, args.preset_range[1]))

    # Pre-calculate total possible tasks for a more accurate tqdm description
    total_possible_tasks = len(source_images) * (args.crf_range[1] - args.crf_range[0] + 1) * (valid_preset_range[1] - valid_preset_range[0] + 1)

    with tqdm(total=total_possible_tasks, desc="Scanning tasks") as pbar:
        for crf in range(args.crf_range[0], args.crf_range[1] + 1):
            for preset in range(valid_preset_range[0], valid_preset_range[1] + 1):
                for img_path in source_images:
                    pbar.update(1)
                    # Use a more descriptive output filename
                    output_subfolder = output_dir / f"crf_{crf}" / f"preset_{preset}"
                    output_file = output_subfolder / f"{img_path.stem}_crf{crf}_p{preset}.avif"
                    
                    # This is the resume logic: only add the task if the output file doesn't exist.
                    if output_file.exists(): # and os.path.getsize(output_file) > 100: # 100 bytes is a safe threshold... if file exists but incomplete wrote
                        skipped_count += 1
                    else:
                        tasks.append((img_path, output_file, crf, preset))
    
    print(f"Found {skipped_count} already completed tasks. Skipping.")
    print(f"Generated {len(tasks)} new degradation tasks to run.")

    if not tasks:
        print("All tasks are already complete! Nothing to do.")
        return

    print(f"Starting processing with {args.num_workers} workers...")
    with Pool(processes=args.num_workers) as pool:
        results = list(tqdm(pool.imap_unordered(process_image, tasks), total=len(tasks), desc="Degrading Images"))

    errors = [r for r in results if r is not None]
    if errors:
        print(f"\nEncountered {len(errors)} errors during processing.")
        error_log_path = output_dir / "error_log.txt"
        with open(error_log_path, "w") as f:
            for error in errors:
                f.write(f"{error}\n")
        print(f"See {error_log_path} for details.")

    print("\nDegradation process complete!")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="AV1 Degradation Script for AURA-Net Dataset Creation")
    parser.add_argument('--input_dir', type=str, required=True, help='Directory containing the high-quality source images.')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save the degraded AV1 images.')
    parser.add_argument('--crf_range', type=int, nargs=2, default=[23, 63], help='Range of CRF values to use (e.g., 25 60).')
    parser.add_argument('--preset_range', type=int, nargs=2, default=[4, 4], help='Range of encoder presets (cpu-used) to use (0-8). Default [2, 8] includes slower, higher-quality presets.')
    parser.add_argument('--num_workers', type=int, default=max(1, cpu_count() - 2), help='Number of parallel processes to use.')
    
    args = parser.parse_args()
    main(args)

