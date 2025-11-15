#!/usr/bin/env python3
"""
restore_video.py - Video Inference Script for AV1 Artifact Restoration
======================================================================

Applies trained AV1 restoration models to video files frame-by-frame,
preserving the original audio track.

Features:
    ✓ Frame-by-frame restoration with progress tracking
    ✓ Automatic audio extraction and muxing (FFmpeg)
    ✓ All model architectures supported
    ✓ Test-Time Augmentation (TTA) support
    ✓ Memory-efficient tiling for large frames
    ✓ FPS reporting

Requirements:
    - opencv-python: pip install opencv-python
    - FFmpeg: Must be installed and in system PATH

Usage Examples:
    # Basic usage
    python scripts/restore_video.py -c checkpoints/best.pth \
        -i input.mp4 -o output.mp4 --crf 45 --preset 4
    
    # With TTA (8x slower, highest quality)
    python scripts/restore_video.py -c checkpoints/best.pth \
        -i input.mkv -o output.mp4 --crf 35 --preset 4 --tta
    
    # Large frames with tiling
    python scripts/restore_video.py -c checkpoints/best.pth \
        -i 4k_video.mp4 -o restored_4k.mp4 --crf 40 \
        --tile 1024 --overlap 128

Author: Soham Mukherjee
Version: 5.0 (Production Final)
License: MIT
"""

import argparse
import logging
import sys
import time
import subprocess
import shutil
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

# Check OpenCV
try:
    import cv2
except ImportError:
    print("="*80, file=sys.stderr)
    print("✗ ERROR: opencv-python not installed", file=sys.stderr)
    print("Install: pip install opencv-python\n", file=sys.stderr)
    print("="*80, file=sys.stderr)
    sys.exit(1)

# ==============================================================================
# SECTION 1: Project Setup
# ==============================================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

try:
    from av1_restorer.unified_inference_av1_restorer import AV1RestorerInference
except ImportError as e:
    print(f"✗ Import error: {e}", file=sys.stderr)
    print("  Ensure av1_restorer/unified_inference_av1_restorer.py exists", file=sys.stderr)
    sys.exit(1)

# ==============================================================================
# SECTION 2: Logging
# ==============================================================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger("AV1Restorer.Video")

# ==============================================================================
# SECTION 3: Utilities
# ==============================================================================

def check_ffmpeg() -> bool:
    """
    Check if FFmpeg is available in system PATH.
    
    Returns:
        True if FFmpeg found, False otherwise
    """
    if shutil.which("ffmpeg") is None:
        logger.error("="*80)
        logger.error("✗ FFmpeg not found in system PATH")
        logger.error("FFmpeg is required for audio extraction/muxing")
        logger.error("Download: https://ffmpeg.org/download.html")
        logger.error("="*80)
        return False
    
    logger.info("✓ FFmpeg found")
    return True


def extract_audio(video_path: Path, audio_path: Path) -> bool:
    """
    Extract audio track from video using FFmpeg.
    
    Args:
        video_path: Path to input video
        audio_path: Path to save extracted audio
        
    Returns:
        True if audio extracted successfully, False otherwise
    """
    try:
        subprocess.run(
            [
                "ffmpeg", "-i", str(video_path),
                "-vn",  # No video
                "-acodec", "copy",  # Copy audio codec
                "-y",  # Overwrite
                str(audio_path)
            ],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE
        )
        
        if audio_path.exists() and audio_path.stat().st_size > 0:
            logger.info("✓ Audio extracted")
            return True
        else:
            logger.warning("No audio stream found")
            return False
    
    except subprocess.CalledProcessError as e:
        logger.warning(f"Audio extraction failed: {e.stderr.decode()}")
        return False
    except FileNotFoundError:
        logger.error("FFmpeg not found")
        return False


def mux_video_audio(video_path: Path, audio_path: Path, output_path: Path) -> bool:
    """
    Combine video and audio streams into final output.
    
    Args:
        video_path: Path to video stream
        audio_path: Path to audio stream
        output_path: Path to save final output
        
    Returns:
        True if muxing successful, False otherwise
    """
    try:
        subprocess.run(
            [
                "ffmpeg",
                "-i", str(video_path),
                "-i", str(audio_path),
                "-c:v", "copy",  # Copy video codec
                "-c:a", "copy",  # Copy audio codec
                "-y",  # Overwrite
                str(output_path)
            ],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
        
        logger.info(f"✓ Final video saved: {output_path.name}")
        return True
    
    except subprocess.CalledProcessError as e:
        logger.error(f"Muxing failed: {e}")
        return False


# ==============================================================================
# SECTION 4: Video Processing
# ==============================================================================

def process_video(
    model: AV1RestorerInference,
    input_path: Path,
    output_path: Path,
    crf: int,
    preset: int,
    use_tta: bool = False
) -> None:
    """
    Process video frame-by-frame with audio preservation.
    
    Pipeline:
        1. Extract audio track (FFmpeg)
        2. Open video for reading (OpenCV)
        3. Create temp video writer (OpenCV)
        4. Loop: Read → Restore → Write
        5. Mux video + audio (FFmpeg)
        6. Cleanup temp files
    
    Args:
        model: Configured AV1RestorerInference
        input_path: Path to input video
        output_path: Path to save output video
        crf: CRF value
        preset: Preset value
        use_tta: Enable Test-Time Augmentation
    """
    logger.info(f"Processing: {input_path.name}")
    logger.info(f"  Output: {output_path.name}")
    logger.info(f"  Params: CRF={crf}, Preset={preset}, TTA={use_tta}")
    
    # Define temp file paths
    temp_video = output_path.with_suffix(".temp_video.mp4")
    temp_audio = output_path.with_suffix(".temp_audio.aac")
    
    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # --- STEP 1: Extract Audio ---
    logger.info("Step 1/3: Extracting audio...")
    has_audio = extract_audio(input_path, temp_audio)
    
    # --- STEP 2: Process Frames ---
    logger.info("Step 2/3: Restoring frames...")
    
    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        logger.error(f"Failed to open: {input_path}")
        return
    
    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    logger.info(f"  Video: {width}×{height} @ {fps:.2f} FPS, {frame_count} frames")
    
    # Create video writer (H.264 codec for compatibility)
    fourcc = cv2.VideoWriter_fourcc(*'avc1')
    writer = cv2.VideoWriter(str(temp_video), fourcc, fps, (width, height))
    
    if not writer.isOpened():
        logger.error(f"Failed to create writer: {temp_video}")
        cap.release()
        return
    
    # Process frames with progress bar
    start_time = time.perf_counter()
    pbar = tqdm(total=frame_count, desc="Restoring", unit="frame")
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break  # End of video
            
            # Convert BGR (OpenCV) → RGB (PIL)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(frame_rgb)
            
            # Restore frame
            restored_pil = model.restore(pil_img, crf, preset, use_tta=use_tta)
            
            # Convert RGB (PIL) → BGR (OpenCV)
            restored_bgr = cv2.cvtColor(np.array(restored_pil), cv2.COLOR_RGB2BGR)
            
            # Write restored frame
            writer.write(restored_bgr)
            pbar.update(1)
    
    finally:
        pbar.close()
        cap.release()
        writer.release()
        cv2.destroyAllWindows()
    
    # Calculate statistics
    end_time = time.perf_counter()
    total_time = end_time - start_time
    avg_fps = frame_count / total_time if total_time > 0 else 0
    
    logger.info(f"✓ Processed {frame_count} frames in {total_time:.2f}s")
    logger.info(f"  Average: {avg_fps:.2f} FPS")
    
    # --- STEP 3: Mux Audio + Video ---
    logger.info("Step 3/3: Muxing audio and video...")
    
    if has_audio:
        # Combine video and audio
        success = mux_video_audio(temp_video, temp_audio, output_path)
        if not success:
            logger.warning("Muxing failed, copying video without audio")
            shutil.move(str(temp_video), str(output_path))
    else:
        # No audio, just move video
        shutil.move(str(temp_video), str(output_path))
        logger.info(f"✓ Video saved: {output_path.name}")
    
    # --- STEP 4: Cleanup ---
    for tmp in [temp_video, temp_audio]:
        if tmp.exists():
            try:
                tmp.unlink()
            except Exception as e:
                logger.warning(f"Could not delete: {tmp} ({e})")
    
    logger.info("✓ Cleanup complete")


# ==============================================================================
# SECTION 5: CLI
# ==============================================================================

def main():
    """Parse arguments and orchestrate video restoration."""
    parser = argparse.ArgumentParser(
        description="AV1 Video Restoration (v5.0)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Required
    parser.add_argument('-c', '--checkpoint', required=True,
                       help='Path to model checkpoint (.pth)')
    parser.add_argument('-i', '--input', required=True,
                       help='Input video file')
    parser.add_argument('-o', '--output', required=True,
                       help='Output video file')
    parser.add_argument('--crf', type=int, required=True,
                       help='CRF value (REQUIRED)')
    
    # Parameters
    parser.add_argument('--preset', type=int, default=4,
                       help='Preset value')
    
    # Options
    parser.add_argument('--device', default='auto',
                       choices=['auto', 'cuda', 'mps', 'cpu'],
                       help='Compute device')
    parser.add_argument('--tile', type=int, default=512,
                       help='Tile size for large frames')
    parser.add_argument('--overlap', type=int, default=64,
                       help='Tile overlap')
    parser.add_argument('--tta', action='store_true',
                       help='Test-Time Augmentation (8x slower, +quality)')
    
    args = parser.parse_args()
    
    # Validation
    if not Path(args.input).exists():
        parser.error(f"Input not found: {args.input}")
    
    if args.tta:
        logger.info("🔥 TTA enabled (8x slower, higher quality)")
    
    if not check_ffmpeg():
        sys.exit(1)
    
    # Execute
    try:
        model = AV1RestorerInference(
            args.checkpoint, args.device, args.tile, args.overlap
        )
        
        process_video(
            model,
            Path(args.input).expanduser(),
            Path(args.output).expanduser(),
            args.crf,
            args.preset,
            args.tta
        )
        
        logger.info("✓ Video restoration complete!")
    
    except Exception as e:
        logger.error(f"✗ Error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    main()