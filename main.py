import os
import glob
import time
import argparse
import multiprocessing
import signal
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm 

# Import our worker logic
import pipeline

# --- 1. Define a Worker Initializer ---
def worker_init():
    """Ensures workers ignore Ctrl+C so the main process handles cleanup."""
    signal.signal(signal.SIGINT, signal.SIG_IGN)

def main():
    # --- 2. CRITICAL: Handle Ctrl+C to release GPU ---
    def urgent_shutdown(signum, frame):
        print("\n\n:: INTERRUPTED! Force killing all workers to release GPUs...")
        try:
            os.killpg(os.getpgid(os.getpid()), signal.SIGKILL)
        except Exception:
            sys.exit(1)

    signal.signal(signal.SIGINT, urgent_shutdown)
    
    # Force 'spawn' context for CUDA compatibility
    try:
        multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError:
        pass

    parser = argparse.ArgumentParser(description="Cryo-EM Filament Picker")
    
    # --- CHANGED: Positional Arguments ---
    # These allow you to run: python main.py /path/to/input /path/to/output
    parser.add_argument('input_dir', type=str, help="Directory containing input MRC files")
    parser.add_argument('output_dir', type=str, help="Directory to save STAR file and heatmaps")
    
    # Optional Physics Parameters
    parser.add_argument('--pixel_size', type=float, default=0.83, help="Pixel size in A/pix")
    parser.add_argument('--periodicity', type=float, default=4.7, help="Filament periodicity in Angstroms")
    parser.add_argument('--threshold', type=float, default=0.165, help="Z-score threshold for picking")
    
    # System Parameters
    parser.add_argument('--workers', type=int, default=8, help="Number of workers (Recommend 2-3 per GPU)")
    
    args = parser.parse_args()

    # Validate Input Directory
    if not os.path.isdir(args.input_dir):
        print(f"Error: Input directory not found: {args.input_dir}")
        return

    # Create Output Directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    # Construct file paths based on the positional arguments
    # Note: We append the glob pattern (*_patch_aligned.mrc) here automatically
    input_pattern = os.path.join(args.input_dir, '*_patch_aligned.mrc')
    output_star = os.path.join(args.output_dir, 'picked_micrographs.star')

    config = {
        'threshold': args.threshold,
        'pixel_size': args.pixel_size,
        'periodicity': args.periodicity,
        'outer_resolution': 4.0, 
        'min_patches': 1,
        # Save subfolders inside the user-defined output folder
        'picked_dir': os.path.join(args.output_dir, 'picked_heatmap'),
        'trashed_dir': os.path.join(args.output_dir, 'trashed_heatmap')
    }
    
    os.makedirs(config['picked_dir'], exist_ok=True)
    os.makedirs(config['trashed_dir'], exist_ok=True)
    
    files = sorted(glob.glob(input_pattern))
    if not files:
        print(f"No files found matching pattern: {input_pattern}")
        return

    print(f"Starting batch process on {len(files)} files...")
    print(f"Input:   {args.input_dir}")
    print(f"Output:  {args.output_dir}")
    print(f"System:  {args.workers} workers on {multiprocessing.cpu_count()} CPUs")
    print(":: Note: Press Ctrl+C once to force kill all GPUs.")
    
    start_time = time.time()
    count_picked = 0
    
    with open(output_star, 'w') as star_f:
        star_f.write("\ndata_picked_micrographs\n\nloop_\n_rlnMicrographName #1\n_rlnFilamentPatchCount #2\n")
        
        with ProcessPoolExecutor(max_workers=args.workers, initializer=worker_init) as executor:
            future_to_file = {executor.submit(pipeline.process_single_file, f, config): f for f in files}
            
            try:
                with tqdm(total=len(files), unit="img") as pbar:
                    for future in as_completed(future_to_file):
                        try:
                            filename, count, status = future.result()
                            
                            if count >= 1:
                                star_f.write(f"{filename} {count}\n")
                                star_f.flush()
                                count_picked += 1
                            
                            pbar.set_postfix(last=status, picked=count_picked)
                        except Exception as e:
                            pbar.write(f"Critical error: {e}")
                        
                        pbar.update(1)
            except KeyboardInterrupt:
                print("\nStopping...")
                executor.shutdown(wait=False, cancel_futures=True)
                raise

    print(f"\n✓ Done in {time.time() - start_time:.1f}s.")
    print(f"  Picked {count_picked}/{len(files)} micrographs.")

if __name__ == '__main__':
    main()