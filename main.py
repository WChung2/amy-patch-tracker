import os
import glob
import time
import argparse
import multiprocessing
import signal
import sys
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
    
    parser.add_argument('input_dir', type=str, help="Directory of your cryoSPARC S folder")
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

    # Create Output Directory
    os.makedirs(args.output_dir, exist_ok=True)

    input_pattern = os.path.join(args.input_dir, 'motioncorrected/*_patch_aligned_doseweighted.mrc')
    output_star = os.path.join(args.output_dir, 'picked_micrographs.star')

    config = {
        'threshold': args.threshold,
        'pixel_size': args.pixel_size,
        'periodicity': args.periodicity,
        'outer_resolution': 4.0, 
        'min_patches': 1,
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
    print(f"System:  {args.workers} workers (Recycling every 1000 tasks)")
    print(":: Note: Press Ctrl+C once to force kill all GPUs.")
    
    start_time = time.time()
    count_picked = 0
    
    # Prepare arguments for map (list of tuples)
    task_args = [(f, config) for f in files]

    with open(output_star, 'w') as star_f:
        star_f.write("\ndata_picked_micrographs\n\nloop_\n_rlnMicrographName #1\n_rlnFilamentPatchCount #2\n")
        
        # maxtasksperchild=1000: Restarts worker after 1000 images to clear RAM/GPU memory
        with multiprocessing.Pool(processes=args.workers, initializer=worker_init, maxtasksperchild=1000) as pool:
            
            try:
                # BEST APPROACH: Just map a helper function that unpacks the tuple
                results_iter = pool.imap_unordered(pipeline_wrapper, task_args, chunksize=1)

                with tqdm(total=len(files), unit="img") as pbar:
                    for filename, count, status in results_iter:
                        if count >= 1:
                            star_f.write(f"{filename} {count}\n")
                            star_f.flush()
                            count_picked += 1
                        
                        pbar.set_postfix(last=status, picked=count_picked)
                        pbar.update(1)

            except KeyboardInterrupt:
                print("\nStopping...")
                pool.terminate()
                pool.join()
                raise
            except Exception as e:
                print(f"Main loop error: {e}")
                pool.terminate()

    print(f"\n✓ Done in {time.time() - start_time:.1f}s.")
    print(f"  Picked {count_picked}/{len(files)} micrographs.")

# --- Helper wrapper for Pool.imap ---
def pipeline_wrapper(args):
    return pipeline.process_single_file(args[0], args[1])

if __name__ == '__main__':
    main()