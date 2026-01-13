import os
import glob
import time
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm # pip install tqdm

# Import our worker logic
import pipeline

def main():
    parser = argparse.ArgumentParser(description="Cryo-EM Filament Picker")
    parser.add_argument('--input', type=str, default='./S2/*.mrc', help="Input file pattern")
    parser.add_argument('--output', type=str, default='picked_micrographs.star', help="Output STAR file")
    
    # Physics Parameters
    parser.add_argument('--pixel_size', type=float, default=0.83, help="Pixel size in A/pix")
    parser.add_argument('--periodicity', type=float, default=4.7, help="Filament periodicity in Angstroms")
    parser.add_argument('--threshold', type=float, default=0.165, help="Z-score threshold for picking")
    
    # System Parameters
    parser.add_argument('--workers', type=int, default=4, help="Number of CPU cores")
    
    args = parser.parse_args()

    # Configuration Dict passed to workers
    config = {
        'threshold': args.threshold,
        'pixel_size': args.pixel_size,
        'periodicity': args.periodicity,
        'outer_resolution': 4.0, # You could make this an arg too if needed
        'min_patches': 1,
        'picked_dir': 'picked_heatmap',
        'trashed_dir': 'trashed_heatmap'
    }
    
    os.makedirs(config['picked_dir'], exist_ok=True)
    os.makedirs(config['trashed_dir'], exist_ok=True)
    
    files = sorted(glob.glob(args.input))
    if not files:
        print(f"No files found matching: {args.input}")
        return

    print(f"Starting batch process on {len(files)} files...")
    print(f"Physics: {args.pixel_size} A/pix, {args.periodicity} A periodicity")
    start_time = time.time()
    
    count_picked = 0
    
    with open(args.output, 'w') as star_f:
        star_f.write("\ndata_picked_micrographs\n\nloop_\n_rlnMicrographName #1\n_rlnFilamentPatchCount #2\n")
        
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            # Pass config dict to worker
            future_to_file = {executor.submit(pipeline.process_single_file, f, config): f for f in files}
            
            # Use tqdm for a professional progress bar
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

    print(f"\n✓ Done in {time.time() - start_time:.1f}s.")
    print(f"  Picked {count_picked}/{len(files)} micrographs.")

if __name__ == '__main__':
    main()