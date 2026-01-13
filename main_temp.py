import os
import glob
import numpy as np
import mrcfile
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter, gaussian_filter
from scipy.ndimage import uniform_filter, gaussian_filter
from scipy.fft import fft2, fftshift
from skimage.util import view_as_windows
from concurrent.futures import ProcessPoolExecutor, as_completed
import time

# Set up matplotlib configuration
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['figure.dpi'] = 100

def load_mrc_file(mrc_path):
    """Load the MRC/MRCS file safely."""
    try:
        with mrcfile.open(mrc_path, mode='r') as mrc:
            data = mrc.data.copy()
            if len(data.shape) == 3:
                data = data[0]
            data = data.astype(np.float32)
        return data
    except Exception as e:
        print(f"Error loading {mrc_path}: {e}")
        return None

# ================= VECTORIZED MATH FUNCTIONS =================

def create_fast_patches_and_fft(image, patch_size, step_size):
    patches_4d = view_as_windows(image, patch_size, step=step_size)
    n_rows, n_cols, h, w = patches_4d.shape
    
    # Batched FFT
    fft_4d = fft2(patches_4d, axes=(-2, -1))
    fft_shifted = fftshift(fft_4d, axes=(-2, -1))
    magnitude_4d = np.log1p(np.abs(fft_shifted))
    
    return magnitude_4d, (n_rows, n_cols)

def fast_average_fft(fft_4d_grid, kernel_size=3):
    filter_size = (kernel_size, kernel_size, 1, 1)
    averaged_grid = uniform_filter(fft_4d_grid, size=filter_size, mode='reflect')
    return averaged_grid

def create_mask_matrices(shape, y_positions, stripe_width, inner_radius, outer_radius, angles):
    """
    Creates masks that match Version 2 logic:
    1. Create binary mask for all 3 stripes.
    2. Soften it (Gaussian blur).
    3. Apply Bandpass.
    """
    h, w = shape
    n_pixels = h * w
    center_y, center_x = h // 2, w // 2
    y, x = np.ogrid[-center_y:h-center_y, -center_x:w-center_x]
    
    # 1. Bandpass Mask
    radius_sq = x**2 + y**2
    if outer_radius is None:
        bandpass = (radius_sq >= inner_radius**2)
    else:
        bandpass = (radius_sq >= inner_radius**2) & (radius_sq <= outer_radius**2)
    
    # Convert bandpass to float for math
    bandpass_flat = bandpass.ravel().astype(np.float32)
    
    n_angles = len(angles)
    W_stripe = np.zeros((n_angles, n_pixels), dtype=np.float32)
    half_width = stripe_width // 2
    
    for i, angle in enumerate(angles):
        angle_rad = np.radians(-angle)
        cos_a = np.cos(angle_rad)
        sin_a = np.sin(angle_rad)
        y_rot = y * cos_a - x * sin_a
        
        # Build COMPLETE binary mask first (all 3 stripes)
        mask = np.zeros((h, w), dtype=bool)
        for y_pos in y_positions:
            if y_pos == 0:
                mask |= (np.abs(y_rot) <= half_width)
            else:
                mask |= (np.abs(y_rot - y_pos) <= half_width)
                mask |= (np.abs(y_rot + y_pos) <= half_width)
        
        # SOFTEN MASK (Gaussian Blur) - Matches Version 2
        m = mask.astype(np.float32)
        m = gaussian_filter(m, sigma=2.0)
        
        # Normalize peak to 1.0 if it exists
        if m.max() > 0:
            m /= m.max()
            
        # Combine with bandpass and flatten
        # W_stripe contains the "soft" weights for the stripe region within the bandpass
        W_stripe[i, :] = m.ravel() * bandpass_flat

    return W_stripe, bandpass_flat

def solve_matrix_scoring(fft_grid, W_stripe, bandpass_flat, z_thr):
    """
    Vectorized solver matching Version 2 'calculate_stripe_zscore':
    Z = (Mean_stripe - Mean_background) / Std_background
    
    Where Background = (Total_Bandpass_Area - Stripe_Area)
    """
    n_rows, n_cols, h, w = fft_grid.shape
    n_pixels = h * w
    
    # F: (N_patches, N_pixels)
    F = fft_grid.reshape(-1, n_pixels)
    
    # Pre-calculate squared F for variance
    F2 = F**2
    
    # --- 1. Global Bandpass Stats (Per Patch) ---
    # These are constant across all angles for a specific patch
    Sum_all_bp = np.dot(F, bandpass_flat)       # (N_patches,)
    Sum_sq_all_bp = np.dot(F2, bandpass_flat)   # (N_patches,)
    N_all_bp = bandpass_flat.sum()              # Scalar
    
    # --- 2. Stripe Stats (Per Patch, Per Angle) ---
    # W_stripe contains soft weights. 
    Sum_stripe = np.dot(F, W_stripe.T)          # (N_patches, N_angles)
    Sum_sq_stripe = np.dot(F2, W_stripe.T)      # (N_patches, N_angles)
    N_stripe = W_stripe.sum(axis=1)             # (N_angles,)
    
    # Mean of Stripe
    Mean_stripe = Sum_stripe / (N_stripe + 1e-10)
    
    # --- 3. Background Stats (Per Patch, Per Angle) ---
    # Background is defined as the Bandpass region MINUS the Stripe region
    # We broadcast Sum_all_bp (N_patches, 1) to match (N_patches, N_angles)
    Sum_bg = Sum_all_bp[:, None] - Sum_stripe
    Sum_sq_bg = Sum_sq_all_bp[:, None] - Sum_sq_stripe
    N_bg = N_all_bp - N_stripe                  # (N_angles,)
    
    # Mean of Background
    Mean_bg = Sum_bg / (N_bg + 1e-10)
    
    # Variance of Background: E[X^2] - (E[X])^2
    Mean_sq_bg = Sum_sq_bg / (N_bg + 1e-10)
    Var_bg = Mean_sq_bg - (Mean_bg**2)
    
    # Standard Deviation of Background
    Std_bg = np.sqrt(np.maximum(Var_bg, 1e-10))
    
    # --- 4. Final Z-Score Calculation ---
    Z_scores = (Mean_stripe - Mean_bg) / (Std_bg + 1e-10)
    
    # --- 5. Packaging Results ---
    best_indices = np.argmax(Z_scores, axis=1)
    row_indices = np.arange(Z_scores.shape[0])
    best_scores = Z_scores[row_indices, best_indices]
    
    results = []
    for i in range(len(best_scores)):
        results.append({
            'row': i // n_cols,
            'col': i % n_cols,
            'fit_score': best_scores[i],
            'best_angle_idx': best_indices[i],
            'has_filament': best_scores[i] > z_thr
        })
        
    return results

# ================= WORKER FUNCTION =================

def process_single_file(file_path):
    """
    Worker function for processing one file.
    Handles Analysis AND Plotting logic.
    """
    # Config
    pixel_size_angstrom = 0.83
    filament_periodicity_angstrom = 4.7
    outer_resolution_angstrom = 4.0
    
    kernel_size = 1 
    stripe_width = 20
    inner_radius_px = 30
    angle_range = (-90, 90)
    z_thr = 0.165
    min_patches_to_pick = 1
    
    # Defined Directories (Must match main function)
    picked_dir = 'picked_heatmap'
    trashed_dir = 'trashed_heatmap'

    filename = os.path.basename(file_path)
    
    try:
        # Load
        data = load_mrc_file(file_path)
        if data is None: return (filename, 0, "Load Error")
        
        # Size Check
        if data.shape[0] < 100 or data.shape[1] < 100: return (filename, 0, "Too Small")
        
        h, w = data.shape
        base_patch_h = h // 4
        base_patch_w = w // 4
        patch_size = (base_patch_h, base_patch_w)
        step_size = (int(base_patch_h * 0.5), int(base_patch_w * 0.5))
        
        # FFT
        fft_grid, grid_shape = create_fast_patches_and_fft(data, patch_size, step_size)
        fft_avg_grid = fast_average_fft(fft_grid, kernel_size)
        
        # Physics
        fft_h, fft_w = fft_avg_grid.shape[-2:]
        freq_val = 1.0 / filament_periodicity_angstrom
        
        # Helper for frequency
        nyquist = 1.0 / (2.0 * pixel_size_angstrom)
        y_stripe_px = (freq_val / nyquist) * (min(fft_h, fft_w) / 2)
        
        outer_res_freq = 1.0 / outer_resolution_angstrom
        outer_radius_px = (outer_res_freq / nyquist) * (min(fft_h, fft_w) / 2)
        
        y_positions = [0, y_stripe_px]
        
        # Matrices
        angles = np.arange(angle_range[0], angle_range[1] + 1, 2)
        W_stripe, bandpass_flat = create_mask_matrices(
            (fft_h, fft_w), y_positions, stripe_width, 
            inner_radius_px, outer_radius_px, angles
        )
        
        # Solve
        results = solve_matrix_scoring(fft_avg_grid, W_stripe, bandpass_flat, z_thr)
        
        # Stats
        filament_patch_count = sum(1 for r in results if r['has_filament'])
        
        # --- VISUALIZATION (ALWAYS GENERATED) ---
        fit_scores_grid = np.zeros(grid_shape)
        angles_grid = np.zeros(grid_shape)
        
        for r in results:
            row, col = r['row'], r['col']
            fit_scores_grid[row, col] = r['fit_score']
            angles_grid[row, col] = angles[r['best_angle_idx']]
            
        fig4, axes = plt.subplots(1, 2, figsize=(16, 7))
        im1 = axes[0].imshow(fit_scores_grid, cmap='hot', interpolation='nearest')
        
        # Dynamic Title
        status_text = "PICKED" if filament_patch_count >= min_patches_to_pick else "TRASHED"
        axes[0].set_title(f'{filename}\n{status_text} (Found {filament_patch_count} patches)')
        plt.colorbar(im1, ax=axes[0])
        
        # Annotations
        for res in results:
            color = 'green' if res['has_filament'] else 'cyan'
            weight = 'bold' if res['has_filament'] else 'normal'
            axes[0].text(res['col'], res['row'], f"{res['fit_score']:.2f}",
                        ha='center', va='center', color=color, 
                        fontsize=8, fontweight=weight)

        im2 = axes[1].imshow(angles_grid, cmap='twilight', interpolation='nearest', vmin=-90, vmax=90)
        axes[1].set_title('Angles')
        plt.colorbar(im2, ax=axes[1])
        
        output_name = filename.replace('.mrc', '_heatmap.png')
        
        # --- SAVE TO CORRECT FOLDER ---
        if filament_patch_count >= min_patches_to_pick:
            output_path = os.path.join(picked_dir, output_name)
        else:
            output_path = os.path.join(trashed_dir, output_name)
            
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close(fig4)
            
        return (filename, filament_patch_count, "Success")

    except Exception as e:
        print(f"Failed {filename}: {e}")
        return (filename, 0, str(e))

def main():
    input_pattern = './S2/*.mrc'  
    output_star_file = 'picked_micrographs.star'
    
    # Create output directories
    os.makedirs('picked_heatmap', exist_ok=True)
    os.makedirs('trashed_heatmap', exist_ok=True)
    
    files = sorted(glob.glob(input_pattern))
    if not files:
        print("No files found.")
        return
        
    print(f"Starting Batch Process on {len(files)} files...")
    start_time = time.time()
    
    # Open STAR file (Overwrite mode)
    with open(output_star_file, 'w') as star_f:
        star_f.write("\ndata_picked_micrographs\n\nloop_\n_rlnMicrographName #1\n_rlnFilamentPatchCount #2\n")
        
        count_picked = 0
        total_files = len(files)
        
        # Parallel Execution
        with ProcessPoolExecutor(max_workers=4) as executor:
            future_to_file = {executor.submit(process_single_file, f): f for f in files}
            
            # Iterate as results arrive
            for i, future in enumerate(as_completed(future_to_file)):
                filename, count, status = future.result()
                
                print(f"[{i+1}/{total_files}] {filename}: {count} patches ({status})")
                
                if count >= 1:
                    star_f.write(f"{filename} {count}\n")
                    star_f.flush() # Ensure it writes to disk immediately
                    count_picked += 1

    print(f"\n✓ Done in {time.time() - start_time:.1f}s.")
    print(f"  Picked {count_picked} micrographs.")

if __name__ == '__main__':
    main()