import os
import mrcfile
import numpy as np
import matplotlib
matplotlib.use('Agg') # Essential for parallel processing
import matplotlib.pyplot as plt

# Import our math module
import calc 

def load_mrc_file(mrc_path):
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

def process_single_file(file_path, config):
    """
    Main worker function. 
    Accepts file_path and a config dictionary.
    """
    # Unpack params
    pixel_size = config.get('pixel_size', 0.83)
    periodicity = config.get('periodicity', 4.7)
    z_thr = config.get('threshold', 0.165)
    
    filename = os.path.basename(file_path)
    
    try:
        data = load_mrc_file(file_path)
        if data is None: return (filename, 0, "Load Error")
        
        h, w = data.shape
        if h < 100 or w < 100: return (filename, 0, "Too Small")
        
        # Patch params
        base_patch_h = h // 4
        base_patch_w = w // 4
        patch_size = (base_patch_h, base_patch_w)
        step_size = (int(base_patch_h * 0.5), int(base_patch_w * 0.5))
        
        # --- CALCULATION PHASE (Using calc.py) ---
        fft_grid, grid_shape = calc.create_fast_patches_and_fft(data, patch_size, step_size)
        fft_avg_grid = calc.fast_average_fft(fft_grid, kernel_size=1)
        
        fft_h, fft_w = fft_avg_grid.shape[-2:]
        
        # Frequency mappings
        nyquist = 1.0 / (2.0 * pixel_size)
        freq_val = 1.0 / periodicity
        y_stripe_px = (freq_val / nyquist) * (min(fft_h, fft_w) / 2)
        
        outer_res_freq = 1.0 / 4.0 # 4.0 Angstrom outer res
        outer_radius_px = (outer_res_freq / nyquist) * (min(fft_h, fft_w) / 2)
        
        angles = np.arange(-90, 91, 2)
        
        W_stripe, bandpass_flat = calc.create_mask_matrices(
            (fft_h, fft_w), [0, y_stripe_px], 
            stripe_width=20, 
            inner_radius=30, 
            outer_radius=outer_radius_px, 
            angles=angles
        )
        
        results = calc.solve_matrix_scoring(fft_avg_grid, W_stripe, bandpass_flat, z_thr)
        
        filament_patch_count = sum(1 for r in results if r['has_filament'])
        
        # --- PLOTTING PHASE ---
        save_plot(results, grid_shape, angles, filename, filament_patch_count, config)

        return (filename, filament_patch_count, "Success")

    except Exception as e:
        # In production, logging.error(e) is better than print
        return (filename, 0, str(e))

def save_plot(results, grid_shape, angles, filename, count, config):
    """Helper function to handle plotting logic"""
    fit_scores_grid = np.zeros(grid_shape)
    angles_grid = np.zeros(grid_shape)
    
    for r in results:
        row, col = r['row'], r['col']
        fit_scores_grid[row, col] = r['fit_score']
        angles_grid[row, col] = angles[r['best_angle_idx']]
        
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    # Heatmap
    im1 = axes[0].imshow(fit_scores_grid, cmap='hot', interpolation='nearest')
    status = "PICKED" if count >= config['min_patches'] else "TRASHED"
    axes[0].set_title(f'{filename}\n{status} (Found {count} patches)')
    plt.colorbar(im1, ax=axes[0])
    
    # Overlay text
    for res in results:
        color = 'green' if res['has_filament'] else 'cyan'
        weight = 'bold' if res['has_filament'] else 'normal'
        axes[0].text(res['col'], res['row'], f"{res['fit_score']:.2f}",
                    ha='center', va='center', color=color, 
                    fontsize=8, fontweight=weight)

    # Angle map
    im2 = axes[1].imshow(angles_grid, cmap='twilight', interpolation='nearest', vmin=-90, vmax=90)
    axes[1].set_title('Angles')
    plt.colorbar(im2, ax=axes[1])
    
    output_name = filename.replace('.mrc', '_heatmap.png')
    out_dir = config['picked_dir'] if count >= config['min_patches'] else config['trashed_dir']
    
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, output_name))
    plt.close(fig)