import os
import mrcfile
import numpy as np
import matplotlib
matplotlib.use('Agg') # Essential for parallel processing
import matplotlib.pyplot as plt

# Import our math module
import calc 

# Try to import cupy for device management
try:
    import cupy
except ImportError:
    cupy = None

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
    # Unpack params
    pixel_size = config.get('pixel_size', 0.83)
    periodicity = config.get('periodicity', 4.7)
    z_thr = config.get('threshold', 0.165)
    outer_res = config.get('outer_resolution', 4.0)
    
    # ---------------------------------------------------------
    # NEW: MULTI-GPU DISTRIBUTION
    # ---------------------------------------------------------
    # We use the Process ID (PID) to assign a specific GPU.
    # Process 1 -> GPU 0, Process 2 -> GPU 1, Process 3 -> GPU 2, etc.
    if calc.HAS_GPU and cupy is not None:
        try:
            n_devices = cupy.cuda.runtime.getDeviceCount()
            if n_devices > 0:
                # Round-robin assignment based on OS Process ID
                pid = os.getpid()
                device_id = pid % n_devices
                
                # Force this process to use ONLY this specific GPU
                cupy.cuda.Device(device_id).use()
        except Exception as e:
            # If something fails (e.g. driver issue), just keep going on default
            pass
    # ---------------------------------------------------------
    
    filename = os.path.basename(file_path)
    
    try:
        data = load_mrc_file(file_path)
        if data is None: return (filename, 0, "Load Error")
        
        h, w = data.shape
        if h < 200 or w < 200: return (filename, 0, "Too Small")
        
        # Patch setup
        base_patch_h = h // 4
        base_patch_w = w // 4
        patch_size = (base_patch_h, base_patch_w)
        step_size = (int(base_patch_h * 0.5), int(base_patch_w * 0.5))
        
        # --- FAST CALCULATION (OPTIMIZED) ---
        # Note: We now pass pixel_size and outer_res directly to calc
        fft_grid, grid_shape = calc.create_fast_patches_and_fft(
            data, patch_size, step_size, pixel_size, outer_res
        )
        
        # Average on the small grid
        fft_avg_grid = calc.fast_average_fft(fft_grid, kernel_size=3)
        
        # Recalculate geometry based on the CROPPED grid size
        current_h, current_w = fft_avg_grid.shape[-2:]
        crop_res_freq = 1.0 / outer_res
        
        # Map physics to the new small pixel grid
        freq_val = 1.0 / periodicity
        y_stripe_px = (freq_val / crop_res_freq) * (current_h / 2)
        
        outer_radius_px = (current_h / 2) - 1
        
        # Scale parameters (approximate scalar based on crop ratio)
        nyquist = 1.0 / (2.0 * pixel_size)
        fraction = crop_res_freq / nyquist
        inner_radius_px = 30 * fraction
        stripe_width_px = int(20 * fraction)
        
        angles = np.arange(-90, 91, 2)
        
        # Create masks
        W_stripe, bandpass_flat = calc.create_mask_matrices(
            (current_h, current_w), [0, y_stripe_px], 
            stripe_width=stripe_width_px, 
            inner_radius=inner_radius_px, 
            outer_radius=outer_radius_px, 
            angles=angles
        )
        
        # Solve
        results = calc.solve_matrix_scoring(fft_avg_grid, W_stripe, bandpass_flat, z_thr)
        filament_patch_count = sum(1 for r in results if r['has_filament'])
        
        save_plot(results, grid_shape, angles, filename, filament_patch_count, config)

        return (filename, filament_patch_count, "Success")

    except Exception as e:
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