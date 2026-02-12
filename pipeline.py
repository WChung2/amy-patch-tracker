import os
import mrcfile
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import scipy.ndimage as ndi 

import calc 

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
    pixel_size = config.get('pixel_size', 0.83)
    periodicity = config.get('periodicity', 4.7)
    
    z_thr = config.get('threshold', 0.25)
    config['threshold'] = z_thr 
    
    outer_res = config.get('outer_resolution', 4.0)
    
    if calc.HAS_GPU and cupy is not None:
        try:
            n_devices = cupy.cuda.runtime.getDeviceCount()
            if n_devices > 0:
                pid = os.getpid()
                device_id = pid % n_devices
                cupy.cuda.Device(device_id).use()
        except Exception:
            pass
    
    filename = os.path.basename(file_path)
    
    try:
        data = load_mrc_file(file_path)
        if data is None: return (filename, 0, "Load Error")
        
        h, w = data.shape
        if h < 200 or w < 200: return (filename, 0, "Too Small")
        
        base_patch_h = h // 8
        base_patch_w = w // 8
        patch_size = (base_patch_h, base_patch_w)
        
        step_size = (int(base_patch_h * 0.2), int(base_patch_w * 0.2))
        step_size = (max(1, step_size[0]), max(1, step_size[1]))
        
        fft_grid, grid_shape = calc.create_fast_patches_and_fft(
            data, patch_size, step_size, pixel_size, outer_res
        )
        
        fft_avg_grid = calc.fast_average_fft(fft_grid, kernel_size=3)
        
        current_h, current_w = fft_avg_grid.shape[-2:]
        crop_res_freq = 1.0 / outer_res
        
        freq_val = 1.0 / periodicity
        y_stripe_px = (freq_val / crop_res_freq) * (current_h / 2)
        outer_radius_px = (current_h / 2) - 1
        
        nyquist = 1.0 / (2.0 * pixel_size)
        fraction = crop_res_freq / nyquist
        inner_radius_px = 30 * fraction
        stripe_width_px = int(20 * fraction)
        
        angles = np.arange(-90, 91, 2)
        
        W_stripe, bandpass_flat = calc.create_mask_matrices(
            (current_h, current_w), [0, y_stripe_px], 
            stripe_width=stripe_width_px, 
            inner_radius=inner_radius_px, 
            outer_radius=outer_radius_px, 
            angles=angles
        )
        
        results = calc.solve_matrix_scoring(fft_avg_grid, W_stripe, bandpass_flat, z_thr, angles)
        
        # 1. Create Overlap Map
        overlap_map = calc.calculate_overlap_map(results, grid_shape, patch_size, step_size, (h, w))
        
        # 2. Process to Skeleton (Returns TWO maps now)
        binary_mask, skeleton_map = calc.process_overlap_to_skeleton(overlap_map)
        
        filament_patch_count = sum(1 for r in results if r['has_filament'])
        dims = {'h': h, 'w': w, 'step': step_size, 'patch': patch_size}
        
        save_plot(data, overlap_map, binary_mask, skeleton_map, results, grid_shape, filename, filament_patch_count, config, dims)

        return (filename, filament_patch_count, "Success")

    except Exception as e:
        import traceback
        traceback.print_exc()
        return (filename, 0, str(e))

def save_plot(data, overlap_map, binary_mask, skeleton_map, results, grid_shape, filename, count, config, dims):
    """
    Plots 5 Panels:
    1. Real Space + SKELETON (Red)
    2. Grid Heatmap + VECTORS (Cyan)
    3. Overlap Density + VECTORS (Cyan)
    4. Binary Mask + VECTORS (Cyan)
    5. Skeleton
    """
    fit_scores_grid = np.zeros(grid_shape)
    
    quiver_y = []
    quiver_x = []
    quiver_u = [] 
    quiver_v = [] 
    
    step_y, step_x = dims['step']
    patch_h, patch_w = dims['patch']
    start_y = patch_h // 2
    start_x = patch_w // 2

    for r in results:
        r_idx, c_idx = r['row'], r['col']
        score = r['fit_score']
        fit_scores_grid[r_idx, c_idx] = score
        
        if r['has_filament']:
            y_pos = start_y + (r_idx * step_y)
            x_pos = start_x + (c_idx * step_x)
            
            angle_rad = np.radians(r['angle']) 
            u = np.cos(angle_rad)
            v = np.sin(angle_rad) 
            
            quiver_y.append(y_pos)
            quiver_x.append(x_pos)
            quiver_u.append(u)
            quiver_v.append(v)

    fig, axes = plt.subplots(1, 5, figsize=(50, 10))
    
    # Pre-calculate dilated skeleton for display
    if skeleton_map.max() > 0:
        vis_skeleton = ndi.binary_dilation(skeleton_map > 0, iterations=3).astype(float)
    else:
        vis_skeleton = skeleton_map
        
    # Create masked skeleton (transparent where 0) for overlay
    skeleton_overlay = np.ma.masked_where(vis_skeleton == 0, vis_skeleton)

    # --- 1. Real Space (Bin 4) + SKELETON ---
    bin_n = 4
    h_orig, w_orig = data.shape
    h_crop = (h_orig // bin_n) * bin_n
    w_crop = (w_orig // bin_n) * bin_n
    
    data_binned = data[:h_crop, :w_crop].reshape(
        h_crop // bin_n, bin_n, 
        w_crop // bin_n, bin_n
    ).mean(axis=(1, 3))

    vmin, vmax = np.percentile(data_binned, [1, 99])
    axes[0].imshow(data_binned, cmap='gray', vmin=vmin, vmax=vmax, 
                   extent=[0, w_orig, h_orig, 0], origin='upper', interpolation='bilinear')
    # Overlay Skeleton in Red
    axes[0].imshow(skeleton_overlay, cmap='Reds', interpolation='nearest',
                   extent=[0, w_orig, h_orig, 0], origin='upper', vmin=0, vmax=1, alpha=0.7)
    axes[0].set_title(f"Original + Skeleton (Red)")
    
    # --- 2. Grid Heatmap + VECTORS ---
    im = axes[1].imshow(fit_scores_grid, cmap='hot', interpolation='bilinear', 
                   extent=[0, w_orig, h_orig, 0], origin='upper')
    axes[1].set_title(f"Grid Heatmap")
    plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)

    # --- 3. Overlap Density + VECTORS ---
    im_ov = axes[2].imshow(overlap_map, cmap='inferno', interpolation='nearest',
                   extent=[0, w_orig, h_orig, 0], origin='upper')
    axes[2].set_title("Overlap Density")
    plt.colorbar(im_ov, ax=axes[2], fraction=0.046, pad=0.04)

    # --- 4. Binary Mask + VECTORS ---
    axes[3].imshow(binary_mask, cmap='gray', interpolation='nearest',
                   extent=[0, w_orig, h_orig, 0], origin='upper')
    axes[3].set_title("Binary Ridge (Smoothed)")

    # --- 5. Skeleton ---
    axes[4].imshow(vis_skeleton, cmap='gray', interpolation='nearest',
                   extent=[0, w_orig, h_orig, 0], origin='upper')
    axes[4].set_title("Filament Bone (Skeleton)")

    # Overlay vectors ONLY on panels 2, 3, 4
    if quiver_x:
        for ax in [axes[1], axes[2], axes[3]]:
            ax.quiver(quiver_x, quiver_y, quiver_u, quiver_v, 
                      color='cyan', headaxislength=0, headlength=0, 
                      pivot='mid', scale=40, width=0.005, alpha=1.0)

    output_name = filename.replace('.mrc', '_heatmap.png')
    out_dir = config['picked_dir'] if count >= config['min_patches'] else config['trashed_dir']
    
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, output_name), dpi=150)
    plt.close(fig)