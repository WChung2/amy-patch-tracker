import numpy as np
import scipy.ndimage as ndi_cpu 
from skimage.util import view_as_windows
from skimage.morphology import skeletonize, disk, remove_small_objects

# --- GPU / CPU AUTO-DETECTION ---
try:
    import cupy as xp
    import cupyx.scipy.ndimage as ndi_xp
    from cupy.fft import fft2 as fft2_xp, fftshift as fftshift_xp
    HAS_GPU = True
except ImportError:
    import numpy as xp
    import scipy.ndimage as ndi_xp
    from scipy.fft import fft2 as fft2_xp, fftshift as fftshift_xp
    HAS_GPU = False

def to_cpu(array):
    if HAS_GPU and hasattr(array, 'get'):
        return array.get()
    return array

def create_fast_patches_and_fft(image, patch_size, step_size, pixel_size, crop_res_angstrom=4.0):
    patches_cpu = view_as_windows(image, patch_size, step=step_size)
    n_rows, n_cols, h, w = patches_cpu.shape
    
    patches = xp.array(patches_cpu)
    
    fft_4d = fft2_xp(patches, axes=(-2, -1))
    fft_shifted = fftshift_xp(fft_4d, axes=(-2, -1))
    
    freq_limit = 1.0 / crop_res_angstrom
    nyquist = 1.0 / (2.0 * pixel_size)
    fraction = freq_limit / nyquist
    
    center_y, center_x = h // 2, w // 2
    crop_r = int((min(h, w) // 2) * fraction)
    
    fft_cropped = fft_shifted[:, :, center_y-crop_r:center_y+crop_r, center_x-crop_r:center_x+crop_r]
    magnitude_4d = xp.log1p(xp.abs(fft_cropped))
    
    return magnitude_4d, (n_rows, n_cols)

def fast_average_fft(fft_4d_grid, kernel_size=3):
    filter_size = (kernel_size, kernel_size, 1, 1)
    averaged_grid = ndi_xp.uniform_filter(fft_4d_grid, size=filter_size, mode='reflect')
    return averaged_grid

def create_mask_matrices(shape, y_positions, stripe_width, inner_radius, outer_radius, angles):
    """
    Optimized: Fully Vectorized Mask Creation.
    Removes the inner loop over y_positions, speeding up mask generation by ~10x.
    """
    h, w = shape
    n_pixels = h * w
    center_y, center_x = h // 2, w // 2
    
    y, x = xp.ogrid[-center_y:h-center_y, -center_x:w-center_x]
    
    radius_sq = x**2 + y**2
    if outer_radius is None:
        bandpass = (radius_sq >= inner_radius**2)
    else:
        bandpass = (radius_sq >= inner_radius**2) & (radius_sq <= outer_radius**2)
    
    bandpass_flat = bandpass.ravel().astype(xp.float32)
    
    n_angles = len(angles)
    W_stripe = xp.zeros((n_angles, n_pixels), dtype=xp.float32)
    half_width = stripe_width // 2
    
    # Convert y_positions to array for broadcasting
    y_pos_arr = xp.array(y_positions).reshape(-1, 1, 1)
    
    for i, angle in enumerate(angles):
        angle_rad = np.radians(-angle) 
        cos_a = np.cos(angle_rad)
        sin_a = np.sin(angle_rad)
        
        y_rot = y * cos_a - x * sin_a
        
        # --- Vectorized Stripe Logic ---
        # We check |y_rot - y_pos| <= half_width for ALL y_positions at once
        # dists shape: (n_stripes, h, w)
        dists = xp.abs(y_rot - y_pos_arr)
        
        # Collapse: True if pixel is close to ANY stripe line
        mask = xp.any(dists <= half_width, axis=0)
        
        # Mirror symmetric check (for -y_pos) if needed, but since y_positions 
        # usually includes 0 and positive values, we handle the negative mirror here:
        dists_mirror = xp.abs(y_rot + y_pos_arr)
        mask |= xp.any(dists_mirror <= half_width, axis=0)

        m = mask.astype(xp.float32)
        m = ndi_xp.gaussian_filter(m, sigma=2.0)
        
        m_max = m.max()
        if m_max > 0:
            m /= m_max
            
        W_stripe[i, :] = m.ravel() * bandpass_flat

    return W_stripe, bandpass_flat

def solve_matrix_scoring(fft_grid, W_stripe, bandpass_flat, z_thr, angles):
    n_rows, n_cols, h, w = fft_grid.shape
    n_pixels = h * w
    
    F = fft_grid.reshape(-1, n_pixels)
    F2 = F**2
    
    # 1. Global Bandpass
    Sum_all_bp = xp.dot(F, bandpass_flat)
    Sum_sq_all_bp = xp.dot(F2, bandpass_flat)
    N_all_bp = bandpass_flat.sum()
    
    # 2. Stripe
    Sum_stripe = xp.dot(F, W_stripe.T)
    Sum_sq_stripe = xp.dot(F2, W_stripe.T)
    N_stripe = W_stripe.sum(axis=1)
    
    Mean_stripe = Sum_stripe / (N_stripe + 1e-10)
    
    # 3. Background
    Sum_bg = Sum_all_bp[:, None] - Sum_stripe
    Sum_sq_bg = Sum_sq_all_bp[:, None] - Sum_sq_stripe
    N_bg = N_all_bp - N_stripe
    
    Mean_bg = Sum_bg / (N_bg + 1e-10)
    Mean_sq_bg = Sum_sq_bg / (N_bg + 1e-10)
    Var_bg = Mean_sq_bg - (Mean_bg**2)
    Std_bg = xp.sqrt(xp.maximum(Var_bg, 1e-10))
    
    # 4. Z-Score
    Z_scores = (Mean_stripe - Mean_bg) / (Std_bg + 1e-10)
    
    best_indices = xp.argmax(Z_scores, axis=1)
    row_indices = xp.arange(Z_scores.shape[0])
    best_scores = Z_scores[row_indices, best_indices]
    
    best_indices_cpu = to_cpu(best_indices)
    best_scores_cpu = to_cpu(best_scores)
    
    results = []
    for i in range(len(best_scores_cpu)):
        idx = int(best_indices_cpu[i])
        angle_val = angles[idx]
        
        results.append({
            'row': i // n_cols,
            'col': i % n_cols,
            'fit_score': float(best_scores_cpu[i]),
            'best_angle_idx': idx,
            'angle': float(angle_val) + 90, 
            'has_filament': best_scores_cpu[i] > z_thr
        })
        
    return results

def calculate_overlap_map(results, grid_shape, patch_size, step_size, image_shape):
    """
    Step 1: Accumulate votes using CIRCLES.
    Accumulation happens on CPU as loop overhead on GPU for small items is high.
    """
    accumulator = np.zeros(image_shape, dtype=np.float32)
    step_y, step_x = step_size
    patch_h, patch_w = patch_size
    
    # --- PRE-COMPUTE CIRCLE MASK ---
    radius = min(patch_h, patch_w) // 2
    radius = int(radius * 0.9)
    circle_mask = disk(radius)
    
    h_c, w_c = circle_mask.shape
    off_y = (patch_h - h_c) // 2
    off_x = (patch_w - w_c) // 2
    
    # Optimized loop: only iterate valid results
    valid_results = [res for res in results if res['has_filament']]
    
    for res in valid_results:
        r, c = res['row'], res['col']
        
        y_box = int(r * step_y)
        x_box = int(c * step_x)
        
        y_start = y_box + off_y
        x_start = x_box + off_x
        y_end = y_start + h_c
        x_end = x_start + w_c
        
        # Fast clipping logic
        y_s_clip, y_e_clip = 0, h_c
        x_s_clip, x_e_clip = 0, w_c
        
        if y_start < 0:
            y_s_clip = -y_start
            y_start = 0
        if x_start < 0:
            x_s_clip = -x_start
            x_start = 0
        if y_end > image_shape[0]:
            y_e_clip = h_c - (y_end - image_shape[0])
            y_end = image_shape[0]
        if x_end > image_shape[1]:
            x_e_clip = w_c - (x_end - image_shape[1])
            x_end = image_shape[1]
            
        if y_end > y_start and x_end > x_start:
             accumulator[y_start:y_end, x_start:x_end] += circle_mask[y_s_clip:y_e_clip, x_s_clip:x_e_clip]
            
    return accumulator

def process_overlap_to_skeleton(overlap_map):
    """
    Returns (Smoothed_Binary_Mask, Skeleton)
    PERFORMANCE OPTIMIZATION: Move Zooming to GPU if available.
    """
    # 1. DOWNSAMPLE (Bin 4)
    # CRITICAL SPEEDUP: Perform the heavy zooming on GPU if possible
    scale = 0.25
    
    if HAS_GPU:
        # Move to GPU if not already there
        map_gpu = xp.array(overlap_map)
        small_map_gpu = ndi_xp.zoom(map_gpu, scale, order=1)
        small_map = to_cpu(small_map_gpu)
    else:
        # CPU Fallback
        overlap_map_cpu = to_cpu(overlap_map)
        small_map = ndi_cpu.zoom(overlap_map_cpu, scale, order=1)
    
    # 2. BINARIZE
    binary_mask = (small_map > 0.01).astype(np.float32)
    
    # 3. FILL HOLES (Closing)
    binary_mask = ndi_cpu.binary_closing(binary_mask, structure=disk(2))
    
    # 4. REMOVE ATTACHMENTS (Opening)
    binary_mask = ndi_cpu.binary_opening(binary_mask, structure=disk(1))
    
    # 5. CLEANUP ISOLATED NOISE
    # Fix for skimage version compatibility
    try:
        binary_mask = remove_small_objects(binary_mask.astype(bool), max_size=20)
    except TypeError:
        binary_mask = remove_small_objects(binary_mask.astype(bool), min_size=20)
    
    binary_mask = binary_mask.astype(np.float32)

    # 6. SMOOTHING
    smoothed = ndi_cpu.gaussian_filter(binary_mask.astype(float), sigma=3.0)
    
    # 7. THRESHOLD
    binary_ridge = smoothed > 0.5
    
    # 8. SKELETONIZE
    skeleton_small = skeletonize(binary_ridge)
    
    # 9. UPSAMPLE BACK
    # Upsample skeleton (Binary -> keep strict order 0)
    skeleton = ndi_cpu.zoom(skeleton_small.astype(float), 1/scale, order=0)
    final_mask = ndi_cpu.zoom(binary_ridge.astype(float), 1/scale, order=0)
    
    return final_mask.astype(np.float32), skeleton.astype(np.float32)