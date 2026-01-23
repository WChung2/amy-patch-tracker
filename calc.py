import numpy as np
from skimage.util import view_as_windows

# --- GPU / CPU AUTO-DETECTION ---
try:
    import cupy as xp
    import cupyx.scipy.ndimage as ndi_xp
    from cupy.fft import fft2 as fft2_xp, fftshift as fftshift_xp
    HAS_GPU = True
    # print(":: GPU Detected. Using CuPy for acceleration.")
except ImportError:
    import numpy as xp
    import scipy.ndimage as ndi_xp
    from scipy.fft import fft2 as fft2_xp, fftshift as fftshift_xp
    HAS_GPU = False
    # print(":: CuPy not found. Using CPU (NumPy/SciPy).")

def to_cpu(array):
    """Helper to safely move array back to CPU if it's on GPU."""
    if HAS_GPU and isinstance(array, xp.ndarray):
        return array.get()
    return array

def create_fast_patches_and_fft(image, patch_size, step_size, pixel_size, crop_res_angstrom=4.0):
    """
    Generates 4D patch stack, computes FFT (on GPU if avail), and CROPS it for speed.
    """
    # 1. Vectorized Patching (Always CPU via Skimage - efficient enough)
    patches_cpu = view_as_windows(image, patch_size, step=step_size)
    n_rows, n_cols, h, w = patches_cpu.shape
    
    # 2. Move to GPU (if available)
    patches = xp.array(patches_cpu)
    
    # 3. Vectorized FFT
    fft_4d = fft2_xp(patches, axes=(-2, -1))
    fft_shifted = fftshift_xp(fft_4d, axes=(-2, -1))
    
    # 4. OPTIMIZATION: Crop to 4.0 Angstroms
    # This reduces data size by ~10x, making everything downstream much faster
    freq_limit = 1.0 / crop_res_angstrom
    nyquist = 1.0 / (2.0 * pixel_size)
    fraction = freq_limit / nyquist
    
    # Calculate crop boundaries
    center_y, center_x = h // 2, w // 2
    crop_r = int((min(h, w) // 2) * fraction)
    
    # Slice the massive 4D array
    fft_cropped = fft_shifted[:, :, center_y-crop_r:center_y+crop_r, center_x-crop_r:center_x+crop_r]
    
    magnitude_4d = xp.log1p(xp.abs(fft_cropped))
    
    return magnitude_4d, (n_rows, n_cols)

def fast_average_fft(fft_4d_grid, kernel_size=3):
    """Averages FFT patches with a uniform filter."""
    filter_size = (kernel_size, kernel_size, 1, 1)
    
    # ndi_xp automatically maps to cupyx.scipy.ndimage if GPU is on
    # or scipy.ndimage if CPU is on
    averaged_grid = ndi_xp.uniform_filter(fft_4d_grid, size=filter_size, mode='reflect')
    return averaged_grid

def create_mask_matrices(shape, y_positions, stripe_width, inner_radius, outer_radius, angles):
    """
    Creates vectorized masks for Z-score calculation (GPU-compatible).
    """
    h, w = shape
    n_pixels = h * w
    center_y, center_x = h // 2, w // 2
    
    # xp.ogrid adapts to numpy or cupy
    y, x = xp.ogrid[-center_y:h-center_y, -center_x:w-center_x]
    
    # 1. Bandpass Mask
    radius_sq = x**2 + y**2
    if outer_radius is None:
        bandpass = (radius_sq >= inner_radius**2)
    else:
        bandpass = (radius_sq >= inner_radius**2) & (radius_sq <= outer_radius**2)
    
    bandpass_flat = bandpass.ravel().astype(xp.float32)
    
    n_angles = len(angles)
    W_stripe = xp.zeros((n_angles, n_pixels), dtype=xp.float32)
    half_width = stripe_width // 2
    
    for i, angle in enumerate(angles):
        angle_rad = np.radians(-angle) # np.radians produces scalar, works for both
        cos_a = np.cos(angle_rad)
        sin_a = np.sin(angle_rad)
        
        # This rotation math happens on GPU if xp=cupy
        y_rot = y * cos_a - x * sin_a
        
        # Build binary mask
        mask = xp.zeros((h, w), dtype=bool)
        for y_pos in y_positions:
            if y_pos == 0:
                mask |= (xp.abs(y_rot) <= half_width)
            else:
                mask |= (xp.abs(y_rot - y_pos) <= half_width)
                mask |= (xp.abs(y_rot + y_pos) <= half_width)
        
        # Soften Mask
        m = mask.astype(xp.float32)
        m = ndi_xp.gaussian_filter(m, sigma=2.0)
        if m.max() > 0:
            m /= m.max()
            
        W_stripe[i, :] = m.ravel() * bandpass_flat

    return W_stripe, bandpass_flat

def solve_matrix_scoring(fft_grid, W_stripe, bandpass_flat, z_thr):
    """
    Vectorized Z-score solver (GPU-Accelerated).
    """
    n_rows, n_cols, h, w = fft_grid.shape
    n_pixels = h * w
    
    # Reshape
    F = fft_grid.reshape(-1, n_pixels)
    F2 = F**2
    
    # 1. Global Bandpass Stats (GPU Dot Product)
    Sum_all_bp = xp.dot(F, bandpass_flat)
    Sum_sq_all_bp = xp.dot(F2, bandpass_flat)
    N_all_bp = bandpass_flat.sum()
    
    # 2. Stripe Stats (GPU Dot Product)
    Sum_stripe = xp.dot(F, W_stripe.T)
    Sum_sq_stripe = xp.dot(F2, W_stripe.T)
    N_stripe = W_stripe.sum(axis=1)
    
    Mean_stripe = Sum_stripe / (N_stripe + 1e-10)
    
    # 3. Background Stats (Total - Stripe)
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
    
    # --- Bring results back to CPU for list creation ---
    # We keep data on GPU until the very end to maximize speed
    best_indices_cpu = to_cpu(best_indices)
    best_scores_cpu = to_cpu(best_scores)
    
    results = []
    for i in range(len(best_scores_cpu)):
        results.append({
            'row': i // n_cols,
            'col': i % n_cols,
            'fit_score': float(best_scores_cpu[i]),
            'best_angle_idx': int(best_indices_cpu[i]),
            'has_filament': best_scores_cpu[i] > z_thr
        })
        
    return results