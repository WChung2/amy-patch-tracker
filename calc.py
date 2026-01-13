import numpy as np
from scipy.ndimage import uniform_filter, gaussian_filter
from scipy.fft import fft2, fftshift
from skimage.util import view_as_windows

def create_fast_patches_and_fft(image, patch_size, step_size):
    """Generates 4D patch stack and computes FFT."""
    patches_4d = view_as_windows(image, patch_size, step=step_size)
    n_rows, n_cols, h, w = patches_4d.shape
    
    # Batched FFT
    fft_4d = fft2(patches_4d, axes=(-2, -1))
    fft_shifted = fftshift(fft_4d, axes=(-2, -1))
    magnitude_4d = np.log1p(np.abs(fft_shifted))
    
    return magnitude_4d, (n_rows, n_cols)

def fast_average_fft(fft_4d_grid, kernel_size=3):
    """Averages FFT patches with a uniform filter."""
    filter_size = (kernel_size, kernel_size, 1, 1)
    averaged_grid = uniform_filter(fft_4d_grid, size=filter_size, mode='reflect')
    return averaged_grid

def create_mask_matrices(shape, y_positions, stripe_width, inner_radius, outer_radius, angles):
    """
    Creates vectorized masks for Z-score calculation.
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
    
    bandpass_flat = bandpass.ravel().astype(np.float32)
    
    n_angles = len(angles)
    W_stripe = np.zeros((n_angles, n_pixels), dtype=np.float32)
    half_width = stripe_width // 2
    
    for i, angle in enumerate(angles):
        angle_rad = np.radians(-angle)
        cos_a = np.cos(angle_rad)
        sin_a = np.sin(angle_rad)
        y_rot = y * cos_a - x * sin_a
        
        # Build binary mask
        mask = np.zeros((h, w), dtype=bool)
        for y_pos in y_positions:
            if y_pos == 0:
                mask |= (np.abs(y_rot) <= half_width)
            else:
                mask |= (np.abs(y_rot - y_pos) <= half_width)
                mask |= (np.abs(y_rot + y_pos) <= half_width)
        
        # Soften Mask
        m = mask.astype(np.float32)
        m = gaussian_filter(m, sigma=2.0)
        if m.max() > 0:
            m /= m.max()
            
        W_stripe[i, :] = m.ravel() * bandpass_flat

    return W_stripe, bandpass_flat

def solve_matrix_scoring(fft_grid, W_stripe, bandpass_flat, z_thr):
    """
    Vectorized Z-score solver.
    """
    n_rows, n_cols, h, w = fft_grid.shape
    n_pixels = h * w
    
    F = fft_grid.reshape(-1, n_pixels)
    F2 = F**2
    
    # 1. Global Bandpass Stats
    Sum_all_bp = np.dot(F, bandpass_flat)
    Sum_sq_all_bp = np.dot(F2, bandpass_flat)
    N_all_bp = bandpass_flat.sum()
    
    # 2. Stripe Stats
    Sum_stripe = np.dot(F, W_stripe.T)
    Sum_sq_stripe = np.dot(F2, W_stripe.T)
    N_stripe = W_stripe.sum(axis=1)
    
    Mean_stripe = Sum_stripe / (N_stripe + 1e-10)
    
    # 3. Background Stats (Total - Stripe)
    Sum_bg = Sum_all_bp[:, None] - Sum_stripe
    Sum_sq_bg = Sum_sq_all_bp[:, None] - Sum_sq_stripe
    N_bg = N_all_bp - N_stripe
    
    Mean_bg = Sum_bg / (N_bg + 1e-10)
    Mean_sq_bg = Sum_sq_bg / (N_bg + 1e-10)
    Var_bg = Mean_sq_bg - (Mean_bg**2)
    Std_bg = np.sqrt(np.maximum(Var_bg, 1e-10))
    
    # 4. Z-Score
    Z_scores = (Mean_stripe - Mean_bg) / (Std_bg + 1e-10)
    
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