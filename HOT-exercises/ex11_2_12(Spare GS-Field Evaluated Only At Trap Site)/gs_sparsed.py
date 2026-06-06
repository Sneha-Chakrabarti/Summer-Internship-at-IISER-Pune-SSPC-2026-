"""
Exercise 11.2.12
More efficient GS: compute the focal field only at trap sites.

Reference: Jones, Maragò & Volpe, Optical Tweezers: Principles and Applications, Ch. 7
Notes: SSPC 2026, Sneha Chakrabarti

Motivation (from exercise statement)
--------------------------------------
In standard GS the full FFT is computed at every iteration even though, for a
point-trap target, only N out of Mx*My focal pixels have non-zero target
amplitude. The remaining pixels are zeroed before back-propagation anyway, so
computing them is pure waste.

Alternative: replace the FFT with a direct DFT evaluated only at the N trap
frequencies. The SLM-to-focal-plane mapping is (Eq. 3 + Eq. 4, theory notes):

    E_focus(x_n, y_n) = sum_{m} A_m exp(i phi_m) exp(-i Delta_m(x_n, y_n))

where Delta_m is the lateral steering phase (Eq. 5, z=0):

    Delta_m(x_n, y_n) = (2pi / lambda f) * (x_m * x_n + y_m * y_n)

This is a matrix-vector product once we write it as

    E_n = W_{nm} * slm_field_m,    W_{nm} = exp(-i 2pi (m_x n_x + m_y n_y) / M)

i.e. the DFT matrix evaluated at fractional frequencies (n_x/Mx, n_y/My).

Back-propagation from N trap sites to the full SLM is:

    slm_back_m = sum_n E_n^corrected * exp(+i 2pi (m_x n_x + m_y n_y) / M)

This is the adjoint (conjugate transpose) of W, so the back-propagation is
just W^H applied to the corrected focal vector. Crucially the IFFT is not
needed: we only need the phase of slm_back at each SLM pixel, and that is
computed from the same matrix W.

Complexity comparison
---------------------
Standard GS  (FFT):  O(M log M)  per iteration  (M = Mx*My)
Sparse GS    (DFT):  O(M * N)    per iteration
For N << log M sparse GS wins. For the standard 1000x1000 / N=100 benchmark,
log2(1000^2) ≈ 20, so sparse is faster only when N < 20. At N=100 the full
FFT is cheaper in FLOPS, but this exercise demonstrates the approach
conceptually and the sparse method is exact (no aliasing artefacts).
For very small N (e.g. N=2-5) sparse GS is clearly the winner.
"""

import numpy as np
import matplotlib.pyplot as plt
import time
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "ex11_2_11"))
from gs_algorithm import (gaussian_laser, make_trap_mask,
                          trap_intensities, metrics, gerchberg_saxton)

rng = np.random.default_rng(42)

# ── standard benchmark parameters ────────────────────────────────────────────
# SLM  : 1000 × 1000 pixels  (standard task from Section 5.1, theory notes)
# Traps: N = 100, 10 × 10 lattice, z = 0
# Laser: Gaussian 1/e² radius 400 px (~80 % aperture fill)
# Note : the full DFT matrix at 1000×1000 is (100 × 10^6) complex128 ~ 1.6 GB.
#        Convergence comparison runs at 1000×1000; timing study uses a
#        reduced 512×512 grid so the matrix fits in RAM during the N-sweep.
Mx, My  = 1000, 1000
N_ITER  = 60
LASER_W = 400
FIGURES = Path(__file__).parent / "figures"
FIGURES.mkdir(exist_ok=True)


# ── pixel coordinate grids ────────────────────────────────────────────────────

def pixel_coords(Mx, My):
    """Centred pixel indices for SLM and focal plane."""
    mx = np.arange(Mx) - Mx // 2   # shape (Mx,)
    my = np.arange(My) - My // 2   # shape (My,)
    return mx, my


# ── sparse DFT helpers ────────────────────────────────────────────────────────

def build_dft_matrix(slm_mx, slm_my, trap_coords, Mx, My):
    """
    Build the (N, Mx*My) DFT matrix W such that
        E_focal[n] = W[n, :] @ slm_field.ravel()
    where W[n, m] = exp(-i 2pi (mx_m * nx_n / Mx + my_m * ny_n / My)).

    trap_coords : (N, 2) pixel indices in the UNSHIFTED focal-plane array.
    Returns W of shape (N, Mx*My), complex128.
    """
    # focal pixel indices relative to centre
    nx = trap_coords[:, 0] - Mx // 2   # (N,)
    ny = trap_coords[:, 1] - My // 2   # (N,)

    # SLM pixel indices (flattened)
    MX, MY = np.meshgrid(slm_mx, slm_my, indexing='ij')  # (Mx, My)
    mx_flat = MX.ravel()   # (Mx*My,)
    my_flat = MY.ravel()

    # phase matrix: (N, Mx*My)
    phase = -2j * np.pi * (
        np.outer(nx, mx_flat) / Mx +
        np.outer(ny, my_flat) / My
    )
    W = np.exp(phase)
    return W       # (N, M)


def sparse_gs(laser_amp, trap_coords, I_target_per_trap, n_iter,
              seed_phase=None):
    """
    Sparse Gerchberg-Saxton: DFT evaluated only at N trap sites.

    Forward  step:  E_focal[n]  = W  @ slm_field.ravel()   (N evals)
    Backward step:  slm_back    = W^H @ E_corrected          (M evals)

    The rest of GS is identical to the standard version.
    """
    Mx, My = laser_amp.shape
    N      = len(trap_coords)
    mx, my = pixel_coords(Mx, My)

    if np.isscalar(I_target_per_trap):
        A_target = np.full(N, np.sqrt(float(I_target_per_trap)))
    else:
        A_target = np.sqrt(np.asarray(I_target_per_trap, dtype=float))

    if seed_phase is None:
        seed_phase = rng.uniform(0, 2 * np.pi, (Mx, My))

    # pre-build DFT matrix (N x M) — expensive once, cheap thereafter
    print("  Building DFT matrix... ", end="", flush=True)
    W = build_dft_matrix(mx, my, trap_coords, Mx, My)   # (N, M)
    print("done.")

    slm_field = laser_amp * np.exp(1j * seed_phase)     # (Mx, My)
    laser_flat = laser_amp.ravel()

    history = {k: [] for k in ['I_tot', 'I_mean', 'u', 'sigma']}

    for _ in range(n_iter):
        # 1. forward: focal amplitudes at trap sites only
        E_focal = W @ slm_field.ravel()                  # (N,)

        # record metrics
        I_traps = np.abs(E_focal)**2
        for k, v in zip(['I_tot', 'I_mean', 'u', 'sigma'], metrics(I_traps)):
            history[k].append(v)

        # 2. replace amplitude, keep phase
        E_corrected = A_target * np.exp(1j * np.angle(E_focal))   # (N,)

        # 3. backward: adjoint DFT  W^H @ E_corrected -> SLM plane
        slm_back_flat = W.conj().T @ E_corrected         # (M,)  = W^H E

        # 4. replace SLM amplitude with laser profile
        slm_back = slm_back_flat.reshape(Mx, My)
        slm_field = laser_amp * np.exp(1j * np.angle(slm_back))

    return np.angle(slm_field), history


# ── timing comparison across N ────────────────────────────────────────────────

def timing_study():
    """Compare wall-clock time of standard vs sparse GS as N varies.
    Uses a reduced 512×512 grid so the full DFT matrix (N × M) fits in RAM
    across all N values. The main convergence comparison runs at 1000×1000.
    """
    Mx_t, My_t = 512, 512      # reduced grid for timing only
    laser_amp = gaussian_laser(Mx_t, My_t, 200)
    N_values  = [4, 9, 25, 49, 100]
    N_ITER_T  = 20   # fewer iterations for timing fairness

    t_fft, t_sparse = [], []

    for N in N_values:
        n_side = int(np.round(np.sqrt(N)))
        coords = make_trap_mask(Mx_t, My_t, N_side=n_side)[:N]
        I_tgt  = 1.0 / len(coords)

        # standard GS
        t0 = time.perf_counter()
        gerchberg_saxton(laser_amp, coords, I_tgt, N_ITER_T)
        t_fft.append(time.perf_counter() - t0)

        # sparse GS
        t0 = time.perf_counter()
        sparse_gs(laser_amp, coords, I_tgt, N_ITER_T)
        t_sparse.append(time.perf_counter() - t0)

        print(f"  N={N:4d}:  FFT={t_fft[-1]:.2f}s   sparse={t_sparse[-1]:.2f}s")

    return N_values, t_fft, t_sparse


def run_and_plot():
    laser_amp   = gaussian_laser(Mx, My, LASER_W)
    trap_coords = make_trap_mask(Mx, My, N_side=10)
    N           = len(trap_coords)
    I_target    = 1.0 / N

    print("Running sparse GS (N=100)...")
    slm_phase_sparse, hist_sparse = sparse_gs(
        laser_amp, trap_coords, I_target, N_ITER
    )

    print("Running standard GS (N=100) for comparison...")
    slm_phase_std, hist_std = gerchberg_saxton(
        laser_amp, trap_coords, I_target, N_ITER
    )

    iters = np.arange(1, N_ITER + 1)

    # ── Figure 1: convergence comparison ────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    fig.suptitle("Sparse vs Standard GS: Convergence (Ex 11.2.12)", fontsize=13)

    for ax, key, ylabel in [
        (axes[0], 'u',     r'Uniformity $u$'),
        (axes[1], 'sigma', r'$\sigma$ (%)'),
    ]:
        ax.plot(iters, hist_std[key],    lw=1.8, label='Standard GS (FFT)', color='steelblue')
        ax.plot(iters, hist_sparse[key], lw=1.8, label='Sparse GS (DFT)',   color='tomato', ls='--')
        ax.set_xlabel("Iteration", fontsize=10)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(FIGURES / "fig1_sparse_vs_standard_convergence.png", dpi=150)
    plt.close(fig)

    # ── Figure 2: timing study ────────────────────────────────────────────────
    print("\nTiming study...")
    N_vals, t_fft, t_sparse = timing_study()

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(N_vals, t_fft,    'o-', color='steelblue', lw=1.8, label='Standard GS (FFT)')
    ax.plot(N_vals, t_sparse, 's--', color='tomato',   lw=1.8, label='Sparse GS (DFT)')
    ax.set_xlabel("Number of traps N", fontsize=11)
    ax.set_ylabel(f"Wall time ({20} iterations, s)", fontsize=11)
    ax.set_title("Computational cost vs N\n(512×512 grid; timing study only)", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(FIGURES / "fig2_timing_vs_N.png", dpi=150)
    plt.close(fig)

    # ── Figure 3: focal-plane intensity (sparse result) ───────────────────────
    slm_field   = laser_amp * np.exp(1j * slm_phase_sparse)
    focal_field = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(slm_field)))
    focal_int   = np.abs(focal_field)**2

    fig, ax = plt.subplots(figsize=(5, 5))
    log_int = np.log1p(focal_int / focal_int.max() * 1e4)
    ax.imshow(log_int.T, cmap='inferno', origin='lower')
    ax.scatter(trap_coords[:, 0], trap_coords[:, 1],
               s=10, c='cyan', marker='x', linewidths=0.8)
    ax.set_title("Sparse GS: focal-plane intensity (log scale)", fontsize=11)
    ax.set_xlabel("Focal pixel x")
    ax.set_ylabel("Focal pixel y")
    plt.tight_layout()
    fig.savefig(FIGURES / "fig3_focal_intensity_sparse.png", dpi=150)
    plt.close(fig)

    print("\nFigures saved to", FIGURES)


if __name__ == "__main__":
    run_and_plot()
