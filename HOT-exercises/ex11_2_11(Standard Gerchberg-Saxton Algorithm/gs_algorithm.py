"""
Exercise 11.2.11
Implement the Gerchberg-Saxton algorithm and study its performance.

Reference: Jones, Maragò & Volpe, Optical Tweezers: Principles and Applications, Ch. 7
Notes: SSPC 2026, Sneha Chakrabarti

Physical setup
--------------
SLM (Mx x My pixels) conjugated to back focal plane of objective.
Focal-plane field = Fourier transform of SLM exit field (Eq. 3, theory notes).
GS iterates between DOE plane and focal plane, enforcing amplitude constraints
at each plane while keeping the phase free.

Algorithm (per iteration):
  1. Forward FFT of current SLM field  -> focal field
  2. Replace focal amplitude with sqrt(I_target), keep focal phase
  3. Inverse FFT                        -> updated SLM field
  4. Replace SLM amplitude with laser profile, keep SLM phase

Performance metrics (from notes, Section 5.1):
  - Total intensity  I_tot = sum(I_n)
  - Average intensity <I>  = I_tot / N
  - Uniformity       u     = 1 - (max-min)/(max+min)
  - % std error      sigma = 100 * std(I_n) / <I>
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path

# ── reproducibility ──────────────────────────────────────────────────────────
rng = np.random.default_rng(42)

# ── standard benchmark parameters (Section 5.1, theory notes) ────────────────
# SLM  : 1000 × 1000 pixels
# Traps: N = 100 on a 10 × 10 square lattice, z = 0 (focal plane)
# Laser: Gaussian, 1/e² radius = 400 px (fills ~80 % of aperture width)
Mx, My   = 1000, 1000      # SLM pixel grid
N_TRAPS  = 100             # 10×10 lattice → N=100
N_ITER   = 60              # GS iterations
LASER_W  = 400             # Gaussian 1/e² radius in pixels

FIGURES  = Path(__file__).parent / "figures"
FIGURES.mkdir(exist_ok=True)


# ── helpers ───────────────────────────────────────────────────────────────────

def gaussian_laser(Mx, My, w):
    """Normalised Gaussian amplitude profile on the SLM plane."""
    cx, cy = Mx // 2, My // 2
    x = np.arange(Mx) - cx
    y = np.arange(My) - cy
    X, Y = np.meshgrid(x, y, indexing='ij')
    A = np.exp(-(X**2 + Y**2) / (2 * w**2))
    return A / np.sqrt(np.sum(A**2))


def make_trap_mask(Mx, My, N_side=10, margin=0.15):
    """
    Return (N, 2) integer array of trap pixel coords on a uniform lattice,
    and a boolean focal-plane mask of shape (Mx, My).
    Traps are placed symmetrically around the centre with a margin.
    """
    cx, cy = Mx // 2, My // 2
    span_x = int(Mx * (1 - 2 * margin))
    span_y = int(My * (1 - 2 * margin))
    xs = np.linspace(cx - span_x // 2, cx + span_x // 2, N_side, dtype=int)
    ys = np.linspace(cy - span_y // 2, cy + span_y // 2, N_side, dtype=int)
    coords = np.array([[x, y] for x in xs for y in ys])   # (N, 2)
    return coords


def trap_intensities(focal_field, coords):
    """Extract intensity at each trap site."""
    return np.array([np.abs(focal_field[c[0], c[1]])**2 for c in coords])


def metrics(intensities):
    """Compute the four performance metrics from the theory notes."""
    I      = intensities
    I_tot  = I.sum()
    I_mean = I_tot / len(I)
    u      = 1 - (I.max() - I.min()) / (I.max() + I.min())
    sigma  = 100 * I.std() / I_mean
    return I_tot, I_mean, u, sigma


# ── GS algorithm ─────────────────────────────────────────────────────────────

def gerchberg_saxton(laser_amp, trap_coords, I_target_per_trap, n_iter,
                     seed_phase=None):
    """
    Standard Gerchberg-Saxton algorithm.

    Parameters
    ----------
    laser_amp          : (Mx, My) array, SLM amplitude constraint (laser profile)
    trap_coords        : (N, 2) int array of trap pixel indices in focal plane
    I_target_per_trap  : scalar or (N,) array, target intensity at each trap
    n_iter             : number of iterations
    seed_phase         : (Mx, My) initial SLM phase; random if None

    Returns
    -------
    slm_phase   : (Mx, My) final hologram phase
    history     : dict of metric arrays, one value per iteration
    """
    Mx, My = laser_amp.shape
    N = len(trap_coords)

    # target focal amplitude
    if np.isscalar(I_target_per_trap):
        A_target = np.full(N, np.sqrt(float(I_target_per_trap)))
    else:
        A_target = np.sqrt(np.asarray(I_target_per_trap, dtype=float))

    # initialise SLM field
    if seed_phase is None:
        seed_phase = rng.uniform(0, 2 * np.pi, (Mx, My))
    slm_field = laser_amp * np.exp(1j * seed_phase)

    history = {k: [] for k in ['I_tot', 'I_mean', 'u', 'sigma']}

    for _ in range(n_iter):
        # 1. forward FFT -> focal field
        focal_field = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(slm_field)))

        # record metrics BEFORE amplitude replacement
        I_traps = trap_intensities(focal_field, trap_coords)
        for k, v in zip(['I_tot', 'I_mean', 'u', 'sigma'], metrics(I_traps)):
            history[k].append(v)

        # 2. replace focal amplitude with target, keep phase
        focal_phase = np.angle(focal_field)
        # build new focal field: only trap sites get target amplitude;
        # non-trap sites retain their current amplitude (GS leaves them free)
        new_focal = focal_field.copy()
        for i, (px, py) in enumerate(trap_coords):
            new_focal[px, py] = A_target[i] * np.exp(1j * focal_phase[px, py])

        # 3. inverse FFT -> back to SLM plane
        slm_back = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(new_focal)))

        # 4. replace SLM amplitude with laser profile, keep phase
        slm_field = laser_amp * np.exp(1j * np.angle(slm_back))

    slm_phase = np.angle(slm_field)
    return slm_phase, history


# ── run and plot ──────────────────────────────────────────────────────────────

def run_and_plot():
    laser_amp   = gaussian_laser(Mx, My, LASER_W)
    trap_coords = make_trap_mask(Mx, My, N_side=10)
    N           = len(trap_coords)

    # equal target intensities (uniform array)
    I_target = 1.0 / N

    slm_phase, history = gerchberg_saxton(
        laser_amp, trap_coords, I_target, N_ITER
    )

    # ── final focal field ────────────────────────────────────────────────────
    slm_field   = laser_amp * np.exp(1j * slm_phase)
    focal_field = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(slm_field)))
    focal_int   = np.abs(focal_field)**2
    I_final     = trap_intensities(focal_field, trap_coords)
    I_tot, I_mean, u, sigma = metrics(I_final)

    print(f"After {N_ITER} iterations:")
    print(f"  I_tot  = {I_tot:.4f}")
    print(f"  <I>    = {I_mean:.6f}")
    print(f"  u      = {u:.4f}   (1 = perfect)")
    print(f"  sigma  = {sigma:.2f} %")

    iters = np.arange(1, N_ITER + 1)

    # ── Figure 1: convergence curves ─────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(9, 6))
    fig.suptitle("GS Algorithm: Convergence (Ex 11.2.11)", fontsize=13)

    labels = [
        ('I_tot',  r'$I_\mathrm{tot}$',           'Total intensity'),
        ('I_mean', r'$\langle I \rangle$',          'Mean trap intensity'),
        ('u',      r'Uniformity $u$',               'Uniformity'),
        ('sigma',  r'$\sigma$ (%)',                  '% Std error'),
    ]
    for ax, (key, ylabel, title) in zip(axes.flat, labels):
        ax.plot(iters, history[key], color='steelblue', lw=1.8)
        ax.set_xlabel("Iteration", fontsize=10)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_title(title, fontsize=10)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(FIGURES / "fig1_gs_convergence.png", dpi=150)
    plt.close(fig)

    # ── Figure 2: SLM phase hologram ─────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(5, 5))
    im = ax.imshow(slm_phase, cmap='hsv', vmin=-np.pi, vmax=np.pi,
                   origin='lower')
    plt.colorbar(im, ax=ax, label='Phase (rad)')
    ax.set_title(f"SLM hologram phase — {N_ITER} GS iterations", fontsize=11)
    ax.set_xlabel("SLM pixel x")
    ax.set_ylabel("SLM pixel y")
    plt.tight_layout()
    fig.savefig(FIGURES / "fig2_slm_hologram.png", dpi=150)
    plt.close(fig)

    # ── Figure 3: focal-plane intensity + trap overlay ───────────────────────
    fig, ax = plt.subplots(figsize=(5, 5))
    # log scale to see weak background
    log_int = np.log1p(focal_int / focal_int.max() * 1e4)
    ax.imshow(log_int.T, cmap='inferno', origin='lower')
    ax.scatter(trap_coords[:, 0], trap_coords[:, 1],
               s=10, c='cyan', marker='x', linewidths=0.8, label='Trap sites')
    ax.set_title("Focal-plane intensity (log scale)", fontsize=11)
    ax.set_xlabel("Focal pixel x")
    ax.set_ylabel("Focal pixel y")
    ax.legend(fontsize=8, loc='upper right')
    plt.tight_layout()
    fig.savefig(FIGURES / "fig3_focal_intensity.png", dpi=150)
    plt.close(fig)

    # ── Figure 4: per-trap intensity histogram ───────────────────────────────
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(np.arange(N), I_final / I_final.mean(), color='steelblue',
           alpha=0.8, width=1.0)
    ax.axhline(1.0, color='crimson', lw=1.5, ls='--', label='Uniform target')
    ax.set_xlabel("Trap index", fontsize=11)
    ax.set_ylabel(r"$I_n / \langle I \rangle$", fontsize=11)
    ax.set_title(
        f"Per-trap intensity after {N_ITER} GS iterations\n"
        f"u = {u:.3f},  σ = {sigma:.1f} %",
        fontsize=11
    )
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    fig.savefig(FIGURES / "fig4_trap_uniformity.png", dpi=150)
    plt.close(fig)

    print("\nFigures saved to", FIGURES)


if __name__ == "__main__":
    run_and_plot()
