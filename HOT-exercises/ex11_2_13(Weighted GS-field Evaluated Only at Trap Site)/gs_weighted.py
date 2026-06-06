
"""
Exercise 11.2.13
Weighted Gerchberg-Saxton algorithm: improve trap uniformity by
adaptively adjusting the target intensity at each iteration.

Reference: Jones, Maragò & Volpe, Optical Tweezers: Principles and Applications, Ch. 7
Notes: SSPC 2026, Sneha Chakrabarti

Principle
---------
Standard GS aims for equal target amplitude A_target at every trap but cannot
enforce it — it is only a phase-retrieval algorithm and cannot simultaneously
satisfy constraints in two planes. The result is unequal trap intensities.

Weighted GS (WGS) corrects for this by asking each trap for *more* than it
currently gets. At iteration k, the target amplitude for trap n is scaled by

    A_target^(k)(n) = A_target^(k-1)(n) * ( I_desired / I_n^(k-1) )^alpha

where I_n^(k-1) is the measured intensity at trap n after the previous
forward pass and alpha in (0,1] controls the correction strength.
- alpha = 0  ->  no update  (reduces to standard GS)
- alpha = 1  ->  full proportional correction
- alpha = 0.5 is a common empirical choice (less oscillation)

Equivalently, writing w_n^(k) = I_desired / I_n^(k-1), the corrected target
amplitude is A_0 * prod_{j<k} w_n^(j)^alpha, i.e. traps that have been
consistently underperforming accumulate a growing weight.

This does not change the algorithm structure: only the amplitude substituted
in step 2 of each GS iteration is modified. All other steps are identical.
"""

import numpy as np
import matplotlib.pyplot as plt
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
# WGS runs the same benchmark as GS for a direct comparison.
Mx, My  = 1000, 1000
N_ITER  = 80
LASER_W = 400
MARGIN  = 0.15
FIGURES = Path(__file__).parent / "figures"
FIGURES.mkdir(exist_ok=True)


# ── weighted GS ───────────────────────────────────────────────────────────────

def weighted_gs(laser_amp, trap_coords, n_iter, alpha=0.5, seed_phase=None):
    """
    Weighted Gerchberg-Saxton algorithm.

    Parameters
    ----------
    laser_amp   : (Mx, My) SLM amplitude constraint
    trap_coords : (N, 2) trap pixel coords in focal plane
    n_iter      : iterations
    alpha       : weight update exponent in (0, 1]
    seed_phase  : (Mx, My) initial SLM phase; random if None

    Returns
    -------
    slm_phase : (Mx, My) final hologram phase
    history   : dict of per-iteration metric arrays + 'weights'
    """
    Mx, My = laser_amp.shape
    N      = len(trap_coords)

    if seed_phase is None:
        seed_phase = rng.uniform(0, 2 * np.pi, (Mx, My))

    slm_field = laser_amp * np.exp(1j * seed_phase)

    # initial target: uniform, normalised so sum(A^2) = 1
    A_target = np.full(N, 1.0 / np.sqrt(N))

    history = {k: [] for k in ['I_tot', 'I_mean', 'u', 'sigma', 'weights']}

    for _ in range(n_iter):
        # 1. forward FFT
        focal_field = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(slm_field)))

        # measure trap intensities
        I_traps = trap_intensities(focal_field, trap_coords)

        # record metrics
        for k, v in zip(['I_tot', 'I_mean', 'u', 'sigma'], metrics(I_traps)):
            history[k].append(v)
        history['weights'].append(A_target.copy())

        # 2. weight update: traps that are too weak get boosted target
        I_mean = I_traps.mean()
        # avoid division by zero at the first iteration
        safe_I = np.where(I_traps > 0, I_traps, 1e-30)
        correction = (I_mean / safe_I) ** alpha
        A_target   = A_target * correction
        # renormalise so total power constraint is respected
        A_target  /= np.linalg.norm(A_target) / np.sqrt(N / np.sum(A_target**2))
        # simpler normalisation: keep geometric mean fixed
        A_target  *= (1.0 / np.sqrt(N)) / np.exp(np.mean(np.log(A_target)))

        # replace focal amplitude with updated target, keep phase
        focal_phase = np.angle(focal_field)
        new_focal = focal_field.copy()
        for i, (px, py) in enumerate(trap_coords):
            new_focal[px, py] = A_target[i] * np.exp(1j * focal_phase[px, py])

        # 3. inverse FFT
        slm_back = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(new_focal)))

        # 4. SLM amplitude constraint
        slm_field = laser_amp * np.exp(1j * np.angle(slm_back))

    return np.angle(slm_field), history


# ── run and plot ──────────────────────────────────────────────────────────────

def run_and_plot():
    laser_amp   = gaussian_laser(Mx, My, LASER_W)
    trap_coords = make_trap_mask(Mx, My, N_side=10, margin=MARGIN)
    N           = len(trap_coords)
    I_target    = 1.0 / N

    # standard GS for comparison
    print("Standard GS...")
    _, hist_std = gerchberg_saxton(laser_amp, trap_coords, I_target, N_ITER)

    # WGS for several alpha values
    alphas   = [0.2, 0.5, 1.0]
    colors   = ['tomato', 'darkorange', 'seagreen']
    wgs_hists = {}
    wgs_phases = {}

    for a in alphas:
        print(f"Weighted GS (alpha={a})...")
        ph, h = weighted_gs(laser_amp, trap_coords, N_ITER, alpha=a)
        wgs_hists[a]  = h
        wgs_phases[a] = ph

    iters = np.arange(1, N_ITER + 1)

    # ── Figure 1: uniformity comparison ──────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    fig.suptitle("Weighted GS: Uniformity Improvement (Ex 11.2.13)", fontsize=13)

    for ax, key, ylabel in [
        (axes[0], 'u',     r'Uniformity $u$'),
        (axes[1], 'sigma', r'$\sigma$ (%)'),
    ]:
        ax.plot(iters, hist_std[key], lw=1.8, ls='--',
                color='steelblue', label='Standard GS')
        for a, col in zip(alphas, colors):
            ax.plot(iters, wgs_hists[a][key], lw=1.8, color=col,
                    label=rf'WGS $\alpha$={a}')
        ax.set_xlabel("Iteration", fontsize=10)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(FIGURES / "fig1_wgs_uniformity.png", dpi=150)
    plt.close(fig)

    # ── Figure 2: per-trap intensity bars — standard vs best WGS ─────────────
    best_alpha = alphas[np.argmax([wgs_hists[a]['u'][-1] for a in alphas])]

    def final_trap_intensities(slm_phase):
        slm_field   = laser_amp * np.exp(1j * slm_phase)
        focal_field = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(slm_field)))
        return trap_intensities(focal_field, trap_coords)

    # standard GS final hologram
    slm_std, _ = gerchberg_saxton(laser_amp, trap_coords, I_target, N_ITER)
    I_std       = final_trap_intensities(slm_std)
    I_wgs       = final_trap_intensities(wgs_phases[best_alpha])

    u_std,  sig_std  = metrics(I_std)[2],  metrics(I_std)[3]
    u_wgs,  sig_wgs  = metrics(I_wgs)[2],  metrics(I_wgs)[3]

    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    for ax, I, label, u, sig, col in [
        (axes[0], I_std, 'Standard GS',           u_std, sig_std, 'steelblue'),
        (axes[1], I_wgs, f'WGS α={best_alpha}', u_wgs, sig_wgs, 'seagreen'),
    ]:
        ax.bar(np.arange(N), I / I.mean(), color=col, alpha=0.8, width=1.0)
        ax.axhline(1.0, color='crimson', lw=1.2, ls='--')
        ax.set_ylabel(r"$I_n / \langle I \rangle$", fontsize=10)
        ax.set_title(f"{label}  |  u = {u:.3f},  σ = {sig:.1f} %", fontsize=10)
        ax.set_ylim(0, max(2.0, (I / I.mean()).max() * 1.1))
        ax.grid(True, alpha=0.3, axis='y')

    axes[1].set_xlabel("Trap index", fontsize=11)
    fig.suptitle(f"Per-trap intensity: Standard GS vs Weighted GS\n({N_ITER} iterations)",
                 fontsize=12)
    plt.tight_layout()
    fig.savefig(FIGURES / "fig2_per_trap_comparison.png", dpi=150)
    plt.close(fig)

    # ── Figure 3: evolution of weight distribution ────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    fig.suptitle(r"Weight $A_\mathrm{target}^{(k)}(n)$ evolution (WGS)", fontsize=12)
    for ax, a, col in zip(axes, alphas, colors):
        weights = np.array(wgs_hists[a]['weights'])   # (n_iter, N)
        for n in range(N):
            ax.plot(iters, weights[:, n], lw=0.5, alpha=0.4, color=col)
        ax.plot(iters, weights.mean(axis=1), lw=2, color='k', label='mean')
        ax.set_xlabel("Iteration")
        ax.set_ylabel(r"$A_\mathrm{target}$")
        ax.set_title(rf"$\alpha = {a}$")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(FIGURES / "fig3_weight_evolution.png", dpi=150)
    plt.close(fig)

    print("\nFigures saved to", FIGURES)


if __name__ == "__main__":
    run_and_plot()
