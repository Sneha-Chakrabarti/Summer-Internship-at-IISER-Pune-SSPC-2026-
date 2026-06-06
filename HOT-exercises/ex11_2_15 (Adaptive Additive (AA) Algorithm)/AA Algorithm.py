"""
Exercise 11.2.15
Implement the Adaptive-Additive (AA) algorithm and study its performance.

Reference: Jones, Maragò & Volpe, Optical Tweezers: Principles and Applications, Ch. 7
Notes: SSPC 2026, Sneha Chakrabarti

Standard benchmark task (Section 5.1 of theory notes)
------------------------------------------------------
SLM   : 1000 × 1000 pixels
Traps : N = 100 on a 10 × 10 square lattice, z = 0 (focal plane)
Laser : Gaussian, full-width filling the SLM aperture

Performance metrics tracked per iteration:
  I_tot  = Σ I_n                               (total intensity)
  <I>    = I_tot / N                            (mean trap intensity)
  u      = 1 − (max I_n − min I_n)/(max I_n + min I_n)   (uniformity, → 1)
  σ      = 100 · std(I_n) / <I>                (% std error, → 0)

AA algorithm (Section 5.2)
--------------------------
Extends GS with a mixing parameter a ∈ [0, 1].
Instead of hard-replacing the focal amplitude at each iteration (GS),
AA mixes a fraction a of the target amplitude into the current focal field:

    E_new(x, y) = (1 − a) · E_current(x, y)  +  a · A_target(x, y) · exp(i φ_current(x, y))

At trap sites n:
    A_target(x_n) = √I_target   (uniform target)
    φ_current     = arg(E_focal) at that site

At non-trap sites:
    E_new = E_current  (field left unchanged; only trap sites are updated)

Limits:
  a = 1  →  reduces exactly to standard GS (hard replacement)
  a → 0  →  field barely updated per iteration; slow but smooth convergence

The mixing prevents stagnation in poor local optima and typically yields
better uniformity than GS at the same iteration count.

Study
-----
1. Vary a ∈ {0.1, 0.25, 0.5, 0.75, 1.0} and compare convergence curves.
2. Compare final-state metrics of AA (best a) vs GS (a=1) on the same seed.
3. Show hologram phase and focal-plane intensity for the best AA result.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "ex11_2_11"))
from gs_algorithm import gaussian_laser, make_trap_mask, trap_intensities, metrics

rng = np.random.default_rng(42)

# ── standard benchmark parameters ────────────────────────────────────────────
Mx, My   = 1000, 1000
N_SIDE   = 10          # 10×10 lattice → N = 100 traps
N_ITER   = 60
LASER_W  = 400         # Gaussian 1/e² radius in pixels (fills aperture well)
MARGIN   = 0.15

FIGURES = Path(__file__).parent / "figures"
FIGURES.mkdir(exist_ok=True)


# ── adaptive-additive algorithm ───────────────────────────────────────────────

def adaptive_additive(laser_amp, trap_coords, I_target_per_trap, n_iter,
                      a=0.5, seed_phase=None):
    """
    Adaptive-Additive (AA) algorithm for hologram design.

    Parameters
    ----------
    laser_amp          : (Mx, My) SLM amplitude constraint (laser profile)
    trap_coords        : (N, 2) int array of trap pixel indices in focal plane
    I_target_per_trap  : scalar or (N,) target intensity at each trap
    n_iter             : number of iterations
    a                  : mixing parameter in (0, 1]; a=1 → standard GS
    seed_phase         : (Mx, My) initial SLM phase; random if None

    Returns
    -------
    slm_phase : (Mx, My) final hologram phase
    history   : dict with per-iteration metric lists
    """
    Mx, My = laser_amp.shape
    N      = len(trap_coords)

    if np.isscalar(I_target_per_trap):
        A_target = np.full(N, np.sqrt(float(I_target_per_trap)))
    else:
        A_target = np.sqrt(np.asarray(I_target_per_trap, dtype=float))

    if seed_phase is None:
        seed_phase = rng.uniform(0, 2 * np.pi, (Mx, My))

    slm_field = laser_amp * np.exp(1j * seed_phase)
    history   = {k: [] for k in ['I_tot', 'I_mean', 'u', 'sigma']}

    for _ in range(n_iter):
        # 1. forward FFT → focal field
        focal_field = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(slm_field)))

        # record metrics before update
        I_traps = trap_intensities(focal_field, trap_coords)
        for k, v in zip(['I_tot', 'I_mean', 'u', 'sigma'], metrics(I_traps)):
            history[k].append(v)

        # 2. AA mixing: update only at trap sites
        #    E_new[n] = (1-a)*E_current[n] + a*A_target[n]*exp(i*phi_current[n])
        new_focal = focal_field.copy()
        phi_focal = np.angle(focal_field)
        for i, (px, py) in enumerate(trap_coords):
            E_target_n        = A_target[i] * np.exp(1j * phi_focal[px, py])
            new_focal[px, py] = (1 - a) * focal_field[px, py] + a * E_target_n

        # 3. inverse FFT → back to SLM plane
        slm_back = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(new_focal)))

        # 4. apply laser amplitude constraint, keep phase
        slm_field = laser_amp * np.exp(1j * np.angle(slm_back))

    return np.angle(slm_field), history


# ── run and plot ──────────────────────────────────────────────────────────────

def run_and_plot():
    print(f"Setting up {Mx}×{My} SLM, N=100 traps...")
    laser_amp   = gaussian_laser(Mx, My, LASER_W)
    trap_coords = make_trap_mask(Mx, My, N_side=N_SIDE, margin=MARGIN)
    N           = len(trap_coords)
    I_target    = 1.0 / N

    # shared seed so GS and AA start identically
    seed = rng.uniform(0, 2 * np.pi, (Mx, My))

    # ── run AA for several values of a ───────────────────────────────────────
    a_values = [0.1, 0.25, 0.5, 0.75, 1.0]
    colors   = ['#1a6faf', '#2ca02c', '#ff7f0e', '#9467bd', '#d62728']
    results  = {}

    for a, col in zip(a_values, colors):
        label = f'AA  a={a}' if a < 1.0 else 'GS  (a=1)'
        print(f"  Running {label}...")
        ph, hist = adaptive_additive(
            laser_amp, trap_coords, I_target, N_ITER, a=a,
            seed_phase=seed.copy()
        )
        results[a] = {'phase': ph, 'history': hist, 'color': col, 'label': label}

    iters = np.arange(1, N_ITER + 1)

    # ── Figure 1: convergence — uniformity u and σ ────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    fig.suptitle(
        "AA Algorithm: Convergence vs Mixing Parameter $a$\n"
        f"(1000×1000 SLM, N=100 traps, z=0)",
        fontsize=12, fontweight='bold'
    )

    for ax, key, ylabel, ylim in [
        (axes[0], 'u',     r'Uniformity $u$',  (0, 1.05)),
        (axes[1], 'sigma', r'$\sigma$ (%)',      None),
    ]:
        for a in a_values:
            r = results[a]
            ax.plot(iters, r['history'][key], lw=1.8,
                    color=r['color'], label=r['label'])
        ax.set_xlabel("Iteration", fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        if ylim:
            ax.set_ylim(*ylim)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(FIGURES / "fig1_aa_convergence.png", dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("  fig1 saved.")

    # ── Figure 2: I_tot and <I> convergence ───────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    fig.suptitle(
        "AA Algorithm: Efficiency Metrics vs $a$\n"
        f"(1000×1000 SLM, N=100 traps, z=0)",
        fontsize=12, fontweight='bold'
    )

    for ax, key, ylabel in [
        (axes[0], 'I_tot',  r'$I_\mathrm{tot} = \sum I_n$'),
        (axes[1], 'I_mean', r'$\langle I \rangle = I_\mathrm{tot}/N$'),
    ]:
        for a in a_values:
            r = results[a]
            ax.plot(iters, r['history'][key], lw=1.8,
                    color=r['color'], label=r['label'])
        ax.set_xlabel("Iteration", fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(FIGURES / "fig2_aa_efficiency.png", dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("  fig2 saved.")

    # ── Figure 3: per-trap intensity bars — GS vs best AA ────────────────────
    # pick best AA = highest final u among a < 1
    best_a = max((a for a in a_values if a < 1.0),
                 key=lambda a: results[a]['history']['u'][-1])

    def final_trap_I(slm_phase):
        sf = laser_amp * np.exp(1j * slm_phase)
        ff = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(sf)))
        return trap_intensities(ff, trap_coords)

    I_gs  = final_trap_I(results[1.0]['phase'])
    I_aa  = final_trap_I(results[best_a]['phase'])
    u_gs, sig_gs = metrics(I_gs)[2], metrics(I_gs)[3]
    u_aa, sig_aa = metrics(I_aa)[2], metrics(I_aa)[3]

    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    fig.suptitle(
        "Per-Trap Intensity: GS vs Best AA\n"
        f"(1000×1000 SLM, N=100 traps, {N_ITER} iterations)",
        fontsize=12, fontweight='bold'
    )
    for ax, I, label, u, sig, col in [
        (axes[0], I_gs, 'GS  (a=1.0)',         u_gs, sig_gs, '#d62728'),
        (axes[1], I_aa, f'AA  (a={best_a})',   u_aa, sig_aa, '#ff7f0e'),
    ]:
        ax.bar(np.arange(N), I / I.mean(), color=col, alpha=0.75, width=1.0)
        ax.axhline(1.0, color='k', lw=1.2, ls='--', label='Uniform')
        ax.set_ylabel(r"$I_n\,/\,\langle I\rangle$", fontsize=10)
        ax.set_title(
            f"{label}  —  uniformity $u$ = {u:.4f},  $\\sigma$ = {sig:.2f} %",
            fontsize=10
        )
        ax.set_ylim(0, max(2.2, (I / I.mean()).max() * 1.15))
        ax.legend(fontsize=8, loc='upper right')
        ax.grid(True, alpha=0.25, axis='y')

    axes[1].set_xlabel("Trap index  (0 – 99)", fontsize=11)
    plt.tight_layout()
    fig.savefig(FIGURES / "fig3_per_trap_gs_vs_aa.png", dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("  fig3 saved.")

    # ── Figure 4: final-state metrics bar chart across a values ──────────────
    final_u     = [results[a]['history']['u'][-1]     for a in a_values]
    final_sigma = [results[a]['history']['sigma'][-1] for a in a_values]
    final_Itot  = [results[a]['history']['I_tot'][-1] for a in a_values]
    x = np.arange(len(a_values))
    xlabels = [r['label'] for r in results.values()]

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
    fig.suptitle(
        f"Final-State Performance vs $a$  ({N_ITER} iterations)\n"
        "1000×1000 SLM, N=100 traps",
        fontsize=12, fontweight='bold'
    )
    for ax, vals, ylabel, col in [
        (axes[0], final_u,     r'Uniformity $u$',           [r['color'] for r in results.values()]),
        (axes[1], final_sigma, r'$\sigma$ (%)',              [r['color'] for r in results.values()]),
        (axes[2], final_Itot,  r'$I_\mathrm{tot}$',         [r['color'] for r in results.values()]),
    ]:
        bars = ax.bar(x, vals, color=col, alpha=0.85, width=0.6)
        ax.set_xticks(x)
        ax.set_xticklabels(xlabels, fontsize=8, rotation=15)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.grid(True, alpha=0.3, axis='y')
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() * 1.01, f'{v:.3f}',
                    ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    fig.savefig(FIGURES / "fig4_final_metrics_vs_a.png", dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("  fig4 saved.")

    # ── Figure 5: SLM hologram (best AA) ─────────────────────────────────────
    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(results[best_a]['phase'], cmap='hsv',
                   vmin=-np.pi, vmax=np.pi, origin='lower')
    plt.colorbar(im, ax=ax, label='Phase (rad)', fraction=0.046, pad=0.04)
    ax.set_title(
        f"SLM Hologram Phase — AA  a={best_a}\n"
        f"({N_ITER} iterations, 1000×1000)",
        fontsize=11
    )
    ax.set_xlabel("SLM pixel x"); ax.set_ylabel("SLM pixel y")
    plt.tight_layout()
    fig.savefig(FIGURES / "fig5_slm_hologram_aa.png", dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("  fig5 saved.")

    # ── Figure 6: focal-plane intensity (best AA) ─────────────────────────────
    sf_best     = laser_amp * np.exp(1j * results[best_a]['phase'])
    focal_best  = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(sf_best)))
    focal_int   = np.abs(focal_best)**2
    log_int     = np.log1p(focal_int / focal_int.max() * 1e4)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(log_int.T, cmap='inferno', origin='lower')
    ax.scatter(trap_coords[:, 0], trap_coords[:, 1],
               s=6, c='cyan', marker='x', linewidths=0.7, label='Trap sites')
    ax.set_title(
        f"Focal-Plane Intensity (log scale) — AA  a={best_a}\n"
        f"({N_ITER} iterations, 1000×1000, z=0)",
        fontsize=11
    )
    ax.set_xlabel("Focal pixel x"); ax.set_ylabel("Focal pixel y")
    ax.legend(fontsize=8, loc='upper right')
    plt.tight_layout()
    fig.savefig(FIGURES / "fig6_focal_intensity_aa.png", dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("  fig6 saved.")

    # ── print summary table ───────────────────────────────────────────────────
    print(f"\n{'':─<58}")
    print(f"{'Algorithm':<18} {'I_tot':>10} {'<I>':>10} {'u':>8} {'σ (%)':>8}")
    print(f"{'':─<58}")
    for a in a_values:
        h  = results[a]['history']
        It = h['I_tot'][-1]; Im = h['I_mean'][-1]
        u  = h['u'][-1];     sg = h['sigma'][-1]
        print(f"{results[a]['label']:<18} {It:>10.4f} {Im:>10.6f} {u:>8.4f} {sg:>8.2f}")
    print(f"{'':─<58}")
    print(f"\nBest AA mixing parameter: a = {best_a}")
    print(f"Figures saved to {FIGURES}")


if __name__ == "__main__":
    run_and_plot()
