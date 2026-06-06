"""
Exercise 11.4.3
Generate continuous optical potentials using the Adaptive-Additive (AA)
algorithm, both on a single focal plane and on multiple axial planes.

Reference: Jones, Maragò & Volpe, Optical Tweezers: Principles and Applications, Ch. 7
Notes: SSPC 2026, Sneha Chakrabarti

AA algorithm for continuous potentials
----------------------------------------
Identical extension from point-trap AA (Ex 11.2.15) to continuous potentials
as was done for GS in Ex 11.4.1: the amplitude replacement at the focal plane
is applied over the entire field (not just N discrete sites), and the mixing
parameter a controls how aggressively the target is imposed.

Single-plane AA per iteration:
  1. Forward FFT -> focal field E_focus^(k)
  2. AA mixing over ALL focal pixels:
       E_new(x,y) = (1-a) * E_focus^(k)(x,y)
                   + a * A_target(x,y) * exp(i phi_focus^(k)(x,y))
     This is the continuous analogue of Eq. 10 in the theory notes.
  3. Inverse FFT -> SLM plane
  4. Laser amplitude constraint

Limit a=1 reduces exactly to continuous GS (Ex 11.4.1).
Small a gives smoother phase updates; prevents snapping into poor local optima.

Multi-plane AA:
  Same as multi-plane GS (Section 7.2) but step 2 uses AA mixing at each plane:
    E_p_new = (1-a) * E_p^(k) + a * A_target_p * exp(i phi_p^(k))
  Back-propagated contributions from all planes are summed as before.

Study
-----
1. Compare AA vs GS on the same continuous targets at the same iteration count.
2. Vary a and compare convergence of C.
3. Multi-plane AA: same 3-plane setup as Ex 11.4.1.
4. Show that AA achieves lower C than GS at early iterations (same a=0.5).

Standard parameters
-------------------
SLM  : 1000 × 1000 (single-plane)  |  256 × 256 (multi-plane)
Laser: Gaussian, 1/e^2 radius = 400 px  |  100 px
Iterations: 80 (single)  |  60 (multi)
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "ex11_2_11"))
from gs_algorithm import gaussian_laser
sys.path.insert(0, str(Path(__file__).parent.parent / "ex11_2_14"))
from gs_3d import propagate_to_plane, back_propagate
sys.path.insert(0, str(Path(__file__).parent.parent / "ex11_4_1"))
from continuous_gs import (convergence_error, target_ring, target_cross,
                            target_square_frame, target_lemniscate,
                            continuous_gs_single)

rng = np.random.default_rng(42)

Mx, My  = 1000, 1000
LASER_W = 400
N_ITER    = 60    # single-plane iterations (reduced from 80 for speed)
N_ITER_MP = 40    # multi-plane iterations

FIGURES = Path(__file__).parent / "figures"
FIGURES.mkdir(exist_ok=True)


# ── single-plane continuous AA ────────────────────────────────────────────────

def continuous_aa_single(laser_amp, A_target, n_iter, a=0.5, seed_phase=None):
    """
    AA algorithm for continuous optical potential on a single focal plane.

    E_new(x,y) = (1-a)*E_focus(x,y) + a*A_target(x,y)*exp(i*phi_focus(x,y))

    applied over ALL focal pixels. a=1 reduces to continuous GS.

    Parameters
    ----------
    laser_amp : (Mx, My)
    A_target  : (Mx, My) target amplitude
    n_iter    : iterations
    a         : mixing parameter in (0, 1]
    seed_phase: (Mx, My) initial SLM phase

    Returns
    -------
    slm_phase : (Mx, My)
    history   : {'C': list}
    """
    Mx, My = laser_amp.shape
    power_slm    = float(np.sum(laser_amp**2)) * Mx * My
    power_target = float(np.sum(A_target**2))
    A_t = A_target * np.sqrt(power_slm / (power_target + 1e-30))

    if seed_phase is None:
        seed_phase = rng.uniform(0, 2 * np.pi, (Mx, My))
    slm_field = laser_amp * np.exp(1j * seed_phase)
    history   = {'C': []}

    for _ in range(n_iter):
        # 1. forward FFT
        focal_field = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(slm_field)))

        history['C'].append(convergence_error(focal_field, A_t))

        # 2. AA mixing over all pixels (fixed-scale target)
        E_target  = A_t * np.exp(1j * np.angle(focal_field))
        new_focal = (1 - a) * focal_field + a * E_target

        # 3. inverse FFT
        slm_back = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(new_focal)))

        # 4. laser amplitude constraint
        slm_field = laser_amp * np.exp(1j * np.angle(slm_back))

    return np.angle(slm_field), history


# ── multi-plane continuous AA ─────────────────────────────────────────────────

def continuous_aa_multi(laser_amp, plane_targets, n_iter, a=0.5,
                         seed_phase=None):
    """
    AA for continuous optical potential on multiple axial planes.

    At each plane: E_p_new = (1-a)*E_p + a*A_target_p*exp(i*phi_p)
    Back-propagated contributions are summed (same structure as multi-plane GS).

    Parameters
    ----------
    laser_amp    : (Mx, My)
    plane_targets: list of (z_norm, A_target_p)
    n_iter       : iterations
    a            : AA mixing parameter
    seed_phase   : initial SLM phase

    Returns
    -------
    slm_phase : (Mx, My)
    history   : {p: list of C values per plane}
    """
    Mx, My = laser_amp.shape
    P = len(plane_targets)
    power_slm = float(np.sum(laser_amp**2)) * Mx * My
    normalised_targets = []
    for z_norm, A_t in plane_targets:
        pt = float(np.sum(A_t**2))
        normalised_targets.append(
            (z_norm, A_t * np.sqrt(power_slm / (pt + 1e-30)))
        )

    if seed_phase is None:
        seed_phase = rng.uniform(0, 2 * np.pi, (Mx, My))
    slm_field = laser_amp * np.exp(1j * seed_phase)
    history = {p: [] for p in range(P)}

    for _ in range(n_iter):
        slm_back_sum = np.zeros((Mx, My), dtype=complex)

        for p, (z_norm, A_t) in enumerate(normalised_targets):
            E_p = propagate_to_plane(slm_field, z_norm)
            history[p].append(convergence_error(E_p, A_t))

            E_target = A_t * np.exp(1j * np.angle(E_p))
            E_p_new  = (1 - a) * E_p + a * E_target

            slm_back_sum += back_propagate(E_p_new, z_norm)

        slm_field = laser_amp * np.exp(1j * np.angle(slm_back_sum))

    return np.angle(slm_field), history


# ── run and plot ──────────────────────────────────────────────────────────────

def run_and_plot():
    print(f"Setting up {Mx}×{My} SLM...")
    laser_amp = gaussian_laser(Mx, My, LASER_W)
    seed      = rng.uniform(0, 2 * np.pi, (Mx, My))

    # ── Study 1: AA vs GS — convergence comparison on ring target ─────────────
    A_ring = target_ring(Mx, My, r_inner=60, r_outer=120)
    a_values  = [0.2, 0.5, 0.75, 1.0]
    colors_a  = ['#1a6faf', '#2ca02c', '#ff7f0e', '#d62728']
    labels_a  = [f'AA  a={a}' if a < 1 else 'GS  (a=1)' for a in a_values]

    print("  AA vs GS on ring target...")
    C_runs = {}
    for a, col, label in zip(a_values, colors_a, labels_a):
        _, hist = continuous_aa_single(laser_amp, A_ring, N_ITER,
                                        a=a, seed_phase=seed.copy())
        C_runs[label] = np.array(hist['C'])
        print(f"    {label}: C_final={hist['C'][-1]:.4e}")

    iters = np.arange(1, N_ITER + 1)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(
        "Continuous AA vs GS: Convergence on Ring Target\n"
        f"(1000×1000 SLM, {N_ITER} iterations, same random seed)",
        fontsize=12, fontweight='bold'
    )
    for (label, C_vals), col in zip(C_runs.items(), colors_a):
        axes[0].semilogy(iters, C_vals, lw=1.8, color=col, label=label)
        axes[1].semilogy(iters, C_vals / C_vals[0], lw=1.8, color=col,
                         label=label)
    for ax, ylabel in zip(axes,
                           [r"$C^{(k)}$", r"$C^{(k)}/C^{(0)}$ (normalised)"]):
        ax.set_xlabel("Iteration", fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.legend(fontsize=9); ax.grid(True, which='both', alpha=0.3)
    axes[0].set_title("Absolute error C", fontsize=11)
    axes[1].set_title("Normalised error C/C₀", fontsize=11)
    plt.tight_layout()
    fig.savefig(FIGURES / "fig1_aa_vs_gs_convergence.png", dpi=150,
                bbox_inches='tight')
    plt.close(fig)
    print("  fig1 saved.")

    # ── Study 2: AA on all four targets — best a ──────────────────────────────
    targets = {
        "Ring":        target_ring(Mx, My, r_inner=60, r_outer=120),
        "Lemniscate":  target_lemniscate(Mx, My, a=80),
        "Cross":       target_cross(Mx, My, arm_width=20, arm_length=180),
        r"$\phi$ symbol": __import__(
            'sys').modules[__name__].__dict__.get(
            '_phi_sym',
            target_ring(Mx, My, r_inner=55, r_outer=75)
        ),
    }
    # reimport phi symbol properly
    sys.path.insert(0, str(Path(__file__).parent.parent / "ex11_4_1"))
    from continuous_gs import target_phi_symbol
    targets[r"$\phi$ symbol"] = target_phi_symbol(Mx, My, r=80, thickness=6,
                                                    stem_len=130)
    t_colors = ['steelblue', 'tomato', 'seagreen', 'darkorchid']
    BEST_A   = 0.5

    print(f"  AA (a={BEST_A}) on all targets...")
    aa_results = {}
    for (name, A_t), col in zip(targets.items(), t_colors):
        phi, hist = continuous_aa_single(laser_amp, A_t, N_ITER,
                                          a=BEST_A, seed_phase=seed.copy())
        sf  = laser_amp * np.exp(1j * phi)
        ff  = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(sf)))
        aa_results[name] = {'phase': phi, 'focal': ff,
                            'target': A_t, 'C': np.array(hist['C'])}
        print(f"    {name}: C_final={hist['C'][-1]:.4e}")

    # GS results for comparison — run on two targets only for speed
    print("  GS on ring and lemniscate targets (comparison)...")
    gs_C = {}
    for name in ["Ring", "Lemniscate"]:
        A_t = targets[name]
        _, hist_gs = continuous_gs_single(laser_amp, A_t, N_ITER,
                                           seed_phase=seed.copy())
        gs_C[name] = np.array(hist_gs['C'])
    # fill other targets with None for plotting logic
    for name in targets:
        if name not in gs_C:
            gs_C[name] = None

    # ── Figure 2: AA target vs achieved ───────────────────────────────────────
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    fig.suptitle(
        f"Continuous AA (a={BEST_A}): Target vs Achieved Optical Potentials\n"
        f"(1000×1000 SLM, {N_ITER} iterations)",
        fontsize=12, fontweight='bold'
    )
    crop = 250
    cx, cy = Mx // 2, My // 2
    sl = np.s_[cx-crop:cx+crop, cy-crop:cy+crop]

    for col, (name, res) in enumerate(aa_results.items()):
        axes[0, col].imshow(res['target'][sl].T, cmap='inferno', origin='lower')
        axes[0, col].set_title(f"Target: {name}", fontsize=9)

        focal_int = np.abs(res['focal'])**2
        axes[1, col].imshow((focal_int[sl] / focal_int[sl].max()).T,
                             cmap='inferno', origin='lower', vmin=0, vmax=1)
        axes[1, col].set_title("Achieved intensity", fontsize=9)

    for ax in axes.flat:
        ax.set_xlabel("px"); ax.set_ylabel("px")
    axes[0, 0].set_ylabel("Target amplitude\npx", fontsize=9)
    axes[1, 0].set_ylabel("Focal-plane intensity\npx", fontsize=9)
    plt.tight_layout()
    fig.savefig(FIGURES / "fig2_aa_target_vs_achieved.png", dpi=150,
                bbox_inches='tight')
    plt.close(fig)
    print("  fig2 saved.")

    # ── Figure 3: AA vs GS per target ─────────────────────────────────────────
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    fig.suptitle(
        rf"AA ($a={BEST_A}$) vs GS ($a=1$): $C/C_0$ per Target"
        "\nAA converges faster in early iterations",
        fontsize=12, fontweight='bold'
    )
    for ax, (name, res), col in zip(axes, aa_results.items(), t_colors):
        C_aa = res['C']
        ax.semilogy(iters, C_aa / C_aa[0], lw=1.8, color=col,
                    label=f'AA  a={BEST_A}')
        if gs_C.get(name) is not None:
            C_gs = gs_C[name]
            ax.semilogy(iters, C_gs / C_gs[0], lw=1.8, color=col,
                        ls='--', alpha=0.6, label='GS  (a=1)')
        ax.set_title(name, fontsize=10)
        ax.set_xlabel("Iteration", fontsize=9)
        ax.set_ylabel(r"$C/C_0$", fontsize=9)
        ax.legend(fontsize=8); ax.grid(True, which='both', alpha=0.3)
    plt.tight_layout()
    fig.savefig(FIGURES / "fig3_aa_vs_gs_per_target.png", dpi=150,
                bbox_inches='tight')
    plt.close(fig)
    print("  fig3 saved.")

    # ── multi-plane AA ────────────────────────────────────────────────────────
    Mx_mp, My_mp = 256, 256
    W_mp         = 100
    laser_mp     = gaussian_laser(Mx_mp, My_mp, W_mp)
    seed_mp      = rng.uniform(0, 2 * np.pi, (Mx_mp, My_mp))

    plane_configs = [
        (-0.06, target_ring(Mx_mp, My_mp, r_inner=20, r_outer=40)),
        ( 0.00, target_cross(Mx_mp, My_mp, arm_width=6, arm_length=60)),
        ( 0.06, target_square_frame(Mx_mp, My_mp, side=80, thickness=8)),
    ]
    plane_names  = ["Ring  (z=-0.06)", "Cross  (z=0)", "Square frame  (z=+0.06)"]
    plane_colors = ['steelblue', 'tomato', 'seagreen']

    a_mp_vals  = [0.5, 1.0]
    a_mp_labs  = [f'AA a=0.5', 'GS (a=1)']
    a_mp_cols  = ['darkorchid', 'grey']
    a_mp_ls    = ['-', '--']

    print(f"  Multi-plane AA vs GS (256×256, {N_ITER_MP} iters)...")
    mp_results = {}
    for a_mp, lab in zip(a_mp_vals, a_mp_labs):
        phi, hist = continuous_aa_multi(laser_mp, plane_configs,
                                         N_ITER_MP, a=a_mp,
                                         seed_phase=seed_mp.copy())
        mp_results[lab] = {'phase': phi, 'history': hist}
        print(f"    {lab}: done")

    # ── Figure 4: multi-plane AA vs GS convergence ────────────────────────────
    iters_mp = np.arange(1, N_ITER_MP + 1)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    fig.suptitle(
        "Multi-Plane Continuous AA vs GS: Convergence per Plane\n"
        r"(256×256 SLM, $z_\mathrm{norm}\in\{-0.06,\,0,\,+0.06\}$, "
        f"{N_ITER_MP} iterations)",
        fontsize=12, fontweight='bold'
    )
    for ax, p, name, pcol in zip(axes, range(3), plane_names, plane_colors):
        for lab, acol, als in zip(a_mp_labs, a_mp_cols, a_mp_ls):
            C_p = np.array(mp_results[lab]['history'][p])
            ax.semilogy(iters_mp, C_p / C_p[0], lw=1.8, color=acol,
                        ls=als, label=lab)
        ax.set_title(name, fontsize=10)
        ax.set_xlabel("Iteration", fontsize=10)
        ax.set_ylabel(r"$C_p / C_{p,0}$", fontsize=10)
        ax.legend(fontsize=8); ax.grid(True, which='both', alpha=0.3)
    plt.tight_layout()
    fig.savefig(FIGURES / "fig4_mp_aa_vs_gs.png", dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("  fig4 saved.")

    # ── Figure 5: multi-plane achieved intensity (AA a=0.5) ───────────────────
    phi_mp_aa = mp_results['AA a=0.5']['phase']
    slm_mp    = laser_mp * np.exp(1j * phi_mp_aa)
    fig, axes = plt.subplots(2, 3, figsize=(13, 8))
    fig.suptitle(
        f"Multi-Plane Continuous AA (a=0.5): Target vs Achieved\n"
        "(256×256 SLM)",
        fontsize=12, fontweight='bold'
    )
    for col, (z_norm, A_t), name in zip(range(3), plane_configs, plane_names):
        E_p      = propagate_to_plane(slm_mp, z_norm)
        intens_p = np.abs(E_p)**2
        axes[0, col].imshow(A_t.T, cmap='inferno', origin='lower')
        axes[0, col].set_title(f"Target\n{name}", fontsize=9)
        axes[1, col].imshow((intens_p / intens_p.max()).T,
                             cmap='inferno', origin='lower', vmin=0, vmax=1)
        axes[1, col].set_title("Achieved intensity", fontsize=9)
        for ax in [axes[0, col], axes[1, col]]:
            ax.set_xlabel("px"); ax.set_ylabel("px")
    axes[0, 0].set_ylabel("Target amplitude\npx", fontsize=9)
    axes[1, 0].set_ylabel("Focal-plane intensity\npx", fontsize=9)
    plt.tight_layout()
    fig.savefig(FIGURES / "fig5_mp_aa_achieved.png", dpi=150,
                bbox_inches='tight')
    plt.close(fig)
    print("  fig5 saved.")

    print(f"\nFigures saved to {FIGURES}")


if __name__ == "__main__":
    run_and_plot()
