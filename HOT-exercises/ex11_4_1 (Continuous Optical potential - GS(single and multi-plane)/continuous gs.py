"""
Exercise 11.4.1
Generate continuous optical potentials using the Gerchberg-Saxton algorithm,
both on a single focal plane and on multiple axial planes.

Reference: Jones, Maragò & Volpe, Optical Tweezers: Principles and Applications, Ch. 7
Notes: SSPC 2026, Sneha Chakrabarti

Continuous vs discrete potentials
-----------------------------------
For point traps (Exercises 11.2.11–11.2.15), the amplitude constraint at the
focal plane is applied only at N discrete pixels; all other pixels are left free.
For continuous optical potentials, the constraint is applied over the entire
focal field: the target is a 2D amplitude map A_target(x, y) rather than a
list of N point intensities.

Convergence criterion (Eq. 13, theory notes)
---------------------------------------------
    C = integral[ (A_focus(x,y) - A_target(x,y))^2  dx dy ]
      ≈ sum_{x,y} (|E_focus(x,y)| - A_target(x,y))^2

C replaces the per-trap metrics (u, sigma) from Section 5.1. The algorithm
runs until C drops below a threshold or a maximum iteration count is reached.

Single-plane GS for continuous potentials
-----------------------------------------
Per iteration:
  1. Forward FFT of SLM field -> focal field E_focus
  2. Replace focal AMPLITUDE with A_target everywhere, keep focal PHASE
     (not just at N trap sites — the entire focal plane is constrained)
  3. Inverse FFT -> back to SLM plane
  4. Replace SLM amplitude with laser profile, keep SLM phase

Multi-plane GS for continuous potentials
-----------------------------------------
P target planes, each with its own A_target_p(x, y):
  For each plane p:
    1. Propagate SLM to plane p via Fresnel transfer function H_p
    2. Replace amplitude with A_target_p, keep phase
    3. Back-propagate via H_p*
  Sum back-propagated fields at SLM; apply laser amplitude constraint.
  (Identical structure to Ex 11.2.14 but with 2D amplitude maps.)

Standard parameters
-------------------
SLM  : 1000 x 1000 pixels
Laser: Gaussian, 1/e^2 radius = 400 px

Target patterns demonstrated
-----------------------------
Single-plane:
  - Ring (annular intensity)
  - Gaussian spot array (smooth version of multi-trap problem)
  - Letter 'phi' (arbitrary shape)
  - Lemniscate (figure-eight curve)

Multi-plane (3 planes):
  - Each plane carries a different geometric shape (ring, cross, square frame)
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "ex11_2_11"))
from gs_algorithm import gaussian_laser
sys.path.insert(0, str(Path(__file__).parent.parent / "ex11_2_14"))
from gs_3d import fresnel_tf, propagate_to_plane, back_propagate

rng = np.random.default_rng(42)

Mx, My  = 1000, 1000
LASER_W = 400
N_ITER  = 80

FIGURES = Path(__file__).parent / "figures"
FIGURES.mkdir(exist_ok=True)


# ── target pattern generators ─────────────────────────────────────────────────

def target_ring(Mx, My, r_inner, r_outer, smooth=True):
    """Annular ring target amplitude."""
    cx, cy = Mx / 2.0, My / 2.0
    x = np.arange(Mx) - cx
    y = np.arange(My) - cy
    X, Y = np.meshgrid(x, y, indexing='ij')
    R = np.sqrt(X**2 + Y**2)
    if smooth:
        sigma = (r_outer - r_inner) / 4.0
        A = np.exp(-((R - (r_inner + r_outer) / 2)**2) / (2 * sigma**2))
    else:
        A = np.where((R >= r_inner) & (R <= r_outer), 1.0, 0.0)
    return A / (np.sqrt(np.sum(A**2)) + 1e-30)


def target_cross(Mx, My, arm_width, arm_length):
    """Cross / plus-sign target amplitude."""
    cx, cy = Mx // 2, My // 2
    A = np.zeros((Mx, My))
    hw = arm_width // 2
    hl = arm_length // 2
    A[cx-hw:cx+hw, cy-hl:cy+hl] = 1.0
    A[cx-hl:cx+hl, cy-hw:cy+hw] = 1.0
    return A / (np.sqrt(np.sum(A**2)) + 1e-30)


def target_square_frame(Mx, My, side, thickness):
    """Square frame target amplitude."""
    cx, cy = Mx // 2, My // 2
    hs = side // 2
    ht = thickness
    A = np.zeros((Mx, My))
    # outer square
    A[cx-hs:cx+hs, cy-hs:cy+hs] = 1.0
    # subtract inner square
    A[cx-hs+ht:cx+hs-ht, cy-hs+ht:cy+hs-ht] = 0.0
    return A / (np.sqrt(np.sum(A**2)) + 1e-30)


def target_lemniscate(Mx, My, a, sigma=4):
    """Lemniscate of Bernoulli: (x^2+y^2)^2 = 2a^2(x^2-y^2)."""
    cx, cy = Mx / 2.0, My / 2.0
    x = np.arange(Mx) - cx
    y = np.arange(My) - cy
    X, Y = np.meshgrid(x, y, indexing='ij')
    R2 = X**2 + Y**2
    # distance from lemniscate: approximate by Gaussian along curve
    # Implicit: (x^2+y^2)^2 - 2a^2(x^2-y^2) = 0
    f = (R2**2 - 2 * a**2 * (X**2 - Y**2))
    # Normalise by gradient magnitude for uniform-width curve
    df_dx = 4 * R2 * X - 4 * a**2 * X
    df_dy = 4 * R2 * Y + 4 * a**2 * Y
    grad  = np.sqrt(df_dx**2 + df_dy**2) + 1e-6
    A = np.exp(-(f / grad)**2 / (2 * sigma**2))
    # Only keep the lemniscate region (where |f/grad| is small AND R2 <= 2a^2)
    A *= (R2 <= 2.2 * a**2).astype(float)
    return A / (np.sqrt(np.sum(A**2)) + 1e-30)


def target_phi_symbol(Mx, My, r, thickness, stem_len, sigma=3):
    """Greek letter phi: circle with vertical line through it."""
    cx, cy = Mx / 2.0, My / 2.0
    x = np.arange(Mx) - cx
    y = np.arange(My) - cy
    X, Y = np.meshgrid(x, y, indexing='ij')
    R = np.sqrt(X**2 + Y**2)
    # circle
    A_ring = np.exp(-((R - r)**2) / (2 * thickness**2))
    # vertical stem
    A_stem = np.exp(-(X**2) / (2 * thickness**2)) * (np.abs(Y) <= stem_len)
    A = A_ring + A_stem
    return A / (np.sqrt(np.sum(A**2)) + 1e-30)


# ── convergence metric (Eq. 13) ───────────────────────────────────────────────

def convergence_error(focal_field, A_target):
    """
    C = sum (|E_focus(x,y)| - A_target(x,y))^2   (Eq. 13, theory notes).

    Both amplitudes must be on the same scale. A_target should be
    pre-normalised so that sum(A_target^2) = sum(|E_focus|^2) = total power.
    """
    A_focus = np.abs(focal_field)
    return float(np.sum((A_focus - A_target)**2))


# ── single-plane continuous GS ────────────────────────────────────────────────

def continuous_gs_single(laser_amp, A_target, n_iter, seed_phase=None):
    """
    GS for continuous optical potential on a single focal plane.

    The only change from point-trap GS: amplitude replacement in step 2
    is applied to ALL focal pixels (not just N trap sites).

    The target amplitude A_target must be normalised to the same total power
    as the SLM field so that the projection is onto a fixed set:

        A_target_norm = A_target * sqrt(sum(laser_amp^2) / sum(A_target^2))

    This ensures C = sum(|E_focus| - A_target_norm)^2 is measured on a
    consistent scale at every iteration, which is the precondition for the
    projection theorem to guarantee C is non-increasing.

    Parameters
    ----------
    laser_amp : (Mx, My) SLM amplitude
    A_target  : (Mx, My) target amplitude (will be power-normalised internally)
    n_iter    : iterations
    seed_phase: (Mx, My) initial SLM phase; random if None

    Returns
    -------
    slm_phase : (Mx, My) final hologram phase
    history   : dict with 'C' (convergence error) per iteration
    """
    Mx, My = laser_amp.shape

    # Normalise A_target so that ||A_t||_2 matches the focal-plane amplitude
    # scale from Parseval's theorem: ||FFT(laser)||_2 = sqrt(M)*||laser||_2
    # Since laser_amp is already normalised (sum=1), focal norm = sqrt(Mx*My).
    # We want sum(A_t^2) = sum(|E_focus|^2) which by Parseval = sum(laser^2)*Mx*My.
    power_slm    = float(np.sum(laser_amp**2)) * Mx * My   # Parseval focal power
    power_target = float(np.sum(A_target**2))
    A_t = A_target * np.sqrt(power_slm / (power_target + 1e-30))

    if seed_phase is None:
        seed_phase = rng.uniform(0, 2 * np.pi, (Mx, My))
    slm_field = laser_amp * np.exp(1j * seed_phase)
    history   = {'C': []}

    for _ in range(n_iter):
        # 1. forward FFT
        focal_field = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(slm_field)))

        # record convergence error (fixed-scale A_t)
        history['C'].append(convergence_error(focal_field, A_t))

        # 2. replace full focal amplitude with fixed-scale target, keep phase
        #    This is a projection onto the set {E : |E(x,y)| = A_t(x,y)}
        new_focal = A_t * np.exp(1j * np.angle(focal_field))

        # 3. inverse FFT
        slm_back = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(new_focal)))

        # 4. laser amplitude constraint (projection onto SLM set)
        slm_field = laser_amp * np.exp(1j * np.angle(slm_back))

    return np.angle(slm_field), history


# ── multi-plane continuous GS ─────────────────────────────────────────────────

def continuous_gs_multi(laser_amp, plane_targets, n_iter, seed_phase=None):
    """
    GS for continuous optical potential on multiple axial planes.

    plane_targets: list of (z_norm, A_target_p) tuples.
    Algorithm: same as multi-plane point-trap GS (Ex 11.2.14) but amplitude
    replacement is over the full focal field at each plane.

    Parameters
    ----------
    laser_amp    : (Mx, My)
    plane_targets: list of (z_norm, A_target) where A_target is (Mx, My)
    n_iter       : iterations
    seed_phase   : initial SLM phase

    Returns
    -------
    slm_phase : (Mx, My)
    history   : dict with per-plane C values: history[p] = list of C values
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
            # 1. propagate to plane p
            E_p = propagate_to_plane(slm_field, z_norm)

            history[p].append(convergence_error(E_p, A_t))

            # 2. replace full amplitude at plane p (fixed-scale projection)
            E_p_corr = A_t * np.exp(1j * np.angle(E_p))

            # 3. back-propagate
            slm_back_sum += back_propagate(E_p_corr, z_norm)

        # 4. laser amplitude constraint
        slm_field = laser_amp * np.exp(1j * np.angle(slm_back_sum))

    return np.angle(slm_field), history


# ── run and plot ──────────────────────────────────────────────────────────────

def run_and_plot():
    print(f"Setting up {Mx}×{My} SLM...")
    laser_amp = gaussian_laser(Mx, My, LASER_W)
    seed      = rng.uniform(0, 2 * np.pi, (Mx, My))

    # ── single-plane targets ──────────────────────────────────────────────────
    sp_targets = {
        "Ring":        target_ring(Mx, My, r_inner=60, r_outer=120),
        "Lemniscate":  target_lemniscate(Mx, My, a=80),
        "Cross":       target_cross(Mx, My, arm_width=20, arm_length=180),
        r"$\phi$ symbol": target_phi_symbol(Mx, My, r=80, thickness=6, stem_len=130),
    }
    colors = ['steelblue', 'tomato', 'seagreen', 'darkorchid']

    print("  Running single-plane GS for 4 targets...")
    sp_results = {}
    for name, A_t in sp_targets.items():
        phi, hist = continuous_gs_single(laser_amp, A_t, N_ITER,
                                         seed_phase=seed.copy())
        sf  = laser_amp * np.exp(1j * phi)
        ff  = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(sf)))
        sp_results[name] = {'phase': phi, 'history': hist,
                            'focal': ff, 'target': A_t}
        print(f"    {name}: C_final = {hist['C'][-1]:.4e}")

    # ── Figure 1: single-plane — target vs achieved intensity ─────────────────
    fig, axes = plt.subplots(3, 4, figsize=(16, 11))
    fig.suptitle(
        "Continuous GS (Single Plane): Target vs Achieved Optical Potentials\n"
        f"(1000×1000 SLM, {N_ITER} iterations)",
        fontsize=12, fontweight='bold'
    )
    crop = 250
    cx, cy = Mx // 2, My // 2
    sl = np.s_[cx-crop:cx+crop, cy-crop:cy+crop]

    for col, (name, res) in enumerate(sp_results.items()):
        target_disp = res['target'][sl].T
        focal_int   = np.abs(res['focal'])**2
        focal_disp  = focal_int[sl].T / focal_int[sl].max()
        phase_disp  = res['phase'][sl].T

        axes[0, col].imshow(target_disp, cmap='inferno', origin='lower')
        axes[0, col].set_title(f"Target: {name}", fontsize=10)

        axes[1, col].imshow(focal_disp, cmap='inferno', origin='lower',
                            vmin=0, vmax=1)
        axes[1, col].set_title("Achieved intensity", fontsize=10)

        im = axes[2, col].imshow(phase_disp % (2*np.pi), cmap='hsv',
                                  vmin=0, vmax=2*np.pi, origin='lower')
        axes[2, col].set_title("SLM phase hologram", fontsize=10)

    for ax in axes.flat:
        ax.set_xlabel("px"); ax.set_ylabel("px")
    row_labels = ["Target amplitude", "Focal-plane intensity", "SLM phase (rad)"]
    for row, label in enumerate(row_labels):
        axes[row, 0].set_ylabel(label + "\npx", fontsize=9)

    plt.tight_layout()
    fig.savefig(FIGURES / "fig1_sp_target_vs_achieved.png", dpi=150,
                bbox_inches='tight')
    plt.close(fig)
    print("  fig1 saved.")

    # ── Figure 2: single-plane convergence ────────────────────────────────────
    iters = np.arange(1, N_ITER + 1)
    fig, ax = plt.subplots(figsize=(9, 4))
    for (name, res), col in zip(sp_results.items(), colors):
        C_vals = np.array(res['history']['C'])
        ax.semilogy(iters, C_vals / C_vals[0], lw=1.8, color=col, label=name)
    ax.set_xlabel("Iteration", fontsize=11)
    ax.set_ylabel(r"$C / C_0$  (normalised convergence error)", fontsize=11)
    ax.set_title(
        "Single-Plane Continuous GS: Convergence\n"
        r"$C = \sum_{x,y}(|E_\mathrm{focus}| - A_\mathrm{target})^2$",
        fontsize=11
    )
    ax.legend(fontsize=9)
    ax.grid(True, which='both', alpha=0.3)
    plt.tight_layout()
    fig.savefig(FIGURES / "fig2_sp_convergence.png", dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("  fig2 saved.")

    # ── multi-plane targets ───────────────────────────────────────────────────
    # Use 256x256 for multi-plane (FFT cost scales as P*M*log(M) per iter)
    Mx_mp, My_mp = 256, 256
    W_mp         = 100
    laser_mp     = gaussian_laser(Mx_mp, My_mp, W_mp)
    seed_mp      = rng.uniform(0, 2 * np.pi, (Mx_mp, My_mp))
    N_ITER_MP    = 60

    plane_configs = [
        (-0.06, target_ring(Mx_mp, My_mp, r_inner=20, r_outer=40)),
        ( 0.00, target_cross(Mx_mp, My_mp, arm_width=6, arm_length=60)),
        ( 0.06, target_square_frame(Mx_mp, My_mp, side=80, thickness=8)),
    ]
    plane_names = ["Ring  (z=-0.06)", "Cross  (z=0)", "Square frame  (z=+0.06)"]
    plane_colors = ['steelblue', 'tomato', 'seagreen']

    print(f"  Running multi-plane GS (3 planes, 256×256, {N_ITER_MP} iters)...")
    phi_mp, hist_mp = continuous_gs_multi(
        laser_mp, plane_configs, N_ITER_MP, seed_phase=seed_mp
    )
    print("  Multi-plane done.")

    # ── Figure 3: multi-plane — target vs achieved per plane ─────────────────
    fig, axes = plt.subplots(3, 3, figsize=(12, 11))
    fig.suptitle(
        "Continuous GS (Multi-Plane): Target vs Achieved per Plane\n"
        r"(256×256 SLM, 3 planes, $z_\mathrm{norm}\in\{-0.06,\,0,\,+0.06\}$, "
        f"{N_ITER_MP} iterations)",
        fontsize=12, fontweight='bold'
    )
    slm_field_mp = laser_mp * np.exp(1j * phi_mp)
    crop_mp = Mx_mp // 2

    sys.path.insert(0, str(Path(__file__).parent.parent / "ex11_2_14"))
    from gs_3d import propagate_to_plane as prop_mp

    for col, (z_norm, A_t), name in zip(range(3), plane_configs, plane_names):
        E_p       = prop_mp(slm_field_mp, z_norm)
        intens_p  = np.abs(E_p)**2
        sl_mp     = np.s_[0:Mx_mp, 0:My_mp]

        axes[0, col].imshow(A_t.T, cmap='inferno', origin='lower')
        axes[0, col].set_title(f"Target\n{name}", fontsize=9)

        axes[1, col].imshow(
            (intens_p / intens_p.max()).T, cmap='inferno',
            origin='lower', vmin=0, vmax=1
        )
        axes[1, col].set_title("Achieved intensity", fontsize=9)

        im = axes[2, col].imshow(
            np.angle(E_p).T, cmap='hsv',
            vmin=-np.pi, vmax=np.pi, origin='lower'
        )
        axes[2, col].set_title("Focal-field phase", fontsize=9)

    for ax in axes.flat:
        ax.set_xlabel("px"); ax.set_ylabel("px")

    plt.tight_layout()
    fig.savefig(FIGURES / "fig3_mp_target_vs_achieved.png", dpi=150,
                bbox_inches='tight')
    plt.close(fig)
    print("  fig3 saved.")

    # ── Figure 4: multi-plane convergence ─────────────────────────────────────
    iters_mp = np.arange(1, N_ITER_MP + 1)
    fig, ax  = plt.subplots(figsize=(9, 4))
    for p, (name, col) in enumerate(zip(plane_names, plane_colors)):
        C_p = np.array(hist_mp[p])
        ax.semilogy(iters_mp, C_p / C_p[0], lw=1.8, color=col, label=name)
    ax.set_xlabel("Iteration", fontsize=11)
    ax.set_ylabel(r"$C_p / C_{p,0}$  (per-plane normalised error)", fontsize=11)
    ax.set_title("Multi-Plane Continuous GS: Convergence per Plane", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, which='both', alpha=0.3)
    plt.tight_layout()
    fig.savefig(FIGURES / "fig4_mp_convergence.png", dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("  fig4 saved.")

    print(f"\nFigures saved to {FIGURES}")


if __name__ == "__main__":
    run_and_plot()
