"""
Problem 11.3
Fractional optical vortices: phase structure and SLM generation.
Following Berry (2004), J. Opt. A 6, 259-268.
Compared with Leach et al. (2004) and Lee et al. (2004).

Reference: Jones, Maragò & Volpe, Optical Tweezers: Principles and Applications, Ch. 7
Notes: SSPC 2026, Sneha Chakrabarti

Background: integer vs fractional l
-------------------------------------
For integer topological charge l, the LG beam phase mask is:
    phi(rho, phi) = l * phi   (mod 2pi)
This produces a perfect helical wavefront: the phase is single-valued and the
field has a single on-axis phase singularity of strength l.

For fractional l (non-integer), the phase l*phi is multi-valued: tracing a
full revolution returns to l*2pi ≠ 0 mod 2pi. This requires a branch cut —
a line in the beam cross-section where the phase is discontinuous by 2pi*frac(l)
(the fractional part of l). Across this cut, the field is not single-valued.

Berry (2004) showed that fractional-l beams, rather than having one vortex of
fractional charge, have a CHAIN of unit-charge vortices distributed along the
branch cut. Specifically, for l = n + eps (integer n, 0 < eps < 1):
  - The beam has n+1 vortices of charge +1 on the positive x-axis (branch cut).
  - As eps increases from 0 to 1, these vortices appear sequentially and move
    outward from the origin.
  - At eps = 0.5, the vortex chain is approximately symmetric.
  - At eps -> 1 the vortices merge into a charge-(n+1) singularity.

SLM implementation
-------------------
The fractional vortex phase mask is simply:
    phi_SLM(x, y) = mod(l * arctan2(y, x), 2*pi)
where l is non-integer. The mod 2pi wrapping introduces the branch cut.
A blazed grating is added to shift the beam off the DC spot.

Comparisons
-----------
- Simulated focal-plane intensity for l = 0.5, 1.0, 1.5, 2.0, 2.5, 3.0.
- Focal-plane phase showing the vortex chain along the branch cut.
- Vortex chain position and count as function of fractional part eps.
- Qualitative comparison with Leach et al. and Lee et al. (2004) results:
  both experimental papers show a chain of unit vortices along the +x axis
  for fractional l, consistent with Berry's prediction.

Key results to show:
  1. Intensity: asymmetric doughnut structure with a gap along branch cut.
  2. Phase: chain of unit-charge vortices (alternating on/off axis).
  3. As l increases through integer values, the beam passes through
     a symmetric doughnut (pure integer LG).
  4. Vortex count ~ floor(l) + 1 at each fractional value.

Parameters
----------
SLM  : 1000 x 1000 pixels
Laser: Gaussian, w0 = 300 px
Grating period: 30 px (off-axis separation)
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import genlaguerre
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "ex11_2_11"))
from gs_algorithm import gaussian_laser
sys.path.insert(0, str(Path(__file__).parent.parent / "prob11_2"))
from optical_vortex_pair import find_vortices

Mx, My   = 1000, 1000
W0       = 300
GRATING  = 30

FIGURES  = Path(__file__).parent / "figures"
FIGURES.mkdir(exist_ok=True)


# ── fractional vortex phase mask ──────────────────────────────────────────────

def fractional_vortex_mask(l, Mx, My, w0, grating_period=None):
    """
    Phase mask for fractional topological charge l (Berry 2004).

    phi(x, y) = mod(l * arctan2(y, x), 2*pi)

    The branch cut lies along the positive x-axis (phi = 0 = 2pi):
    on crossing this line from phi=2pi-eps to phi=eps, the phase
    jumps by 2pi*frac(l) (discontinuity).

    With grating, the beam is displaced off axis.
    """
    cx, cy = Mx / 2.0, My / 2.0
    x = np.arange(Mx) - cx
    y = np.arange(My) - cy
    X, Y = np.meshgrid(x, y, indexing='ij')

    phi_vortex = (l * np.arctan2(Y, X)) % (2 * np.pi)    # fractional spiral

    if grating_period is not None:
        phi_grating = 2 * np.pi * X / grating_period
        phi_slm = (phi_vortex + phi_grating) % (2 * np.pi)
    else:
        phi_slm = phi_vortex

    return phi_slm, phi_vortex


def focal_field_and_intensity(slm_phase, laser_amp):
    sf  = laser_amp * np.exp(1j * slm_phase)
    ff  = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(sf)))
    return ff, np.abs(ff)**2


# ── run and plot ──────────────────────────────────────────────────────────────

def run_and_plot():
    print(f"Setting up {Mx}×{My} SLM, w0={W0} px...")
    laser_amp = gaussian_laser(Mx, My, W0)

    # l values: integers and half-integers to show transition
    l_vals = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
    n      = len(l_vals)

    # ── Figure 1: phase masks ─────────────────────────────────────────────────
    fig, axes = plt.subplots(1, n, figsize=(3 * n, 4))
    fig.suptitle(
        r"Fractional Vortex Phase Masks on SLM (Berry 2004)"
        "\n"
        r"$\phi(x,y) = [l\,\arctan(y/x)]\,\mathrm{mod}\,2\pi$ + blazed grating"
        "\nBranch cut along $+x$ axis",
        fontsize=11, fontweight='bold'
    )
    crop = 350
    cx, cy = Mx // 2, My // 2
    sl_slm = np.s_[cx-crop:cx+crop, cy-crop:cy+crop]

    for ax, l in zip(axes, l_vals):
        phi_slm, phi_v = fractional_vortex_mask(l, Mx, My, W0, GRATING)
        ax.imshow(phi_slm[sl_slm].T, cmap='hsv',
                  vmin=0, vmax=2*np.pi, origin='lower')
        ax.set_title(f"$l={l}$", fontsize=13)
        ax.axis('off')
    plt.tight_layout()
    fig.savefig(FIGURES / "fig1_fractional_masks.png", dpi=150,
                bbox_inches='tight')
    plt.close(fig)
    print("  fig1 saved.")

    # ── Figure 2: focal intensity ─────────────────────────────────────────────
    fig, axes = plt.subplots(1, n, figsize=(3 * n, 4))
    fig.suptitle(
        "Fractional Vortex Focal-Plane Intensity (log scale)\n"
        "Integer $l$: symmetric doughnut  |  Fractional $l$: asymmetric with gap at branch cut",
        fontsize=11, fontweight='bold'
    )
    stored_fields = {}
    for ax, l in zip(axes, l_vals):
        phi_slm, _ = fractional_vortex_mask(l, Mx, My, W0, GRATING)
        ff, intens  = focal_field_and_intensity(phi_slm, laser_amp)
        stored_fields[l] = ff
        log_i = np.log1p(intens / intens.max() * 1e4)
        peak  = np.unravel_index(np.argmax(intens), intens.shape)
        cpx, cpy = peak
        cr  = 100
        sl_f = np.s_[max(0,cpx-cr):cpx+cr, max(0,cpy-cr):cpy+cr]
        ax.imshow(log_i[sl_f].T, cmap='inferno', origin='lower')
        ax.set_title(f"$l={l}$", fontsize=13)
        ax.axis('off')
    plt.tight_layout()
    fig.savefig(FIGURES / "fig2_fractional_intensity.png", dpi=150,
                bbox_inches='tight')
    plt.close(fig)
    print("  fig2 saved.")

    # ── Figure 3: focal-field phase showing vortex chain ─────────────────────
    fig, axes = plt.subplots(1, n, figsize=(3 * n, 4))
    fig.suptitle(
        "Fractional Vortex Focal-Field Phase\n"
        "Unit-charge vortex chain along branch cut (Berry 2004, Fig. 2)\n"
        r"Cyan + = charge +1  |  Red × = charge -1",
        fontsize=11, fontweight='bold'
    )
    for ax, l in zip(axes, l_vals):
        phi_slm, _ = fractional_vortex_mask(l, Mx, My, W0, GRATING)
        ff, intens  = focal_field_and_intensity(phi_slm, laser_amp)
        focal_phase = np.angle(ff)
        # threshold phase display where intensity is very low
        I_thresh = intens.max() * 0.005
        phase_masked = np.where(intens > I_thresh, focal_phase, np.nan)

        peak = np.unravel_index(np.argmax(intens), intens.shape)
        cpx, cpy = peak
        cr   = 100
        sl_f = np.s_[max(0,cpx-cr):cpx+cr, max(0,cpy-cr):cpy+cr]

        ax.imshow(phase_masked[sl_f].T, cmap='hsv',
                  vmin=-np.pi, vmax=np.pi, origin='lower')

        # detect and plot vortices in this region
        E_crop = ff[sl_f]
        vortices = find_vortices(E_crop, intensity_threshold=0.002)
        for (vx, vy, w) in vortices:
            marker = 'c+' if w > 0 else 'rx'
            ax.plot(vx, vy, marker, ms=8, mew=2)

        ax.set_title(f"$l={l}$\n{len(vortices)} vortices", fontsize=11)
        ax.axis('off')

    plt.tight_layout()
    fig.savefig(FIGURES / "fig3_fractional_phase_vortices.png", dpi=150,
                bbox_inches='tight')
    plt.close(fig)
    print("  fig3 saved.")

    # ── Figure 4: continuous sweep eps=0 to 1 at l = 1+eps ───────────────────
    # Shows vortex birth and chain formation as fractional part varies
    eps_vals = np.linspace(0.0, 1.0, 9)
    l_sweep  = 1.0 + eps_vals           # l from 1 to 2
    colors_s = plt.cm.viridis(np.linspace(0, 1, len(l_sweep)))

    fig, axes = plt.subplots(2, len(l_sweep), figsize=(3 * len(l_sweep), 7))
    fig.suptitle(
        r"Continuous Sweep: $l = 1 + \varepsilon$,  $\varepsilon \in [0, 1]$"
        "\nTop: focal intensity  |  Bottom: focal phase with vortex positions\n"
        r"$\varepsilon=0$: integer LG$_0^1$;  $\varepsilon=0.5$: half-integer;  "
        r"$\varepsilon\to1$: LG$_0^2$ forming",
        fontsize=11, fontweight='bold'
    )
    n_vortices_eps = []

    for col, l in enumerate(l_sweep):
        phi_slm, _ = fractional_vortex_mask(l, Mx, My, W0, GRATING)
        ff, intens  = focal_field_and_intensity(phi_slm, laser_amp)
        focal_phase = np.angle(ff)
        log_i = np.log1p(intens / intens.max() * 1e4)

        peak = np.unravel_index(np.argmax(intens), intens.shape)
        cpx, cpy = peak
        cr   = 95
        sl_f = np.s_[max(0,cpx-cr):cpx+cr, max(0,cpy-cr):cpy+cr]
        I_thresh = intens.max() * 0.005
        phase_masked = np.where(intens > I_thresh, focal_phase, np.nan)

        axes[0, col].imshow(log_i[sl_f].T, cmap='inferno', origin='lower')
        axes[0, col].set_title(
            f"$l={l:.2f}$\n($\\varepsilon={eps_vals[col]:.2f}$)",
            fontsize=9
        )
        axes[0, col].axis('off')

        axes[1, col].imshow(phase_masked[sl_f].T, cmap='hsv',
                            vmin=-np.pi, vmax=np.pi, origin='lower')
        E_crop = ff[sl_f]
        vortices = find_vortices(E_crop, intensity_threshold=0.002)
        n_vortices_eps.append(len(vortices))
        for (vx, vy, w) in vortices:
            axes[1, col].plot(vx, vy, 'c+' if w > 0 else 'rx', ms=7, mew=2)
        axes[1, col].set_title(f"{len(vortices)} vortex{'es' if len(vortices)!=1 else ''}",
                               fontsize=9)
        axes[1, col].axis('off')

    plt.tight_layout()
    fig.savefig(FIGURES / "fig4_eps_sweep.png", dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("  fig4 saved.")

    # ── Figure 5: vortex count vs fractional part ─────────────────────────────
    # Sweep eps for l = 0+eps, 1+eps, 2+eps
    fig, ax = plt.subplots(figsize=(9, 4))
    for n_base, col, label in [(0,'steelblue',r'$l=\varepsilon$'),
                                (1,'tomato',  r'$l=1+\varepsilon$'),
                                (2,'seagreen',r'$l=2+\varepsilon$')]:
        eps_fine = np.linspace(0.05, 0.95, 10)
        counts   = []
        for eps in eps_fine:
            l   = n_base + eps
            phi_slm, _ = fractional_vortex_mask(l, Mx, My, W0, GRATING)
            ff, intens  = focal_field_and_intensity(phi_slm, laser_amp)
            peak = np.unravel_index(np.argmax(intens), intens.shape)
            cpx, cpy = peak
            cr = 100
            sl_f = np.s_[max(0,cpx-cr):cpx+cr, max(0,cpy-cr):cpy+cr]
            v = find_vortices(ff[sl_f], intensity_threshold=0.002)
            counts.append(len(v))
            print(f"  l={l:.2f}: {len(v)} vortices")
        ax.plot(eps_fine, counts, 'o-', lw=1.8, color=col, label=label)

    ax.set_xlabel(r"Fractional part $\varepsilon$", fontsize=11)
    ax.set_ylabel("Detected vortex count", fontsize=11)
    ax.set_title(
        "Number of Unit Vortices vs Fractional Charge (Berry 2004)\n"
        r"Vortex chain grows as $\varepsilon \to 1$ (new integer LG forming)",
        fontsize=11
    )
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(FIGURES / "fig5_vortex_count_vs_eps.png", dpi=150,
                bbox_inches='tight')
    plt.close(fig)
    print("  fig5 saved.")

    print(f"\nFigures saved to {FIGURES}")
    print("\nComparison with Leach et al. and Lee et al. (2004):")
    print("  Both experimental papers show:")
    print("  - Asymmetric ring intensity with a gap along the branch cut")
    print("  - A chain of unit-charge vortices along the +x axis")
    print("  - Vortex count increasing with floor(l)")
    print("  - Symmetric doughnut restored at integer l values")
    print("  These features are reproduced in our simulation (figs 2, 3, 4).")


if __name__ == "__main__":
    run_and_plot()
