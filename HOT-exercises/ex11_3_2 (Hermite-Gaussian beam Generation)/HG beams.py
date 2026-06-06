"""
Exercise 11.3.2
Show that a Hermite-Gaussian beam HG_{mx,my} can be generated using the phase
modulation  phi(x,y) = Theta{ H_mx(sqrt(2)*x/w0) * H_my(sqrt(2)*y/w0) }
and simulate its generation using an appropriate phase mask.

Reference: Jones, Maragò & Volpe, Optical Tweezers: Principles and Applications, Ch. 7
Notes: SSPC 2026, Sneha Chakrabarti

Theory
------
The HG field at the beam waist (z=0) is (hint equation):

    E^HG_{mx,my}(x,y,0) = E^G(x,y,0) * H_mx(sqrt(2)*x/w0) * H_my(sqrt(2)*y/w0)

where E^G is the Gaussian field and H_n is the n-th Hermite polynomial.

The sign structure of H_mx * H_my divides the transverse plane into (mx+1)(my+1)
rectangular lobes that alternate in sign. The phase mask that encodes this on
the SLM is:

    phi(x,y) = Theta{ H_mx(sqrt(2)*x/w0) * H_my(sqrt(2)*y/w0) }
             = { 0    if H_mx * H_my > 0
               { pi   if H_mx * H_my < 0

i.e. a binary pi phase grating matching the nodal structure of the HG mode.
This converts a Gaussian beam (E^G > 0 everywhere) into E^G * sign(H_mx * H_my),
which is the real part of the HG field (same amplitude, correct phase signs).

Why this works
--------------
The Heaviside step Theta{f} = (1 + sign(f))/2, so:

    exp(i * pi * Theta{H_mx * H_my}) = exp(i * pi/2) * exp(i * pi/2 * sign(H_mx*H_my))
                                     ∝ sign(H_mx * H_my)

Multiplying E^G by this phase factor (modulo a global phase) gives a field
whose real part matches the HG amplitude structure. The resulting focal-plane
intensity shows the rectangular lobe pattern characteristic of HG modes:
  - mx nodal lines parallel to y
  - my nodal lines parallel to x
  - (mx+1)*(my+1) lobes total, arranged in a grid

Note: this is a binary phase mask (0 or pi only), unlike the continuous LG mask.
This is analogous to a ferroelectric LC SLM (binary phase levels, Section 3.2).

Gouy phase
----------
The full propagated HG field picks up a Gouy phase -i*(mx+my)*arctan(z/z_R),
which is a global phase at z=0 and does not affect intensity. This is why the
phase mask at z=0 suffices to generate the mode; no z-dependent correction needed.

Standard benchmark
------------------
SLM  : 1000 × 1000 pixels
Laser: Gaussian, 1/e^2 radius w_0 = 250 px

Study
-----
1. Generate HG masks for (mx, my) up to order 4.
2. Show the phase mask (binary, 0/pi) and resulting focal-plane intensity.
3. Verify nodal line count: mx lines in x, my lines in y.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import hermite
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "ex11_2_11"))
from gs_algorithm import gaussian_laser

# ── parameters ────────────────────────────────────────────────────────────────
Mx, My  = 1000, 1000
W0      = 250          # beam waist in pixels
GRATING = 40           # blazed grating period (pixels)

FIGURES = Path(__file__).parent / "figures"
FIGURES.mkdir(exist_ok=True)


# ── HG phase mask ─────────────────────────────────────────────────────────────

def hg_phase_mask(mx, my, w0, Mx, My, grating_period=None):
    """
    Compute the binary SLM phase mask for HG_{mx,my}.

    phi(x,y) = pi * Theta{ H_mx(sqrt(2)*x/w0) * H_my(sqrt(2)*y/w0) }
             = { 0   where H_mx * H_my > 0
               { pi  where H_mx * H_my < 0

    A blazed grating (linear ramp) is optionally superimposed to shift
    the beam off the zeroth-order DC spot.

    Parameters
    ----------
    mx, my         : Hermite-Gaussian orders (non-negative integers)
    w0             : beam waist in pixels
    Mx, My         : SLM dimensions
    grating_period : blazed grating period in pixels (None = no grating)

    Returns
    -------
    phi_slm : (Mx, My) phase in [0, 2pi)
    phi_hg  : (Mx, My) raw HG phase (0 or pi, no grating)
    Hmx_Hmy : (Mx, My) product H_mx * H_my (for visualisation)
    """
    cx, cy = Mx / 2.0, My / 2.0
    x = np.arange(Mx) - cx
    y = np.arange(My) - cy
    X, Y = np.meshgrid(x, y, indexing='ij')

    # evaluate Hermite polynomials (scipy uses probabilist's convention;
    # we need physicist's H_n, which is hermite(n) in scipy)
    Hx = hermite(mx)(np.sqrt(2) * X / w0)    # H_mx(sqrt(2)*x/w0)
    Hy = hermite(my)(np.sqrt(2) * Y / w0)    # H_my(sqrt(2)*y/w0)
    product = Hx * Hy

    # binary phase: 0 where product > 0, pi where product < 0
    phi_hg = np.where(product < 0, np.pi, 0.0)    # (Mx, My)

    if grating_period is not None:
        phi_grating = 2 * np.pi * X / grating_period
        phi_slm = (phi_hg + phi_grating) % (2 * np.pi)
    else:
        phi_slm = phi_hg.copy()

    return phi_slm, phi_hg, product


def focal_field(slm_phase, laser_amp):
    slm_field = laser_amp * np.exp(1j * slm_phase)
    return np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(slm_field)))


# ── run and plot ──────────────────────────────────────────────────────────────

def run_and_plot():
    print(f"Setting up {Mx}×{My} SLM, w0={W0} px...")
    laser_amp = gaussian_laser(Mx, My, W0)

    # ── Figure 1: mode gallery — phase masks ─────────────────────────────────
    # show HG modes (mx, my) for mx+my <= 4
    cases = [(mx, my) for mx in range(5) for my in range(5) if mx + my <= 4]
    n     = len(cases)
    ncols = 5
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(15, 6))
    fig.suptitle(
        r"HG Phase Masks: $\phi(x,y) = \pi\,\Theta\{H_{m_x}(\sqrt{2}x/w_0)\,H_{m_y}(\sqrt{2}y/w_0)\}$"
        "\n(binary 0/π mask, 1000×1000 SLM, central 500×500 crop shown)",
        fontsize=11, fontweight='bold'
    )
    crop = 250
    cx, cy = Mx // 2, My // 2
    sl = np.s_[cx-crop:cx+crop, cy-crop:cy+crop]

    for ax, (mx, my) in zip(axes.flat, cases):
        phi_slm, phi_hg, _ = hg_phase_mask(mx, my, W0, Mx, My,
                                            grating_period=GRATING)
        ax.imshow(phi_slm[sl].T, cmap='RdBu', origin='lower',
                  vmin=0, vmax=2*np.pi)
        ax.set_title(f"HG$_{{{mx},{my}}}$", fontsize=10)
        ax.axis('off')
    for ax in axes.flat[n:]:
        ax.axis('off')
    plt.tight_layout()
    fig.savefig(FIGURES / "fig1_hg_phase_masks.png", dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("  fig1 saved.")

    # ── Figure 2: mode gallery — focal intensity ──────────────────────────────
    fig, axes = plt.subplots(nrows, ncols, figsize=(15, 6))
    fig.suptitle(
        "HG Beam Focal-Plane Intensity (log scale)\n"
        "Rectangular lobe pattern: $(m_x+1)(m_y+1)$ lobes, "
        "$m_x$ nodal lines in $x$, $m_y$ in $y$",
        fontsize=11, fontweight='bold'
    )
    for ax, (mx, my) in zip(axes.flat, cases):
        phi_slm, _, _ = hg_phase_mask(mx, my, W0, Mx, My,
                                       grating_period=GRATING)
        ff     = focal_field(phi_slm, laser_amp)
        intens = np.abs(ff)**2
        log_i  = np.log1p(intens / intens.max() * 1e4)
        peak   = np.unravel_index(np.argmax(intens), intens.shape)
        cpx, cpy = peak
        cr = 90
        sl_p = np.s_[max(0,cpx-cr):cpx+cr, max(0,cpy-cr):cpy+cr]
        ax.imshow(log_i[sl_p].T, cmap='inferno', origin='lower')
        ax.set_title(f"HG$_{{{mx},{my}}}$", fontsize=10)
        ax.axis('off')
    for ax in axes.flat[n:]:
        ax.axis('off')
    plt.tight_layout()
    fig.savefig(FIGURES / "fig2_hg_focal_intensity.png", dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("  fig2 saved.")

    # ── Figure 3: detailed view of one mode — mask, product, intensity ────────
    # Focus on HG_{3,2} to show the 4x3 lobe structure clearly
    mx_demo, my_demo = 3, 2
    phi_slm, phi_hg, product = hg_phase_mask(mx_demo, my_demo, W0, Mx, My,
                                              grating_period=GRATING)
    ff     = focal_field(phi_slm, laser_amp)
    intens = np.abs(ff)**2
    log_i  = np.log1p(intens / intens.max() * 1e4)
    peak   = np.unravel_index(np.argmax(intens), intens.shape)
    cpx, cpy = peak

    cr_mask = 300
    cr_foc  = 120
    slm_sl  = np.s_[cx-cr_mask:cx+cr_mask, cy-cr_mask:cy+cr_mask]
    foc_sl  = np.s_[max(0,cpx-cr_foc):cpx+cr_foc, max(0,cpy-cr_foc):cpy+cr_foc]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    fig.suptitle(
        f"HG$_{{{mx_demo},{my_demo}}}$: Phase Mask, Hermite Product, Focal Intensity\n"
        f"({mx_demo+1}×{my_demo+1} = {(mx_demo+1)*(my_demo+1)} lobes; "
        f"{mx_demo} nodal lines in x, {my_demo} in y)",
        fontsize=12, fontweight='bold'
    )
    # product H_mx * H_my (sign structure)
    vabs = np.percentile(np.abs(product[slm_sl]), 99)
    im0 = axes[0].imshow(product[slm_sl].T, cmap='RdBu_r',
                          vmin=-vabs, vmax=vabs, origin='lower')
    axes[0].set_title(r"$H_{m_x}(\sqrt{2}x/w_0)\cdot H_{m_y}(\sqrt{2}y/w_0)$",
                      fontsize=10)
    plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    im1 = axes[1].imshow(phi_hg[slm_sl].T, cmap='RdBu',
                          vmin=0, vmax=np.pi, origin='lower')
    axes[1].set_title("Binary phase mask  (0 or π)\n+ blazed grating", fontsize=10)
    plt.colorbar(im1, ax=axes[1], label='Phase (rad)', fraction=0.046, pad=0.04)

    axes[2].imshow(log_i[foc_sl].T, cmap='inferno', origin='lower')
    axes[2].set_title("Focal-plane intensity (log scale)", fontsize=10)

    for ax in axes:
        ax.set_xlabel("pixel x"); ax.set_ylabel("pixel y")
    plt.tight_layout()
    fig.savefig(FIGURES / "fig3_hg_demo.png", dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("  fig3 saved.")

    # ── Figure 4: nodal structure verification ────────────────────────────────
    # Cross-section through intensity for HG_{3,0} and HG_{0,3}
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    fig.suptitle(
        "Nodal Line Verification: Intensity Cross-Sections\n"
        r"HG$_{3,0}$ (3 x-nodes, 0 y-nodes) and HG$_{0,3}$ (0 x-nodes, 3 y-nodes)",
        fontsize=12, fontweight='bold'
    )
    for row, (mx, my) in enumerate([(3, 0), (0, 3)]):
        phi_slm, _, _ = hg_phase_mask(mx, my, W0, Mx, My, grating_period=GRATING)
        ff     = focal_field(phi_slm, laser_amp)
        intens = np.abs(ff)**2
        log_i  = np.log1p(intens / intens.max() * 1e4)
        peak   = np.unravel_index(np.argmax(intens), intens.shape)
        cpx, cpy = peak
        cr = 100
        sl_f = np.s_[max(0,cpx-cr):cpx+cr, max(0,cpy-cr):cpy+cr]
        sub   = intens[sl_f]

        axes[row, 0].imshow(log_i[sl_f].T, cmap='inferno', origin='lower')
        axes[row, 0].set_title(f"HG$_{{{mx},{my}}}$: focal intensity", fontsize=11)
        axes[row, 0].set_xlabel("x (px)"); axes[row, 0].set_ylabel("y (px)")

        # cross-section through beam centre (row of peak)
        row_idx = sub.shape[1] // 2
        col_idx = sub.shape[0] // 2
        xs      = np.arange(sub.shape[0]) - sub.shape[0] // 2
        ys      = np.arange(sub.shape[1]) - sub.shape[1] // 2

        axes[row, 1].plot(xs, sub[:, row_idx] / sub.max(),
                          color='steelblue', lw=1.8, label='x cross-section')
        axes[row, 1].plot(ys, sub[col_idx, :] / sub.max(),
                          color='tomato',    lw=1.8, ls='--', label='y cross-section')
        axes[row, 1].set_xlabel("Pixel offset from peak", fontsize=10)
        axes[row, 1].set_ylabel("Normalised intensity", fontsize=10)
        axes[row, 1].set_title(f"HG$_{{{mx},{my}}}$: intensity cross-sections", fontsize=11)
        axes[row, 1].legend(fontsize=9)
        axes[row, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(FIGURES / "fig4_nodal_verification.png", dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("  fig4 saved.")

    print(f"\nFigures saved to {FIGURES}")


if __name__ == "__main__":
    run_and_plot()
