"""
Exercise 11.3.1
Simulate the generation of Laguerre-Gaussian beams using a phase mask.

Reference: Jones, Maragò & Volpe, Optical Tweezers: Principles and Applications, Ch. 7
Notes: SSPC 2026, Sneha Chakrabarti

Theory
------
A Laguerre-Gaussian beam LG_p^l is characterised by:
  l : azimuthal index (topological charge)   — OAM of l*hbar per photon
  p : radial index                           — number of radial rings

The phase profile at the beam waist w_0 (Eq. 12 of theory notes) is:

    phi(rho, phi) = l*phi  +  pi * Theta{ L_p^|l|(2*rho^2/w_0^2) }

where:
  l*phi         : helical wavefront — |l| full 2pi wraps per revolution.
                  Creates a phase singularity on axis (intensity zero, doughnut).
  pi*Theta{...} : sign changes of the radial Laguerre polynomial L_p^|l|.
                  L_p^|l| has p zeros; the Heaviside step encodes the pi phase
                  flip across each zero, producing p+1 concentric rings.

Implementation on the SLM
--------------------------
The SLM encodes phi(rho, phi) on a Gaussian incident beam. To avoid
contamination from the unmodulated zeroth-order reflection (undiffracted
beam at DC), a blazed grating (linear phase ramp) is superimposed:

    phi_SLM(x, y) = [ phi(rho, phi)  +  2*pi * (x/Lambda_x + y/Lambda_y) ] mod 2*pi

where Lambda_x, Lambda_y set the grating period (pixels per fringe).
This deflects the first-order diffracted beam off axis so the ring structure
is visible without the DC spot.

Standard benchmark
------------------
SLM  : 1000 × 1000 pixels
Laser: Gaussian, 1/e^2 radius w_0 = 300 px (wider beam fills more rings)

Study
-----
1. Generate LG masks for (l, p) in {(1,0), (3,0), (1,1), (3,2), (-2,0), (-3,1)}.
2. Show phase mask, SLM field amplitude and the simulated focal-plane intensity.
3. Verify doughnut ring count and OAM sign (phase winding direction).
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from scipy.special import genlaguerre
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "ex11_2_11"))
from gs_algorithm import gaussian_laser

# ── parameters ────────────────────────────────────────────────────────────────
Mx, My  = 1000, 1000
W0      = 300          # beam waist in SLM pixels
GRATING = 30           # blazed grating period in pixels (shift off DC)

FIGURES = Path(__file__).parent / "figures"
FIGURES.mkdir(exist_ok=True)


# ── coordinate grids ──────────────────────────────────────────────────────────

def polar_grid(Mx, My):
    """Centred polar coordinates on the SLM plane (pixels)."""
    cx, cy = Mx / 2.0, My / 2.0
    x = np.arange(Mx) - cx
    y = np.arange(My) - cy
    X, Y = np.meshgrid(x, y, indexing='ij')
    R   = np.sqrt(X**2 + Y**2)
    Phi = np.arctan2(Y, X)
    return R, Phi, X, Y


# ── LG phase mask ─────────────────────────────────────────────────────────────

def lg_phase_mask(l, p, w0, Mx, My, grating_period=None):
    """
    Compute the SLM phase mask for LG_p^l.

    phi(rho, phi) = l*phi  +  pi * Theta{ L_p^|l|(2*rho^2/w0^2) }

    With optional blazed grating superimposed to shift the beam off axis.

    Parameters
    ----------
    l              : azimuthal (topological charge); signed integer
    p              : radial index; non-negative integer
    w0             : beam waist in pixels
    Mx, My         : SLM dimensions
    grating_period : period of blazed grating in pixels (None = no grating)

    Returns
    -------
    phi_slm : (Mx, My) phase in [0, 2pi)
    phi_lg  : (Mx, My) raw LG phase (without grating, for display)
    """
    R, Phi, X, Y = polar_grid(Mx, My)

    # azimuthal spiral phase
    phi_azimuthal = l * Phi                              # (Mx, My)

    # radial phase: pi flips across zeros of L_p^|l|
    u = 2 * R**2 / w0**2                                 # argument of Laguerre poly
    Lp = genlaguerre(p, abs(l))(u)                       # L_p^|l|(u)
    phi_radial = np.pi * (Lp < 0).astype(float)          # Heaviside step

    phi_lg = phi_azimuthal + phi_radial                  # full LG phase

    # add blazed grating if requested
    if grating_period is not None:
        phi_grating = 2 * np.pi * X / grating_period
        phi_slm = (phi_lg + phi_grating) % (2 * np.pi)
    else:
        phi_slm = phi_lg % (2 * np.pi)

    return phi_slm, phi_lg


# ── propagate to focal plane ──────────────────────────────────────────────────

def focal_field(slm_phase, laser_amp):
    """Fourier-plane field from SLM phase mask."""
    slm_field = laser_amp * np.exp(1j * slm_phase)
    ff = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(slm_field)))
    return ff


# ── run and plot ──────────────────────────────────────────────────────────────

def run_and_plot():
    print(f"Setting up {Mx}×{My} SLM, w0={W0} px...")
    laser_amp = gaussian_laser(Mx, My, W0)

    # cases to study: (l, p, label)
    cases = [
        ( 1, 0, "LG$_0^1$"),
        ( 3, 0, "LG$_0^3$"),
        (-2, 0, "LG$_0^{-2}$"),
        ( 1, 1, "LG$_1^1$"),
        ( 3, 2, "LG$_2^3$"),
        (-3, 1, "LG$_1^{-3}$"),
    ]
    n_cases = len(cases)

    # ── Figure 1: phase masks ─────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 3, figsize=(13, 8))
    fig.suptitle(
        "LG Beam Phase Masks on SLM  (1000×1000)\n"
        r"$\phi(\rho,\varphi) = l\varphi + \pi\,\Theta\{L_p^{|l|}(2\rho^2/w_0^2)\}$"
        " + blazed grating",
        fontsize=12, fontweight='bold'
    )
    for ax, (l, p, label) in zip(axes.flat, cases):
        phi_slm, _ = lg_phase_mask(l, p, W0, Mx, My, grating_period=GRATING)
        # show central 600×600 crop for clarity
        crop = 300
        cx, cy = Mx // 2, My // 2
        sl = np.s_[cx-crop:cx+crop, cy-crop:cy+crop]
        im = ax.imshow(phi_slm[sl].T, cmap='hsv', vmin=0, vmax=2*np.pi,
                       origin='lower')
        ax.set_title(label, fontsize=13)
        ax.set_xlabel("SLM pixel x"); ax.set_ylabel("SLM pixel y")
        plt.colorbar(im, ax=ax, label='Phase (rad)', fraction=0.046, pad=0.04)
    plt.tight_layout()
    fig.savefig(FIGURES / "fig1_lg_phase_masks.png", dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("  fig1 saved.")

    # ── Figure 2: focal-plane intensity ──────────────────────────────────────
    fig, axes = plt.subplots(2, 3, figsize=(13, 8))
    fig.suptitle(
        "LG Beam Focal-Plane Intensity  (log scale)\n"
        "Grating shifts beam off DC spot",
        fontsize=12, fontweight='bold'
    )
    for ax, (l, p, label) in zip(axes.flat, cases):
        phi_slm, _ = lg_phase_mask(l, p, W0, Mx, My, grating_period=GRATING)
        ff = focal_field(phi_slm, laser_amp)
        intensity = np.abs(ff)**2
        log_int   = np.log1p(intensity / intensity.max() * 1e4)
        # find beam centre (peak)
        peak = np.unravel_index(np.argmax(intensity), intensity.shape)
        cx_p, cy_p = peak
        crop = 80
        sl = np.s_[max(0,cx_p-crop):cx_p+crop, max(0,cy_p-crop):cy_p+crop]
        ax.imshow(log_int[sl].T, cmap='inferno', origin='lower')
        ax.set_title(label, fontsize=13)
        ax.set_xlabel("Focal pixel x"); ax.set_ylabel("Focal pixel y")
    plt.tight_layout()
    fig.savefig(FIGURES / "fig2_lg_focal_intensity.png", dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("  fig2 saved.")

    # ── Figure 3: phase winding — l=+1 vs l=-1 ───────────────────────────────
    # Demonstrates OAM sign: phase increases CW vs CCW
    fig, axes = plt.subplots(2, 2, figsize=(10, 9))
    fig.suptitle(
        r"Phase Winding Direction: $l=+2$ vs $l=-2$  (OAM sign)"
        "\n(central 400×400 crop, no grating)",
        fontsize=12, fontweight='bold'
    )
    for col, l in enumerate([2, -2]):
        phi_slm, phi_lg = lg_phase_mask(l, 0, W0, Mx, My, grating_period=None)
        ff    = focal_field(phi_slm, laser_amp)
        intens = np.abs(ff)**2
        log_i  = np.log1p(intens / intens.max() * 1e4)
        crop = 200
        cx, cy = Mx // 2, My // 2
        sl = np.s_[cx-crop:cx+crop, cy-crop:cy+crop]

        im0 = axes[0, col].imshow(phi_lg[sl].T % (2*np.pi), cmap='hsv',
                                   vmin=0, vmax=2*np.pi, origin='lower')
        axes[0, col].set_title(f"Phase mask  $l={l:+d}$", fontsize=11)
        plt.colorbar(im0, ax=axes[0, col], label='Phase (rad)',
                     fraction=0.046, pad=0.04)

        axes[1, col].imshow(log_i[sl].T, cmap='inferno', origin='lower')
        axes[1, col].set_title(f"Focal intensity  $l={l:+d}$", fontsize=11)

    for ax in axes.flat:
        ax.set_xlabel("pixel x"); ax.set_ylabel("pixel y")
    plt.tight_layout()
    fig.savefig(FIGURES / "fig3_oam_sign.png", dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("  fig3 saved.")

    # ── Figure 4: radial structure — increasing p ─────────────────────────────
    fig, axes = plt.subplots(2, 3, figsize=(13, 8))
    fig.suptitle(
        r"Radial Structure: Fixed $l=2$, Varying $p$"
        "\nEach ring = one additional radial node from $L_p^{|l|}$",
        fontsize=12, fontweight='bold'
    )
    p_vals = [0, 1, 2, 3, 4, 5]
    for ax, p in zip(axes.flat, p_vals):
        phi_slm, _ = lg_phase_mask(2, p, W0, Mx, My, grating_period=GRATING)
        ff     = focal_field(phi_slm, laser_amp)
        intens = np.abs(ff)**2
        log_i  = np.log1p(intens / intens.max() * 1e4)
        peak   = np.unravel_index(np.argmax(intens), intens.shape)
        cx_p, cy_p = peak
        crop = 120
        sl = np.s_[max(0,cx_p-crop):cx_p+crop, max(0,cy_p-crop):cy_p+crop]
        ax.imshow(log_i[sl].T, cmap='inferno', origin='lower')
        ax.set_title(f"$p={p}$  ({p+1} ring{'s' if p>0 else ''})", fontsize=11)
        ax.set_xlabel("Focal pixel x"); ax.set_ylabel("Focal pixel y")
    plt.tight_layout()
    fig.savefig(FIGURES / "fig4_radial_structure.png", dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("  fig4 saved.")

    print(f"\nFigures saved to {FIGURES}")


if __name__ == "__main__":
    run_and_plot()
