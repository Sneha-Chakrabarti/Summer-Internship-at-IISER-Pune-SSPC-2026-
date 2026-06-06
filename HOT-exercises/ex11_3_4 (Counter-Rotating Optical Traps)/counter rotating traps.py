"""
Exercise 11.3.4
How can multiple Laguerre-Gaussian beams be generated to produce
counter-rotating optical traps?

Reference: Jones, Maragò & Volpe, Optical Tweezers: Principles and Applications, Ch. 7
Notes: SSPC 2026, Sneha Chakrabarti

Theory
------
A single LG beam with topological charge l exerts orbital angular momentum (OAM)
of l*hbar per photon on a trapped particle, causing it to orbit the ring trap in
the direction set by sign(l). Two LG beams with charges +l and -l focused to
different transverse positions create a pair of traps exerting opposite OAM:
the particles orbit in opposite directions — counter-rotating traps.

Hologram for multiple LG beams
--------------------------------
Each LG beam at focal position (x_n, y_n) requires a phase mask:

    phi_n(x, y) = l_n * phi  +  pi * Theta{L_p^|l|(2*rho^2/w0^2)}
                  + (2*pi/lambda*f) * (x_slm * x_n + y_slm * y_n)
                                   ^^^ blazed grating: lateral steering

The multi-beam hologram superimposes N such fields using the Superposition of
Gratings and Lenses (SGL) algorithm (theory notes, Section 5.2):

    phi_SGL = arg( sum_n exp(i * phi_n) )

This gives a single SLM mask that simultaneously generates all N LG beams,
each at its designated position and with its designated OAM.

Counter-rotating trap configurations studied
--------------------------------------------
1. Pair: (+l, -l) — two beams, opposite charges, symmetric lateral displacement
2. Quad: (+l, -l, +l, -l) — four beams, charges alternating, 2×2 array
3. Varying |l|: (+1, -1), (+2, -2), (+3, -3) — effect of |l| on ring radius
4. Mixed p: (l=2,p=0) vs (l=2,p=1) — radial index effect on trap structure

Implementation
--------------
SLM  : 1000 × 1000 pixels
Laser: Gaussian, 1/e^2 radius w_0 = 300 px
Trap separation: 150 pixels in focal plane
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import genlaguerre
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "ex11_2_11"))
from gs_algorithm import gaussian_laser

sys.path.insert(0, str(Path(__file__).parent.parent / "ex11_3_1"))
from lg_beams import lg_phase_mask, polar_grid

# ── parameters ────────────────────────────────────────────────────────────────
Mx, My      = 1000, 1000
W0          = 300          # beam waist in pixels
SEPARATION  = 150          # trap separation in focal-plane pixels
GRATING     = 30           # base grating period (sets scale of separation)

FIGURES = Path(__file__).parent / "figures"
FIGURES.mkdir(exist_ok=True)


# ── single LG field at a given focal position ─────────────────────────────────

def lg_field_at(l, p, w0, Mx, My, x_trap, y_trap, grating_period):
    """
    Phase for a single LG_p^l beam steered to focal position (x_trap, y_trap).

    phi = l*phi_polar  +  pi*Theta{L_p^|l|}
          + 2*pi * (x_slm * x_trap / (grating_period * Mx)
                  + y_slm * y_trap / (grating_period * My))

    The steering term is parameterised so that x_trap, y_trap are in focal
    pixels and grating_period sets the pixel-to-focal-pixel conversion.

    In practice: a grating of period Lambda shifts the first-order spot by
    f * lambda / Lambda pixels in the focal plane. Here we absorb the
    lambda*f/Lambda factor into the coordinate scaling.
    """
    R, Phi, X, Y = polar_grid(Mx, My)
    cx, cy = Mx / 2.0, My / 2.0

    # LG phase
    u      = 2 * R**2 / w0**2
    Lp     = genlaguerre(p, abs(l))(u)
    phi_lg = l * Phi + np.pi * (Lp < 0).astype(float)

    # steering: shift beam to (x_trap, y_trap) in focal-plane pixels
    # steering phase = 2*pi * (m_x * nx + m_y * ny) / M  ->  nx,ny in pixels
    phi_steer = 2 * np.pi * (X * x_trap / Mx + Y * y_trap / My)

    return (phi_lg + phi_steer) % (2 * np.pi)


# ── multi-beam SGL hologram ───────────────────────────────────────────────────

def multi_lg_hologram(beam_specs, w0, Mx, My, grating_period):
    """
    SGL hologram for multiple LG beams.

    beam_specs: list of (l, p, x_trap, y_trap)

    phi_SGL = arg( sum_n  exp(i * phi_n) )
    """
    field_sum = np.zeros((Mx, My), dtype=complex)
    for l, p, x_t, y_t in beam_specs:
        phi_n = lg_field_at(l, p, w0, Mx, My, x_t, y_t, grating_period)
        field_sum += np.exp(1j * phi_n)
    return np.angle(field_sum)   # in (-pi, pi)


def focal_intensity(slm_phase, laser_amp):
    sf = laser_amp * np.exp(1j * slm_phase)
    ff = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(sf)))
    return np.abs(ff)**2, ff


# ── run and plot ──────────────────────────────────────────────────────────────

def run_and_plot():
    print(f"Setting up {Mx}×{My} SLM, w0={W0} px...")
    laser_amp = gaussian_laser(Mx, My, W0)

    sep = SEPARATION
    gp  = GRATING

    # ── Figure 1: pair (+l, -l) for l = 1, 2, 3 ─────────────────────────────
    l_vals = [1, 2, 3]
    fig, axes = plt.subplots(2, 3, figsize=(13, 8))
    fig.suptitle(
        r"Counter-Rotating Trap Pairs: $+l$ and $-l$  (SGL hologram)"
        "\nFocal-plane intensity — each pair rotates in opposite directions",
        fontsize=12, fontweight='bold'
    )
    for col, l in enumerate(l_vals):
        specs = [
            ( l, 0, -sep//2,  0),
            (-l, 0,  sep//2,  0),
        ]
        phi_slm = multi_lg_hologram(specs, W0, Mx, My, gp)
        intens, _ = focal_intensity(phi_slm, laser_amp)

        # phase mask (top row)
        crop_slm = 300
        cx, cy   = Mx // 2, My // 2
        sl_slm   = np.s_[cx-crop_slm:cx+crop_slm, cy-crop_slm:cy+crop_slm]
        im = axes[0, col].imshow(phi_slm[sl_slm].T % (2*np.pi), cmap='hsv',
                                  vmin=0, vmax=2*np.pi, origin='lower')
        axes[0, col].set_title(f"SLM mask  $l=±{l}$", fontsize=11)
        plt.colorbar(im, ax=axes[0, col], fraction=0.046, pad=0.04,
                     label='Phase (rad)')

        # focal intensity (bottom row)
        log_i  = np.log1p(intens / intens.max() * 1e4)
        # crop around the two beams
        crop_f = 200
        sl_f   = np.s_[cx-crop_f:cx+crop_f, cy-crop_f:cy+crop_f]
        axes[1, col].imshow(log_i[sl_f].T, cmap='inferno', origin='lower')
        axes[1, col].set_title(
            f"Focal intensity  $l=+{l}$ (left) / $l=-{l}$ (right)",
            fontsize=10
        )
        for ax in [axes[0, col], axes[1, col]]:
            ax.set_xlabel("pixel x"); ax.set_ylabel("pixel y")

    plt.tight_layout()
    fig.savefig(FIGURES / "fig1_counter_rotating_pairs.png", dpi=150,
                bbox_inches='tight')
    plt.close(fig)
    print("  fig1 saved.")

    # ── Figure 2: 2×2 quad array of counter-rotating traps ───────────────────
    l_quad = 2
    specs_quad = [
        ( l_quad, 0, -sep//2, -sep//2),
        (-l_quad, 0,  sep//2, -sep//2),
        (-l_quad, 0, -sep//2,  sep//2),
        ( l_quad, 0,  sep//2,  sep//2),
    ]
    phi_quad = multi_lg_hologram(specs_quad, W0, Mx, My, gp)
    intens_quad, _ = focal_intensity(phi_quad, laser_amp)
    log_quad = np.log1p(intens_quad / intens_quad.max() * 1e4)

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    fig.suptitle(
        r"2×2 Counter-Rotating Quad Array  ($l = ±2$, alternating)"
        "\nOAM alternates: top-left/bottom-right = $+l$, "
        "top-right/bottom-left = $-l$",
        fontsize=12, fontweight='bold'
    )
    cx, cy    = Mx // 2, My // 2
    crop_slm  = 300
    crop_f    = 250
    sl_slm    = np.s_[cx-crop_slm:cx+crop_slm, cy-crop_slm:cy+crop_slm]
    sl_f      = np.s_[cx-crop_f:cx+crop_f, cy-crop_f:cy+crop_f]

    im = axes[0].imshow(phi_quad[sl_slm].T % (2*np.pi), cmap='hsv',
                         vmin=0, vmax=2*np.pi, origin='lower')
    axes[0].set_title("SLM hologram phase", fontsize=11)
    plt.colorbar(im, ax=axes[0], label='Phase (rad)', fraction=0.046, pad=0.04)

    axes[1].imshow(log_quad[sl_f].T, cmap='inferno', origin='lower')
    axes[1].set_title("Focal-plane intensity (log scale)\n"
                      "Four doughnut traps, alternating OAM sign", fontsize=11)
    for ax in axes:
        ax.set_xlabel("pixel x"); ax.set_ylabel("pixel y")

    plt.tight_layout()
    fig.savefig(FIGURES / "fig2_quad_array.png", dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("  fig2 saved.")

    # ── Figure 3: effect of |l| on ring radius ────────────────────────────────
    # Single LG trap (no SGL): show how ring radius scales with |l|
    fig, axes = plt.subplots(1, 4, figsize=(14, 4))
    fig.suptitle(
        r"Effect of $|l|$ on Doughnut Ring Radius"
        "\n(single LG beam, $p=0$; ring radius $\propto \sqrt{|l|}$)",
        fontsize=12, fontweight='bold'
    )
    for ax, l in enumerate([1, 2, 3, 4]):
        phi_single, _ = lg_phase_mask(l, 0, W0, Mx, My, grating_period=gp)
        intens_s, _   = focal_intensity(phi_single, laser_amp)
        log_s = np.log1p(intens_s / intens_s.max() * 1e4)
        peak  = np.unravel_index(np.argmax(intens_s), intens_s.shape)
        cpx, cpy = peak
        cr = 100
        sl = np.s_[max(0,cpx-cr):cpx+cr, max(0,cpy-cr):cpy+cr]
        axes[ax].imshow(log_s[sl].T, cmap='inferno', origin='lower')
        axes[ax].set_title(f"$l={l}$", fontsize=13)
        axes[ax].set_xlabel("pixel x"); axes[ax].set_ylabel("pixel y")
    plt.tight_layout()
    fig.savefig(FIGURES / "fig3_ring_radius_vs_l.png", dpi=150,
                bbox_inches='tight')
    plt.close(fig)
    print("  fig3 saved.")

    # ── Figure 4: OAM phase winding in the focal field ────────────────────────
    # Show focal-plane phase (angle of complex field) for +l and -l
    # Winding direction directly visualises OAM sign
    fig, axes = plt.subplots(2, 3, figsize=(13, 8))
    fig.suptitle(
        "Focal-Field Phase: Opposite Winding for $±l$\n"
        "Phase winding direction = OAM sign; "
        r"$+l$ winds CCW, $-l$ winds CW",
        fontsize=12, fontweight='bold'
    )
    for col, l in enumerate([1, 2, 3]):
        for row, sign in enumerate([1, -1]):
            l_use = sign * l
            phi_slm, _ = lg_phase_mask(l_use, 0, W0, Mx, My,
                                        grating_period=gp)
            _, ff = focal_intensity(phi_slm, laser_amp)
            focal_phase = np.angle(ff)
            intens_mask = np.abs(ff)**2
            # mask phase where intensity is very low (noisy)
            threshold   = intens_mask.max() * 0.01
            focal_phase_masked = np.where(intens_mask > threshold,
                                          focal_phase, np.nan)

            peak = np.unravel_index(np.argmax(intens_mask), intens_mask.shape)
            cpx, cpy = peak
            cr = 70
            sl = np.s_[max(0,cpx-cr):cpx+cr, max(0,cpy-cr):cpy+cr]

            im = axes[row, col].imshow(focal_phase_masked[sl].T, cmap='hsv',
                                        vmin=-np.pi, vmax=np.pi, origin='lower')
            axes[row, col].set_title(f"$l={l_use:+d}$", fontsize=13)
            axes[row, col].set_xlabel("pixel x")
            axes[row, col].set_ylabel("pixel y")
            plt.colorbar(im, ax=axes[row, col], label='Phase (rad)',
                         fraction=0.046, pad=0.04)

    plt.tight_layout()
    fig.savefig(FIGURES / "fig4_focal_phase_winding.png", dpi=150,
                bbox_inches='tight')
    plt.close(fig)
    print("  fig4 saved.")

    print(f"\nFigures saved to {FIGURES}")


if __name__ == "__main__":
    run_and_plot()
