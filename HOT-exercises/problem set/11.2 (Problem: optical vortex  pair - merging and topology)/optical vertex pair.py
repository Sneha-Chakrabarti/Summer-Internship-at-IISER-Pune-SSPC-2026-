"""
Problem 11.2
Phase and intensity structure of two optical vortices within a Gaussian profile.
Study the evolution as vortex positions approach the optical axis until merging.
Investigate opposite-charge vortex pairs.

Reference: Jones, Maragò & Volpe, Optical Tweezers: Principles and Applications, Ch. 7
Notes: SSPC 2026, Sneha Chakrabarti

Physical setup
--------------
Two LG_0^l vortices embedded in a Gaussian beam, placed at positions
(+d, 0) and (-d, 0) relative to the beam axis. The combined field is:

    E(x, y) = E_G(x, y) * phi_1(x, y) * phi_2(x, y)

where phi_n is the vortex phase factor for vortex n:
    phi_n(x, y) = exp( i * l_n * arctan2(y - y_n, x - x_n) )

This is the direct superposition of vortex phase factors on a single Gaussian.
(Not an SLM hologram — this models the field directly.)

Topological charge (vorticity)
-------------------------------
The topological charge of a vortex at (x_0, y_0) is:
    q = (1/2pi) * closed_line_integral grad(phi) . dl
      = l   for a phase singularity with winding number l

For two same-sign vortices (+l, +l): total charge Q = 2l.
As they merge, the phase singularities combine into a single charge-2l vortex.

For two opposite-sign vortices (+l, -l): total charge Q = 0.
As they approach and merge: the singularities annihilate (charge cancellation).

Detection of vortex positions
------------------------------
Vortex cores are located where |E| = 0 (intensity zeros) and the phase
winds by ±2pi*l around the zero. We detect them numerically via:
  1. Intensity minima below a threshold.
  2. Phase winding number computed on a small contour around each minimum.
     Winding = (1/2pi) * sum of phase differences along the contour (mod 2pi).

Study
-----
Part A: Same-sign vortices (+1, +1)
  - Vary separation d from large (well-separated) to 0 (merged).
  - Track: intensity profile, phase map, vortex positions, total charge.

Part B: Opposite-sign vortices (+1, -1)
  - Same d sweep.
  - Show vortex annihilation and charge cancellation.

Does total vorticity change as vortices move toward higher intensity region?
  - Answer: No — topological charge is conserved as long as vortices do not
    leave the beam or annihilate with opposite-charge partners.
    Same-sign: Q = 2 always until merged (then Q = 2 still, one vortex).
    Opposite-sign: Q = 0 always. On merging, both singularities vanish
    and the field becomes locally Gaussian (no phase singularity).

Parameters
----------
Gaussian beam waist: w0 = 150 pixels
Grid: 600 x 600 pixels (centred)
Vortex charge: l = 1
Separation sweep: d in {300, 200, 120, 60, 20, 5, 0} pixels
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from pathlib import Path
import sys

rng = np.random.default_rng(42)

Mx, My = 600, 600
W0     = 150      # Gaussian waist in pixels
L      = 1        # topological charge of each vortex
D_VALS = [280, 180, 100, 50, 20, 5, 0]   # separations to study

FIGURES = Path(__file__).parent / "figures"
FIGURES.mkdir(exist_ok=True)


# ── field construction ────────────────────────────────────────────────────────

def gaussian_field(Mx, My, w0):
    """Gaussian beam amplitude profile E_G(x,y) = exp(-r^2/w0^2)."""
    cx, cy = Mx / 2.0, My / 2.0
    x = np.arange(Mx) - cx
    y = np.arange(My) - cy
    X, Y = np.meshgrid(x, y, indexing='ij')
    return np.exp(-(X**2 + Y**2) / w0**2)


def vortex_phase(Mx, My, x0, y0, l):
    """
    Phase factor exp(i*l*arctan2(y-y0, x-x0)) for vortex at (x0, y0).
    x0, y0 are pixel offsets from beam centre.
    """
    cx, cy = Mx / 2.0, My / 2.0
    x = np.arange(Mx) - cx
    y = np.arange(My) - cy
    X, Y = np.meshgrid(x, y, indexing='ij')
    phi = l * np.arctan2(Y - y0, X - x0)
    return np.exp(1j * phi)


def two_vortex_field(Mx, My, w0, d, l1, l2):
    """
    Combined field: Gaussian * vortex at (+d,0) * vortex at (-d,0).
    Returns complex field (Mx, My).
    """
    E_G   = gaussian_field(Mx, My, w0)
    phi1  = vortex_phase(Mx, My,  d, 0, l1)
    phi2  = vortex_phase(Mx, My, -d, 0, l2)
    return E_G * phi1 * phi2


# ── vortex detection via phase winding ───────────────────────────────────────

def phase_winding(phase_field, px, py, radius=3):
    """
    Compute winding number around pixel (px, py) on a square contour of
    given radius. Returns winding number (integer multiple of 1).
    """
    pts = []
    r   = radius
    # top edge
    for x in range(px - r, px + r + 1): pts.append((x, py - r))
    # right edge
    for y in range(py - r, py + r + 1): pts.append((px + r, y))
    # bottom edge (reversed)
    for x in range(px + r, px - r - 1, -1): pts.append((x, py + r))
    # left edge (reversed)
    for y in range(py + r, py - r - 1, -1): pts.append((px - r, y))

    total_dphi = 0.0
    prev_phi   = phase_field[
        np.clip(pts[0][0], 0, phase_field.shape[0]-1),
        np.clip(pts[0][1], 0, phase_field.shape[1]-1)
    ]
    for (xi, yi) in pts[1:]:
        xi = np.clip(xi, 0, phase_field.shape[0]-1)
        yi = np.clip(yi, 0, phase_field.shape[1]-1)
        curr_phi    = phase_field[xi, yi]
        dphi        = curr_phi - prev_phi
        # wrap to (-pi, pi)
        dphi        = (dphi + np.pi) % (2 * np.pi) - np.pi
        total_dphi += dphi
        prev_phi    = curr_phi

    return round(total_dphi / (2 * np.pi))


def find_vortices(E_field, intensity_threshold=0.005, step=4):
    """
    Find vortex positions by scanning for non-zero phase winding on a grid.
    Uses a coarser step than pixel-by-pixel for speed.

    Returns list of (x, y, winding_number).
    """
    phase  = np.angle(E_field)
    intens = np.abs(E_field)**2
    I_max  = intens.max()
    Mx, My = phase.shape
    vortices = []
    seen     = set()

    for px in range(5, Mx - 5, step):
        for py in range(5, My - 5, step):
            # skip very low intensity (true zeros from Gaussian envelope at edge)
            if intens[px, py] < intensity_threshold * I_max:
                continue
            w = phase_winding(phase, px, py, radius=3)
            if w != 0:
                # avoid double-counting nearby detections
                key = (px // (step * 2), py // (step * 2))
                if key not in seen:
                    seen.add(key)
                    vortices.append((px, py, w))

    return vortices


# ── run and plot ──────────────────────────────────────────────────────────────

def run_and_plot():
    cx, cy = Mx // 2, My // 2
    print(f"Grid: {Mx}×{My}, w0={W0} px, l={L}")

    # ── Part A: same-sign vortices (+l, +l) ──────────────────────────────────
    print("\nPart A: same-sign vortices (+l, +l)")

    n_d  = len(D_VALS)
    ncols = n_d
    crop  = 220

    # ── Figure 1: intensity evolution — same sign ─────────────────────────────
    fig, axes = plt.subplots(2, n_d, figsize=(3 * n_d, 6))
    fig.suptitle(
        r"Same-Sign Vortex Pair ($l_1=+1$, $l_2=+1$): Evolution as $d \to 0$"
        "\nTop: intensity  |  Bottom: phase",
        fontsize=12, fontweight='bold'
    )
    sl = np.s_[cx-crop:cx+crop, cy-crop:cy+crop]

    for col, d in enumerate(D_VALS):
        E   = two_vortex_field(Mx, My, W0, d, L, L)
        I   = np.abs(E)**2
        phi = np.angle(E)
        vortices = find_vortices(E)

        axes[0, col].imshow(I[sl].T / I[sl].max(), cmap='inferno',
                            origin='lower', vmin=0, vmax=1)
        # mark vortex positions
        for (vx, vy, w) in vortices:
            if (cx-crop <= vx < cx+crop) and (cy-crop <= vy < cy+crop):
                axes[0, col].plot(vx-(cx-crop), vy-(cy-crop),
                                  'c+' if w > 0 else 'r+', ms=10, mew=2)
        axes[0, col].set_title(f"$d={d}$", fontsize=10)
        axes[0, col].axis('off')

        axes[1, col].imshow(phi[sl].T, cmap='hsv', vmin=-np.pi, vmax=np.pi,
                            origin='lower')
        axes[1, col].axis('off')

        Q_total = sum(w for _, _, w in vortices)
        print(f"  d={d:4d}: {len(vortices)} vortices, Q_total={Q_total}")

    plt.tight_layout()
    fig.savefig(FIGURES / "fig1_same_sign_evolution.png", dpi=150,
                bbox_inches='tight')
    plt.close(fig)
    print("  fig1 saved.")

    # ── Part B: opposite-sign vortices (+l, -l) ───────────────────────────────
    print("\nPart B: opposite-sign vortices (+l, -l)")

    fig, axes = plt.subplots(2, n_d, figsize=(3 * n_d, 6))
    fig.suptitle(
        r"Opposite-Sign Vortex Pair ($l_1=+1$, $l_2=-1$): Evolution as $d \to 0$"
        "\nTop: intensity  |  Bottom: phase  |  Charge annihilation on merging",
        fontsize=12, fontweight='bold'
    )
    for col, d in enumerate(D_VALS):
        E   = two_vortex_field(Mx, My, W0, d, L, -L)
        I   = np.abs(E)**2
        phi = np.angle(E)
        vortices = find_vortices(E)

        axes[0, col].imshow(I[sl].T / I[sl].max(), cmap='inferno',
                            origin='lower', vmin=0, vmax=1)
        for (vx, vy, w) in vortices:
            if (cx-crop <= vx < cx+crop) and (cy-crop <= vy < cy+crop):
                axes[0, col].plot(vx-(cx-crop), vy-(cy-crop),
                                  'c+' if w > 0 else 'r+', ms=10, mew=2)
        axes[0, col].set_title(f"$d={d}$", fontsize=10)
        axes[0, col].axis('off')

        axes[1, col].imshow(phi[sl].T, cmap='hsv', vmin=-np.pi, vmax=np.pi,
                            origin='lower')
        axes[1, col].axis('off')

        Q_total = sum(w for _, _, w in vortices)
        print(f"  d={d:4d}: {len(vortices)} vortices, Q_total={Q_total}")

    plt.tight_layout()
    fig.savefig(FIGURES / "fig2_opposite_sign_evolution.png", dpi=150,
                bbox_inches='tight')
    plt.close(fig)
    print("  fig2 saved.")

    # ── Figure 3: detailed view — 3 key separations, both sign cases ──────────
    d_select = [180, 50, 0]
    fig, axes = plt.subplots(4, 3, figsize=(10, 12))
    fig.suptitle(
        "Detailed Comparison: Same-Sign vs Opposite-Sign Vortex Pairs\n"
        r"Rows: $+l,+l$ intensity | $+l,+l$ phase | $+l,-l$ intensity | $+l,-l$ phase",
        fontsize=11, fontweight='bold'
    )
    crop2 = 180
    sl2   = np.s_[cx-crop2:cx+crop2, cy-crop2:cy+crop2]

    for col, d in enumerate(d_select):
        E_ss = two_vortex_field(Mx, My, W0, d,  L,  L)
        E_os = two_vortex_field(Mx, My, W0, d,  L, -L)
        I_ss = np.abs(E_ss[sl2])**2
        I_os = np.abs(E_os[sl2])**2

        axes[0, col].imshow(I_ss.T / I_ss.max(),
                            cmap='inferno', origin='lower', vmin=0, vmax=1)
        axes[0, col].set_title(f"$+l,+l$  $d={d}$", fontsize=10)

        axes[1, col].imshow(np.angle(E_ss[sl2]).T, cmap='hsv',
                            vmin=-np.pi, vmax=np.pi, origin='lower')

        axes[2, col].imshow(I_os.T / I_os.max(),
                            cmap='inferno', origin='lower', vmin=0, vmax=1)
        axes[2, col].set_title(f"$+l,-l$  $d={d}$", fontsize=10)

        axes[3, col].imshow(np.angle(E_os[sl2]).T, cmap='hsv',
                            vmin=-np.pi, vmax=np.pi, origin='lower')

    for ax in axes.flat:
        ax.axis('off')
    row_labels = [r"$+l,+l$ intensity", r"$+l,+l$ phase",
                  r"$+l,-l$ intensity", r"$+l,-l$ phase"]
    for row, label in enumerate(row_labels):
        axes[row, 0].set_ylabel(label, fontsize=9)

    plt.tight_layout()
    fig.savefig(FIGURES / "fig3_detailed_comparison.png", dpi=150,
                bbox_inches='tight')
    plt.close(fig)
    print("  fig3 saved.")

    # ── Figure 4: radial intensity profiles at key separations ───────────────
    # Cross-section through beam centre showing intensity zeros moving
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(
        "Radial Intensity Cross-Section Through Beam Centre\n"
        r"Showing intensity zeros (vortex cores) moving as $d \to 0$",
        fontsize=12, fontweight='bold'
    )
    d_plot = [280, 180, 100, 50, 20, 0]
    colors = plt.cm.plasma(np.linspace(0.1, 0.9, len(d_plot)))

    for ax, (l1, l2), title in [
        (axes[0], ( L,  L), r"Same sign ($+l, +l$)"),
        (axes[1], ( L, -L), r"Opposite sign ($+l, -l$)"),
    ]:
        for d, col in zip(d_plot, colors):
            E   = two_vortex_field(Mx, My, W0, d, l1, l2)
            I   = np.abs(E)**2
            row = I[:, cy]   # cross-section along x through centre
            xs  = np.arange(Mx) - cx
            ax.plot(xs, row / row.max(), lw=1.5, color=col,
                    label=f"$d={d}$", alpha=0.85)
        ax.set_xlabel("x (pixels from axis)", fontsize=11)
        ax.set_ylabel("Normalised intensity", fontsize=11)
        ax.set_title(title, fontsize=11)
        ax.set_xlim(-crop, crop)
        ax.legend(fontsize=8, ncol=2)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(FIGURES / "fig4_radial_profiles.png", dpi=150,
                bbox_inches='tight')
    plt.close(fig)
    print("  fig4 saved.")

    # ── Figure 5: total charge vs separation ─────────────────────────────────
    charges_ss, charges_os = [], []
    for d in D_VALS:
        v_ss = find_vortices(two_vortex_field(Mx, My, W0, d,  L,  L))
        v_os = find_vortices(two_vortex_field(Mx, My, W0, d,  L, -L))
        charges_ss.append(sum(w for _, _, w in v_ss))
        charges_os.append(sum(w for _, _, w in v_os))

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(D_VALS, charges_ss, 'o-', lw=2, color='steelblue',
            label=r"Same sign ($+l,+l$): $Q=+2$")
    ax.plot(D_VALS, charges_os, 's--', lw=2, color='tomato',
            label=r"Opposite sign ($+l,-l$): $Q=0$")
    ax.axhline(2, color='steelblue', ls=':', lw=1, alpha=0.5)
    ax.axhline(0, color='tomato',    ls=':', lw=1, alpha=0.5)
    ax.set_xlabel(r"Vortex separation $d$ (pixels)", fontsize=11)
    ax.set_ylabel("Total topological charge $Q$", fontsize=11)
    ax.set_title(
        "Total Topological Charge vs Separation\n"
        "Charge is conserved: same-sign Q=2 always, opposite-sign Q=0 always",
        fontsize=11
    )
    ax.set_ylim(-1, 3)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.invert_xaxis()   # show merging direction (d decreasing) left-to-right
    ax.set_xlabel(r"$d$ (pixels) — decreasing: vortices approach axis →",
                  fontsize=10)
    plt.tight_layout()
    fig.savefig(FIGURES / "fig5_total_charge.png", dpi=150,
                bbox_inches='tight')
    plt.close(fig)
    print("  fig5 saved.")

    print(f"\nFigures saved to {FIGURES}")


if __name__ == "__main__":
    run_and_plot()
