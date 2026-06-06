"""
Exercise 11.2.17
Aberration correction using Zernike polynomials added to the SLM hologram.

Reference: Jones, Maragò & Volpe, Optical Tweezers: Principles and Applications, Ch. 7
Notes: SSPC 2026, Sneha Chakrabarti

Theory
------
Any smooth aberration phase Psi(x, y) across the SLM pupil can be expanded in
the Zernike polynomial basis {Z_j}, which is orthogonal over the unit disk:

    Psi(x, y) = sum_j  c_j * Z_j(rho, phi)

To correct the aberration, the conjugate phase (-Psi) is added to the SLM
hologram. Since the SLM is phase-only and has a finite range [0, 2pi), the
correction is applied modulo 2pi:

    phi_corrected(x, y) = [ phi_hologram(x, y) + phi_correction(x, y) ] mod 2pi

where phi_correction = -Psi (or an estimate of it).

Effect on focusing
------------------
Without correction: the field at the focal point acquires aberration-dependent
phase errors across the pupil, causing the plane-wave modes to arrive with
different phases. Peak intensity is reduced and the PSF is distorted.

With correction: the SLM pre-compensates the aberration. The focal field
reconstructs the diffraction-limited PSF, restoring peak intensity.

Practical measurement of Zernike coefficients
----------------------------------------------
In a real experiment, c_j are measured by:
  (a) Interferometric wavefront sensing (Shack-Hartmann, etc.)
  (b) Iterative optimisation of fluorescence/scattering signal (sensorless AO)
  (c) Direct pupil segmentation (Cizmar 2010, as in Ex 11.2.16)

Here we demonstrate with known Zernike coefficients (simulated aberration) and
measure the improvement in focus quality for each Zernike mode individually and
for the full correction.

Zernike polynomials (Noll indexing convention)
----------------------------------------------
j  (n, m)   Name
1  (0,  0)  Piston          -- not correctable (global phase)
2  (1,  1)  Tip             -- image shift
3  (1, -1)  Tilt            -- image shift
4  (2,  0)  Defocus
5  (2, -2)  Oblique astig.
6  (2,  2)  Vertical astig.
7  (3, -1)  Vertical coma
8  (3,  1)  Horizontal coma
9  (3, -3)  Vertical trefoil
10 (3,  3)  Oblique trefoil
11 (4,  0)  Primary spherical

Standard benchmark
------------------
SLM  : 1000 × 1000 pixels
Laser: Gaussian, 1/e2 radius = 400 px
Target: single focus at (Mx//2, My//2), z=0
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "ex11_2_11"))
from gs_algorithm import gaussian_laser

rng = np.random.default_rng(42)

Mx, My  = 1000, 1000
LASER_W = 400

FIGURES = Path(__file__).parent / "figures"
FIGURES.mkdir(exist_ok=True)


# ── Zernike polynomial evaluation ─────────────────────────────────────────────

def zernike_map(n, m, Mx, My):
    """
    Evaluate Zernike polynomial Z_n^m on the SLM pupil (unit disk).
    Returns (Mx, My) array; zero outside unit disk.
    """
    cx, cy = Mx / 2.0, My / 2.0
    x = (np.arange(Mx) - cx) / cx
    y = (np.arange(My) - cy) / cy
    X, Y = np.meshgrid(x, y, indexing='ij')
    R    = np.sqrt(X**2 + Y**2)
    Phi  = np.arctan2(Y, X)
    pupil = (R <= 1.0).astype(float)

    # radial polynomial
    s_max = (n - abs(m)) // 2
    Rnm   = np.zeros_like(R)
    import math
    for s in range(s_max + 1):
        coef = ((-1)**s * math.factorial(n - s) /
                (math.factorial(s) *
                 math.factorial((n + abs(m)) // 2 - s) *
                 math.factorial((n - abs(m)) // 2 - s)))
        Rnm += coef * R**(n - 2 * s)

    # azimuthal part
    norm = np.sqrt(2 * (n + 1)) if m != 0 else np.sqrt(n + 1)
    if m > 0:
        Z = norm * Rnm * np.cos(m * Phi)
    elif m < 0:
        Z = norm * Rnm * np.sin(abs(m) * Phi)
    else:
        Z = norm * Rnm

    return Z * pupil


def build_aberration(Mx, My, coeffs):
    """
    Build aberration phase screen as weighted sum of normalised Zernike maps.
    coeffs: dict {(n, m): amplitude_rad} using zernike_map normalisation.
    Ensures correction = -sum c_j Z_j exactly cancels aberration = sum c_j Z_j.
    """
    phase = np.zeros((Mx, My))
    for (n, m), c in coeffs.items():
        phase += c * zernike_map(n, m, Mx, My)
    return phase


# ── focus quality metrics ──────────────────────────────────────────────────────

def focus_metrics(slm_phase, laser_amp, aberration_phase, target_px, target_py,
                  I_diffraction_limit=None):
    """
    Compute focal intensity and Strehl ratio at target point.
    Strehl = I(aberrated + corrected hologram) / I(flat phase, no aberration).

    I_diffraction_limit: pre-computed ideal intensity. If None it is computed
    from a flat (phi=0) hologram with no aberration, which is the true
    diffraction-limited reference for a Gaussian beam focused to centre.
    """
    sf = laser_amp * np.exp(1j * (slm_phase + aberration_phase))
    ff = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(sf)))
    I_focus = np.abs(ff[target_px, target_py])**2

    if I_diffraction_limit is None:
        sf_dl = laser_amp * np.exp(1j * np.zeros_like(slm_phase))
        ff_dl = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(sf_dl)))
        I_diffraction_limit = np.abs(ff_dl[target_px, target_py])**2

    strehl = I_focus / I_diffraction_limit if I_diffraction_limit > 0 else 0.0
    return I_focus, strehl, np.abs(ff)**2


# ── run and plot ──────────────────────────────────────────────────────────────

def run_and_plot():
    print(f"Setting up {Mx}×{My} SLM...")
    laser_amp = gaussian_laser(Mx, My, LASER_W)
    target_px, target_py = Mx // 2, My // 2

    # flat hologram for single trap at centre (phi=0 for centre target)
    phi_hologram = np.zeros((Mx, My))

    # ── define aberration: known Zernike composition ──────────────────────────
    # Moderate aberration with defocus, astigmatism, coma, spherical
    aberr_coeffs = {
        (2,  0): 1.2,   # defocus
        (2, -2): 0.8,   # oblique astigmatism
        (2,  2): -0.6,  # vertical astigmatism
        (3,  1): 0.5,   # horizontal coma
        (3, -1): -0.4,  # vertical coma
        (4,  0): 0.3,   # primary spherical
    }
    aberration = build_aberration(Mx, My, aberr_coeffs)
    print(f"  Aberration RMS = {np.std(aberration):.3f} rad")

    # ── baseline: no aberration ───────────────────────────────────────────────
    I_ideal, _, I_map_ideal = focus_metrics(
        phi_hologram, laser_amp, np.zeros((Mx, My)), target_px, target_py
    )
    # aberrated, uncorrected
    I_aberr, S_aberr, I_map_aberr = focus_metrics(
        phi_hologram, laser_amp, aberration, target_px, target_py,
        I_diffraction_limit=I_ideal
    )
    print(f"  I_ideal  = {I_ideal:.4f}")
    print(f"  I_aberr  = {I_aberr:.4f}  (Strehl = {S_aberr:.3f})")

    # ── Zernike terms: Noll indexing ──────────────────────────────────────────
    zernike_terms = [
        (4,  (2,  0),  "Defocus"),
        (5,  (2, -2),  "Oblique astig."),
        (6,  (2,  2),  "Vertical astig."),
        (7,  (3, -1),  "Vertical coma"),
        (8,  (3,  1),  "Horizontal coma"),
        (11, (4,  0),  "Primary spherical"),
    ]

    # ── Study 1: correct one Zernike term at a time ───────────────────────────
    print("\n  Correcting individual Zernike terms...")
    results_individual = []
    for j, (n, m), name in zernike_terms:
        if (n, m) not in aberr_coeffs:
            continue
        c = aberr_coeffs[(n, m)]
        Z = zernike_map(n, m, Mx, My)
        # correction = conjugate of this aberration term only
        phi_corr   = (-c * Z) % (2 * np.pi)
        phi_total  = (phi_hologram + phi_corr) % (2 * np.pi)
        I_c, S_c, _ = focus_metrics(phi_total, laser_amp, aberration,
                                     target_px, target_py,
                                     I_diffraction_limit=I_ideal)
        results_individual.append((name, j, c, I_c, S_c))
        print(f"    Z{j} ({name}, c={c:+.2f}): I={I_c:.4f}, Strehl={S_c:.3f}")

    # ── Study 2: progressively add correction terms ───────────────────────────
    print("\n  Progressive correction (adding terms one by one)...")
    # Build correction as running sum in raw radians, apply mod 2pi only for
    # the SLM phase. This avoids mod-2pi interaction between terms.
    correction_running = np.zeros((Mx, My))
    I_progressive   = [I_aberr]
    S_progressive   = [S_aberr]
    labels_prog     = ["No correction"]

    for j, (n, m), name in zernike_terms:
        if (n, m) not in aberr_coeffs:
            continue
        c = aberr_coeffs[(n, m)]
        Z = zernike_map(n, m, Mx, My)
        correction_running += (-c * Z)      # accumulate raw correction
        phi_total = (phi_hologram + correction_running) % (2 * np.pi)
        I_p, S_p, _ = focus_metrics(phi_total, laser_amp, aberration,
                                     target_px, target_py,
                                     I_diffraction_limit=I_ideal)
        I_progressive.append(I_p)
        S_progressive.append(S_p)
        labels_prog.append(f"+Z{j}\n({name})")
        print(f"    After adding Z{j}: I={I_p:.4f}, Strehl={S_p:.3f}")

    # full correction (all terms) — same as last progressive step
    I_full    = I_progressive[-1]
    S_full    = S_progressive[-1]
    phi_full  = (phi_hologram + correction_running) % (2 * np.pi)
    _, _, I_map_full = focus_metrics(phi_full, laser_amp, aberration,
                                     target_px, target_py,
                                     I_diffraction_limit=I_ideal)
    print(f"\n  Full correction: I={I_full:.4f}, Strehl={S_full:.3f}")

    # ── Study 3: effect of correction amplitude (over/under-correction) ───────
    print("\n  Scanning defocus correction amplitude...")
    Z_defocus    = zernike_map(2, 0, Mx, My)
    c_true       = aberr_coeffs[(2, 0)]
    c_scan       = np.linspace(-2.5, 2.5, 51)
    I_scan       = []
    for c_try in c_scan:
        phi_c = (phi_hologram - c_try * Z_defocus) % (2 * np.pi)
        I_c, _, _ = focus_metrics(phi_c, laser_amp, aberration,
                                   target_px, target_py,
                                   I_diffraction_limit=I_ideal)
        I_scan.append(I_c)

    # ── focal maps ────────────────────────────────────────────────────────────
    crop = 40
    cx, cy = Mx // 2, My // 2
    sl = np.s_[cx-crop:cx+crop, cy-crop:cy+crop]

    # ── Figure 1: focal maps — before and after full correction ───────────────
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    fig.suptitle(
        "Focal-Plane Intensity: Aberration Correction via Zernike Hologram\n"
        f"(1000×1000 SLM, target at centre, z=0)",
        fontsize=12, fontweight='bold'
    )
    vmax = I_map_ideal[sl].max()
    for ax, Im, t in [
        (axes[0], I_map_ideal,  f"Ideal (no aberr)\nI={I_ideal:.3f}"),
        (axes[1], I_map_aberr,  f"Aberrated (no corr)\nStrehl={S_aberr:.3f}"),
        (axes[2], I_map_full,   f"Full Zernike correction\nStrehl={S_full:.3f}"),
    ]:
        ax.imshow(Im[sl].T, cmap='inferno', origin='lower', vmin=0, vmax=vmax)
        ax.set_title(t, fontsize=10)
        ax.set_xlabel("x (px)"); ax.set_ylabel("y (px)")
    plt.tight_layout()
    fig.savefig(FIGURES / "fig1_focal_maps.png", dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("  fig1 saved.")

    # ── Figure 2: progressive correction — Strehl and intensity ───────────────
    x_prog = np.arange(len(I_progressive))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    fig.suptitle("Progressive Zernike Correction: Adding Terms One by One\n"
                 f"(1000×1000 SLM)", fontsize=12, fontweight='bold')

    for ax, vals, ylabel, ideal_val, color in [
        (axes[0], I_progressive, "$I$ at focus",     I_ideal, 'steelblue'),
        (axes[1], S_progressive, "Strehl ratio $S$", 1.0,     'seagreen'),
    ]:
        ax.plot(x_prog, vals, 'o-', color=color, lw=1.8, ms=7)
        ax.axhline(ideal_val, color='k', ls='--', lw=1.2, label='Ideal')
        ax.set_xticks(x_prog)
        ax.set_xticklabels(labels_prog, fontsize=8)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(FIGURES / "fig2_progressive_correction.png", dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("  fig2 saved.")

    # ── Figure 3: individual term corrections — bar chart ─────────────────────
    names   = [r[0] for r in results_individual]
    strehls = [r[4] for r in results_individual]
    j_vals  = [r[1] for r in results_individual]
    colors  = plt.cm.viridis(np.linspace(0.2, 0.85, len(names)))

    fig, ax = plt.subplots(figsize=(9, 4.5))
    bars = ax.bar(np.arange(len(names)), strehls, color=colors, alpha=0.85, width=0.6)
    ax.axhline(S_aberr, color='grey', ls='--', lw=1.5, label=f'No correction (S={S_aberr:.3f})')
    ax.axhline(1.0,     color='k',    ls='-',  lw=1.2, label='Ideal (S=1)')
    ax.set_xticks(np.arange(len(names)))
    ax.set_xticklabels([f"Z{j}\n{n}" for j, n in zip(j_vals, names)], fontsize=9)
    ax.set_ylabel("Strehl ratio after single-term correction", fontsize=11)
    ax.set_title(
        "Strehl Ratio When Correcting Each Zernike Term Individually\n"
        "(other terms left uncorrected)",
        fontsize=11
    )
    for bar, s in zip(bars, strehls):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{s:.3f}', ha='center', va='bottom', fontsize=8)
    ax.legend(fontsize=9)
    ax.set_ylim(0, 1.15)
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    fig.savefig(FIGURES / "fig3_individual_corrections.png", dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("  fig3 saved.")

    # ── Figure 4: over/under-correction scan on defocus ───────────────────────
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(c_scan, I_scan, lw=1.8, color='darkorchid')
    ax.axvline(c_true, color='crimson', ls='--', lw=1.5,
               label=f'True aberration c={c_true:.2f}')
    ax.axvline(0,      color='grey',   ls=':',  lw=1.2, label='No correction')
    ax.set_xlabel(r"Correction amplitude $c$ (defocus Z$_4$)", fontsize=11)
    ax.set_ylabel("$I$ at focus", fontsize=11)
    ax.set_title(
        "Sensitivity of Focus to Defocus Correction Amplitude\n"
        "(scanning correction coefficient; other terms fully corrected)",
        fontsize=11
    )
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(FIGURES / "fig4_defocus_scan.png", dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("  fig4 saved.")

    # ── Figure 5: SLM phase — aberration, correction, combined ──────────────
    phi_correction_display = correction_running % (2 * np.pi)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("SLM Phase Patterns (1000×1000)", fontsize=12, fontweight='bold')
    for ax, phase, title in [
        (axes[0], aberration,                 "Aberration phase (simulated)"),
        (axes[1], phi_correction_display,     "Zernike correction phase\n(mod 2π, added to hologram)"),
        (axes[2], phi_full,                   "Combined: hologram + correction\n(mod 2π)"),
    ]:
        im = ax.imshow(phase, cmap='hsv', vmin=0, vmax=2*np.pi, origin='lower')
        plt.colorbar(im, ax=ax, label='Phase (rad)', fraction=0.046, pad=0.04)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("SLM pixel x"); ax.set_ylabel("SLM pixel y")
    plt.tight_layout()
    fig.savefig(FIGURES / "fig5_slm_phases.png", dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("  fig5 saved.")

    # ── summary ───────────────────────────────────────────────────────────────
    print(f"\n{'':─<55}")
    print(f"{'Condition':<30} {'I':>8} {'Strehl':>8}")
    print(f"{'':─<55}")
    for label, I, S in [
        ("Ideal (no aberration)",    I_ideal, 1.0),
        ("Aberrated, no correction", I_aberr, S_aberr),
        ("Full Zernike correction",  I_full,  S_full),
    ]:
        print(f"{label:<30} {I:>8.4f} {S:>8.3f}")
    print(f"{'':─<55}")
    print(f"\nFigures saved to {FIGURES}")


if __name__ == "__main__":
    run_and_plot()
