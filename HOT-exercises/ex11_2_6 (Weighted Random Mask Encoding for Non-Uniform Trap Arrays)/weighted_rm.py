"""
ex11_2_6.py
-----------
Exercise 11.2.6 -- Weighted Random Mask Encoding

Standard task: 10x10 trap array, N=100 traps, 512x512 SLM.

Standard RM assigns each pixel to a trap chosen uniformly at random.
Weighted RM assigns pixel to trap n with probability proportional to
sqrt(I_target_n), so that the expected number of pixels for trap n is

    p_n = sqrt(I_target_n) / sum_m sqrt(I_target_m)

This is derived from the fact that the intensity at trap n scales as the
square of the number of pixels assigned to it (coherent addition):
    I_n ~ p_n^2 * N_total^2
So to achieve I_n ~ I_target_n we need p_n ~ sqrt(I_target_n).

Three weight profiles are studied:
  1. Uniform weights (standard RM, baseline)
  2. Linear gradient across the 10x10 grid (left traps dim, right traps bright)
  3. Gaussian weights (centre bright, edges dim)
  4. Checkerboard (alternating bright/dim)

Performance metrics: u (uniformity relative to target), sigma, and a new
metric -- weighted efficiency -- measuring how well the relative intensities
match the target ratios.
"""

import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(0)

# ── SLM grid ──────────────────────────────────────────────────────────────────
N_slm = 512
mx = np.arange(N_slm) - N_slm // 2
my = np.arange(N_slm) - N_slm // 2
MX, MY = np.meshgrid(mx, my, indexing='xy')

# ── Trap grid ─────────────────────────────────────────────────────────────────
n_side  = 10
spacing = 8
half    = (n_side - 1) / 2
bx_1d   = (np.arange(n_side) - half) * spacing
TBX, TBY = np.meshgrid(bx_1d, bx_1d)
bx_all  = TBX.ravel()   # shape (100,)
by_all  = TBY.ravel()
N_traps = len(bx_all)

# trap (row, col) index for each trap in the flat array
trap_row = np.repeat(np.arange(n_side), n_side)   # 0..9 repeated 10 times
trap_col = np.tile(np.arange(n_side), n_side)      # 0..9 tiled 10 times

# ── Phase and FFT helpers ─────────────────────────────────────────────────────
def phi_single(bx, by):
    return (2 * np.pi / N_slm) * (MX * bx + MY * by)

def focal_raw(phi):
    ft = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(np.exp(1j * phi))))
    return np.abs(ft) ** 2

def display_norm(I):
    nz = I[I > 0]
    hi = np.percentile(nz, 99) if len(nz) else 1.0
    return np.clip(I / hi, 0, 1)

def read_traps(I, bx_arr, by_arr, w=1):
    c = N_slm // 2
    return np.array([
        I[c + int(round(by)) - w : c + int(round(by)) + w + 1,
          c + int(round(bx)) - w : c + int(round(bx)) + w + 1].sum()
        for bx, by in zip(bx_arr, by_arr)
    ])

# ── Weighted RM hologram ───────────────────────────────────────────────────────
def phi_weighted_RM(bx_all, by_all, weights):
    """
    weights: desired intensity for each trap (unnormalised).
    Pixel assignment probability proportional to sqrt(weights).
    """
    probs = np.sqrt(weights)
    probs = probs / probs.sum()   # normalise to a probability distribution

    phi = np.zeros((N_slm, N_slm))
    N_total = N_slm * N_slm
    # draw a trap index for every pixel according to probs
    trap_idx = rng.choice(N_traps, size=N_total, p=probs)

    for n, (bx, by) in enumerate(zip(bx_all, by_all)):
        mask = (trap_idx == n).reshape(N_slm, N_slm)
        phi[mask] = phi_single(bx, by)[mask]

    return phi

# ── Performance metrics ────────────────────────────────────────────────────────
def metrics(In, I_target=None):
    """
    Standard metrics plus a normalised RMS error relative to target.
    If I_target is None, target is uniform.
    """
    I_mean = In.mean()
    u      = 1 - (In.max() - In.min()) / (In.max() + In.min())
    sigma  = np.sqrt(((In - I_mean)**2).mean()) / I_mean * 100

    if I_target is not None:
        # normalise both to same mean and compute RMS fractional error
        In_n  = In      / In.mean()
        It_n  = I_target / I_target.mean()
        rms_err = np.sqrt(((In_n - It_n)**2).mean()) * 100   # as %
    else:
        rms_err = None

    return I_mean, u, sigma, rms_err

# ── Target weight profiles ────────────────────────────────────────────────────
def make_weights(profile):
    if profile == 'uniform':
        return np.ones(N_traps)
    elif profile == 'gradient':
        # intensity ramps linearly from 0.1 to 1.0 along x (column index)
        return 0.1 + 0.9 * trap_col / (n_side - 1)
    elif profile == 'gaussian':
        # Gaussian centred on the array, sigma = 3 traps
        cx = cy = (n_side - 1) / 2
        r2 = (trap_col - cx)**2 + (trap_row - cy)**2
        return np.exp(-r2 / (2 * 3.0**2))
    elif profile == 'checkerboard':
        return np.where((trap_row + trap_col) % 2 == 0, 1.0, 0.1)

profiles = ['uniform', 'gradient', 'gaussian', 'checkerboard']
labels   = ['Uniform (standard RM)',
            'Linear gradient (x)',
            'Gaussian (centre bright)',
            'Checkerboard']

# ── Compute all four cases ─────────────────────────────────────────────────────
results = {}
for prof in profiles:
    print(f"Computing {prof}...")
    W      = make_weights(prof)
    phi    = phi_weighted_RM(bx_all, by_all, W)
    I_raw  = focal_raw(phi)
    In     = read_traps(I_raw, bx_all, by_all)
    Im, u, sigma, rms = metrics(In, I_target=W)
    results[prof] = dict(W=W, phi=phi, I_raw=I_raw, In=In,
                         Im=Im, u=u, sigma=sigma, rms=rms)
    print(f"  u={u:.4f}  sigma={sigma:.1f}%  rms_err={rms:.1f}%")

# ── Figure 1: target vs achieved intensity maps (2x4 grid) ────────────────────
fig1, axes1 = plt.subplots(2, 4, figsize=(15, 7))
fig1.suptitle("Weighted RM: target (top) vs achieved intensity (bottom)", fontsize=12)

c   = N_slm // 2
ext = int(half * spacing * 1.4)

for col, (prof, label) in enumerate(zip(profiles, labels)):
    r  = results[prof]
    W  = r['W'].reshape(n_side, n_side)
    In = r['In'].reshape(n_side, n_side)

    # top: target weights as heatmap on trap grid
    im0 = axes1[0, col].imshow(W, origin='lower', cmap='hot', vmin=0, vmax=1)
    axes1[0, col].set_title(label, fontsize=9)
    axes1[0, col].set_xlabel("trap col", fontsize=8)
    if col == 0:
        axes1[0, col].set_ylabel("trap row\n(TARGET)", fontsize=8)
    plt.colorbar(im0, ax=axes1[0, col], fraction=0.046, pad=0.04)

    # bottom: achieved intensity at trap sites
    In_norm = In / In.max()
    im1 = axes1[1, col].imshow(In_norm, origin='lower', cmap='hot', vmin=0, vmax=1)
    axes1[1, col].set_xlabel("trap col", fontsize=8)
    if col == 0:
        axes1[1, col].set_ylabel("trap row\n(ACHIEVED)", fontsize=8)
    axes1[1, col].set_title(
        f"$u={r['u']:.3f}$  $\\sigma={r['sigma']:.0f}\\%$\n"
        f"rms err$={r['rms']:.0f}\\%$", fontsize=8)
    plt.colorbar(im1, ax=axes1[1, col], fraction=0.046, pad=0.04)

plt.tight_layout()
fig1.savefig("/mnt/user-data/outputs/ex11_2_6_fig1_target_vs_achieved.png", dpi=150)
print("Saved fig1")

# ── Figure 2: scatter plot -- target vs achieved (all four profiles) ───────────
fig2, axes2 = plt.subplots(1, 4, figsize=(14, 4))
fig2.suptitle("Weighted RM: target vs achieved intensity per trap", fontsize=12)

for ax, prof, label in zip(axes2, profiles, labels):
    r  = results[prof]
    W  = r['W']
    In = r['In']
    # normalise both to mean=1
    Wn  = W  / W.mean()
    Inn = In / In.mean()
    ax.scatter(Wn, Inn, s=12, color='steelblue', alpha=0.7)
    xlim = max(Wn.max(), Inn.max()) * 1.1
    ax.plot([0, xlim], [0, xlim], 'r--', lw=1, label='ideal')
    ax.set_xlabel("Target $I_n$ (norm.)", fontsize=9)
    ax.set_ylabel("Achieved $I_n$ (norm.)", fontsize=9)
    ax.set_title(f"{label}\nrms={r['rms']:.0f}%", fontsize=9)
    ax.legend(fontsize=7)
    ax.tick_params(labelsize=8)

plt.tight_layout()
fig2.savefig("/mnt/user-data/outputs/ex11_2_6_fig2_scatter.png", dpi=150)
print("Saved fig2")

# ── Figure 3: focal plane images for all four profiles ────────────────────────
fig3, axes3 = plt.subplots(1, 4, figsize=(15, 4))
fig3.suptitle("Weighted RM: focal-plane intensity", fontsize=12)

for ax, prof, label in zip(axes3, profiles, labels):
    r = results[prof]
    ax.imshow(display_norm(r['I_raw'])[c-ext:c+ext, c-ext:c+ext],
              origin='lower', extent=[-ext, ext, -ext, ext],
              cmap='inferno', vmin=0, vmax=1)
    ax.set_title(label, fontsize=9)
    ax.set_xlabel("bin x", fontsize=8)
    ax.set_ylabel("bin y", fontsize=8)
    ax.tick_params(labelsize=7)

plt.tight_layout()
fig3.savefig("/mnt/user-data/outputs/ex11_2_6_fig3_focal_plane.png", dpi=150)
print("Saved fig3")

# ── Figure 4: pixel allocation vs achieved intensity (uniform case) ────────────
# Show that achieved I_n ~ p_n^2 as expected from coherent addition
prof = 'gradient'
r    = results[prof]
W    = r['W']
probs = np.sqrt(W); probs /= probs.sum()
In   = r['In']

fig4, (ax4a, ax4b) = plt.subplots(1, 2, figsize=(10, 4))
fig4.suptitle("Verification: $I_n \\propto p_n^2$ (gradient profile)", fontsize=11)

# expected: In ~ (p_n * N_total)^2
N_total = N_slm * N_slm
expected = (probs * N_total)**2
expected_norm = expected / expected.mean()
In_norm = In / In.mean()

ax4a.scatter(expected_norm, In_norm, s=12, color='steelblue', alpha=0.7)
xlim = max(expected_norm.max(), In_norm.max()) * 1.1
ax4a.plot([0, xlim], [0, xlim], 'r--', lw=1, label='$I_n = p_n^2 N^2$')
ax4a.set_xlabel("Expected $p_n^2 N^2$ (norm.)", fontsize=9)
ax4a.set_ylabel("Achieved $I_n$ (norm.)", fontsize=9)
ax4a.set_title("Agreement with $I_n \\propto p_n^2$", fontsize=9)
ax4a.legend(fontsize=8)

ax4b.bar(range(N_traps), probs * N_total, color='steelblue', width=0.8, alpha=0.7,
         label='pixels assigned (expected)')
ax4b.set_xlabel("Trap index"); ax4b.set_ylabel("Expected pixel count")
ax4b.set_title("Pixel allocation (gradient profile)", fontsize=9)
ax4b.legend(fontsize=8)

plt.tight_layout()
fig4.savefig("/mnt/user-data/outputs/ex11_2_6_fig4_verification.png", dpi=150)
print("Saved fig4")

plt.show()
print("Done.")
