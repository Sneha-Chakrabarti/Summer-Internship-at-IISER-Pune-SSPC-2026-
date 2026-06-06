"""
ex11_2_10.py
------------
Exercise 11.2.10 -- Weighted SGL for arrays of weighted traps

Standard SGL:
    phi_SGL = arg{ sum_n exp(i * phi_s_n) }

Weighted SGL (eq. 9 extension):
    phi_wSGL = arg{ sum_n sqrt(I_target_n) * exp(i * phi_s_n) }

Weighting each term by sqrt(I_target_n) before taking arg() biases the
complex sum toward traps with higher target intensity, giving them more
"phase influence" in the hologram.

Why sqrt? In the SGL sum, each term contributes equally to the complex
amplitude. The resulting focal-plane intensity at trap n scales roughly as
the square of the real part of the total phasor projected onto the n-th
direction. Weighting by sqrt(I_target_n) makes the amplitude contribution
proportional to sqrt(I), so the intensity contribution ~ I_target_n.
This is analogous to the RM argument but applied in the complex-amplitude domain.

Four target profiles are tested (same as ex11_2_6 for direct comparison):
  uniform, gradient, gaussian, checkerboard
"""

import numpy as np
import matplotlib.pyplot as plt

# ── SLM grid ──────────────────────────────────────────────────────────────────
N_slm = 512
mx = np.arange(N_slm) - N_slm // 2
my = np.arange(N_slm) - N_slm // 2
MX, MY = np.meshgrid(mx, my, indexing='xy')

def phi_single(bx, by):
    return (2 * np.pi / N_slm) * (MX * bx + MY * by)

def phi_wSGL(bx_arr, by_arr, weights):
    """
    Weighted SGL: phi = arg{ sum_n sqrt(w_n) * exp(i*phi_s_n) }.
    weights: desired intensities (unnormalised).
    """
    field_sum = np.zeros((N_slm, N_slm), dtype=complex)
    sq = np.sqrt(np.asarray(weights, dtype=float))
    for s, bx, by in zip(sq, bx_arr, by_arr):
        field_sum += s * np.exp(1j * phi_single(bx, by))
    return np.angle(field_sum)

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

def metrics(In, I_target=None):
    Im = In.mean()
    u  = 1 - (In.max() - In.min()) / (In.max() + In.min())
    sg = np.sqrt(((In - Im)**2).mean()) / Im * 100
    rms = None
    if I_target is not None:
        In_n = In / Im
        It_n = I_target / I_target.mean()
        rms  = np.sqrt(((In_n - It_n)**2).mean()) * 100
    return Im, u, sg, rms

# ── Trap grid ──────────────────────────────────────────────────────────────────
n_side = 10; spacing = 8; half = (n_side - 1) / 2
bx_1d  = (np.arange(n_side) - half) * spacing
TBX, TBY = np.meshgrid(bx_1d, bx_1d)
bx_all = TBX.ravel(); by_all = TBY.ravel()
N_traps = len(bx_all)
trap_row = np.repeat(np.arange(n_side), n_side)
trap_col = np.tile(np.arange(n_side), n_side)

def make_weights(profile):
    if profile == 'uniform':
        return np.ones(N_traps)
    elif profile == 'gradient':
        return 0.1 + 0.9 * trap_col / (n_side - 1)
    elif profile == 'gaussian':
        cx = cy = (n_side - 1) / 2
        r2 = (trap_col - cx)**2 + (trap_row - cy)**2
        return np.exp(-r2 / (2 * 3.0**2))
    elif profile == 'checkerboard':
        return np.where((trap_row + trap_col) % 2 == 0, 1.0, 0.1)

profiles = ['uniform', 'gradient', 'gaussian', 'checkerboard']
labels   = ['Uniform', 'Gradient (x)', 'Gaussian', 'Checkerboard']

# ── Compute both standard SGL and weighted SGL for all profiles ───────────────
def phi_SGL_uniform(bx_arr, by_arr):
    fs = np.zeros((N_slm, N_slm), dtype=complex)
    for bx, by in zip(bx_arr, by_arr):
        fs += np.exp(1j * phi_single(bx, by))
    return np.angle(fs)

print("Computing standard SGL (uniform weights) for reference...")
phi_std  = phi_SGL_uniform(bx_all, by_all)
I_std    = focal_raw(phi_std)
In_std   = read_traps(I_std, bx_all, by_all)
Im_ref   = In_std.mean()

results_std  = {}
results_wSGL = {}

for prof in profiles:
    W = make_weights(prof)
    print(f"Computing weighted SGL: {prof}...")

    # standard SGL (ignores weights) -- reference
    In_s = read_traps(I_std, bx_all, by_all)  # same hologram, just different target
    Im_s, u_s, sg_s, rms_s = metrics(In_s, W)
    results_std[prof] = dict(W=W, In=In_s, u=u_s, sg=sg_s, rms=rms_s)

    # weighted SGL
    phi_w   = phi_wSGL(bx_all, by_all, W)
    I_w     = focal_raw(phi_w)
    In_w    = read_traps(I_w, bx_all, by_all)
    Im_w, u_w, sg_w, rms_w = metrics(In_w, W)
    results_wSGL[prof] = dict(W=W, phi=phi_w, I_raw=I_w, In=In_w,
                               Im=Im_w, u=u_w, sg=sg_w, rms=rms_w)

    print(f"  Standard SGL:  rms_err={rms_s:.1f}%  u={u_s:.4f}")
    print(f"  Weighted SGL:  rms_err={rms_w:.1f}%  u={u_w:.4f}  sigma={sg_w:.1f}%")

# ── Figure 1: target vs achieved (standard vs weighted SGL, 2 profiles) ───────
fig1, axes1 = plt.subplots(3, 4, figsize=(15, 10))
fig1.suptitle("Standard SGL vs Weighted SGL: target / achieved intensity maps", fontsize=12)

c   = N_slm // 2
ext = int(half * spacing * 1.4)

for col, (prof, label) in enumerate(zip(profiles, labels)):
    W  = make_weights(prof).reshape(n_side, n_side)

    # row 0: target
    im = axes1[0, col].imshow(W / W.max(), origin='lower', cmap='hot', vmin=0, vmax=1)
    axes1[0, col].set_title(label, fontsize=10)
    if col == 0: axes1[0, col].set_ylabel("TARGET", fontsize=9)
    plt.colorbar(im, ax=axes1[0, col], fraction=0.046, pad=0.04)

    # row 1: standard SGL achieved
    r = results_std[prof]
    In_map = r['In'].reshape(n_side, n_side)
    im1 = axes1[1, col].imshow(In_map / In_map.max(), origin='lower', cmap='hot', vmin=0, vmax=1)
    if col == 0: axes1[1, col].set_ylabel("STANDARD SGL", fontsize=9)
    axes1[1, col].set_title(f"rms={r['rms']:.0f}%", fontsize=9)
    plt.colorbar(im1, ax=axes1[1, col], fraction=0.046, pad=0.04)

    # row 2: weighted SGL achieved
    r = results_wSGL[prof]
    In_map = r['In'].reshape(n_side, n_side)
    im2 = axes1[2, col].imshow(In_map / In_map.max(), origin='lower', cmap='hot', vmin=0, vmax=1)
    if col == 0: axes1[2, col].set_ylabel("WEIGHTED SGL", fontsize=9)
    axes1[2, col].set_title(f"rms={r['rms']:.0f}%", fontsize=9)
    plt.colorbar(im2, ax=axes1[2, col], fraction=0.046, pad=0.04)

plt.tight_layout()
fig1.savefig("/mnt/user-data/outputs/ex11_2_10_fig1_maps.png", dpi=150)
print("Saved fig1")

# ── Figure 2: scatter -- target vs achieved, both algorithms ──────────────────
fig2, axes2 = plt.subplots(2, 4, figsize=(15, 7))
fig2.suptitle("Target vs achieved intensity per trap", fontsize=12)

for col, (prof, label) in enumerate(zip(profiles, labels)):
    W = make_weights(prof)
    Wn = W / W.mean()
    for row, (res, alg) in enumerate([(results_std, 'Standard SGL'),
                                       (results_wSGL, 'Weighted SGL')]):
        r   = res[prof]
        Inn = r['In'] / r['In'].mean()
        ax  = axes2[row, col]
        ax.scatter(Wn, Inn, s=10, color='steelblue' if row==1 else 'gray', alpha=0.7)
        lim = max(Wn.max(), Inn.max()) * 1.1
        ax.plot([0, lim], [0, lim], 'r--', lw=1)
        ax.set_xlabel("Target (norm.)", fontsize=8)
        ax.set_ylabel("Achieved (norm.)", fontsize=8)
        ax.set_title(f"{label}\n{alg}  rms={r['rms']:.0f}%", fontsize=8)
        ax.tick_params(labelsize=7)

plt.tight_layout()
fig2.savefig("/mnt/user-data/outputs/ex11_2_10_fig2_scatter.png", dpi=150)
print("Saved fig2")

# ── Figure 3: summary bar chart -- rms error comparison ───────────────────────
fig3, ax3 = plt.subplots(figsize=(9, 4))
x   = np.arange(len(profiles))
w   = 0.35
rms_std  = [results_std[p]['rms']  for p in profiles]
rms_wSGL = [results_wSGL[p]['rms'] for p in profiles]

ax3.bar(x - w/2, rms_std,  w, label='Standard SGL', color='gray',      alpha=0.8)
ax3.bar(x + w/2, rms_wSGL, w, label='Weighted SGL',  color='steelblue', alpha=0.8)
ax3.set_xticks(x); ax3.set_xticklabels(labels, fontsize=9)
ax3.set_ylabel("RMS error relative to target (%)", fontsize=9)
ax3.set_title("Weighted SGL vs Standard SGL: target fidelity across profiles", fontsize=10)
ax3.legend(fontsize=9)
ax3.tick_params(labelsize=8)

plt.tight_layout()
fig3.savefig("/mnt/user-data/outputs/ex11_2_10_fig3_rms_comparison.png", dpi=150)
print("Saved fig3")

# ── Figure 4: focal plane images, weighted SGL ────────────────────────────────
fig4, axes4 = plt.subplots(1, 4, figsize=(15, 4))
fig4.suptitle("Weighted SGL: focal-plane intensity", fontsize=12)

for ax, prof, label in zip(axes4, profiles, labels):
    r = results_wSGL[prof]
    ax.imshow(display_norm(r['I_raw'])[c-ext:c+ext, c-ext:c+ext],
              origin='lower', extent=[-ext,ext,-ext,ext], cmap='inferno', vmin=0, vmax=1)
    ax.set_title(label, fontsize=9)
    ax.set_xlabel("bin x", fontsize=8); ax.set_ylabel("bin y", fontsize=8)
    ax.tick_params(labelsize=7)

plt.tight_layout()
fig4.savefig("/mnt/user-data/outputs/ex11_2_10_fig4_focal.png", dpi=150)
print("Saved fig4")

plt.show()
print("Done.")
