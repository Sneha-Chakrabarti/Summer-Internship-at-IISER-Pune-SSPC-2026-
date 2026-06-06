"""
ex11_2_5.py
-----------
Exercise 11.2.5 -- Helper tweezers via Random Mask Encoding (RM)

A pre-calculated SGL hologram generates a complex light structure (10x10
trap array). A fraction f of SLM pixels is temporarily reassigned using
Random Mask Encoding to generate M=4 helper tweezers outside the main array.
The remaining (1-f) pixels maintain the SGL hologram.

This demonstrates that RM, despite its low efficiency for large N, is very
useful for adding a small number of helper traps on top of an existing
hologram without recomputing the full hologram.
"""

import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(42)

# ── SLM grid ──────────────────────────────────────────────────────────────────
N_slm = 512
mx = np.arange(N_slm) - N_slm // 2
my = np.arange(N_slm) - N_slm // 2
MX, MY = np.meshgrid(mx, my, indexing='xy')

def phi_single(bx, by):
    return (2 * np.pi / N_slm) * (MX * bx + MY * by)

def phi_SGL(bx_arr, by_arr):
    field_sum = np.zeros((N_slm, N_slm), dtype=complex)
    for bx, by in zip(bx_arr, by_arr):
        field_sum += np.exp(1j * phi_single(bx, by))
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
    vals = []
    for bx, by in zip(bx_arr, by_arr):
        bxi, byi = int(round(bx)), int(round(by))
        patch = I[c+byi-w : c+byi+w+1, c+bxi-w : c+bxi+w+1]
        vals.append(patch.sum())
    return np.array(vals)

def metrics(In):
    I_mean = In.mean()
    u      = 1 - (In.max() - In.min()) / (In.max() + In.min())
    sigma  = np.sqrt(((In - I_mean)**2).mean()) / I_mean * 100
    return I_mean, u, sigma

# ── Main 10x10 array ──────────────────────────────────────────────────────────
n_side = 10; spacing = 8; half = (n_side-1)/2
bx_1d  = (np.arange(n_side) - half) * spacing
TBX, TBY = np.meshgrid(bx_1d, bx_1d)
bx_main, by_main = TBX.ravel(), TBY.ravel()

print("Computing main SGL hologram...")
phi_main = phi_SGL(bx_main, by_main)
I_ref    = focal_raw(phi_main)
In_ref   = read_traps(I_ref, bx_main, by_main)
Im_ref, u_ref, sig_ref = metrics(In_ref)
print(f"Main array alone: <I>={Im_ref:.3e}  u={u_ref:.4f}  sigma={sig_ref:.1f}%")

# ── Helper tweezers: 4 traps at corners outside the main array ────────────────
h_off = int(half * spacing * 1.7)   # offset beyond last trap row/col
bx_help = np.array([ h_off, -h_off,  h_off, -h_off], dtype=float)
by_help = np.array([ h_off,  h_off, -h_off, -h_off], dtype=float)
M_help  = len(bx_help)

# ── Build combined mask ───────────────────────────────────────────────────────
def combined_mask(phi_main, bx_help, by_help, f_helper):
    """
    f_helper fraction of pixels -> RM encoding for helper traps.
    Remaining pixels -> original SGL hologram.
    """
    phi_comb = phi_main.copy().ravel()
    N_total  = N_slm * N_slm
    N_help   = int(f_helper * N_total)
    idx      = rng.permutation(N_total)[:N_help]
    trap_idx = rng.integers(0, M_help, size=N_help)
    for i, ti in zip(idx, trap_idx):
        iy, ix   = divmod(i, N_slm)
        phi_comb[i] = phi_single(bx_help[ti], by_help[ti])[iy, ix]
    return phi_comb.reshape(N_slm, N_slm)

# ── Demo: f=0.20 ──────────────────────────────────────────────────────────────
f_demo   = 0.20
phi_comb = combined_mask(phi_main, bx_help, by_help, f_demo)
I_comb   = focal_raw(phi_comb)

In_main_comb = read_traps(I_comb, bx_main, by_main)
In_help_comb = read_traps(I_comb, bx_help, by_help)
Im_c, u_c, sig_c = metrics(In_main_comb)
print(f"\nWith {int(f_demo*100)}% helper pixels:")
print(f"  Main array: <I>={Im_c:.3e}  u={u_c:.4f}  sigma={sig_c:.1f}%")
print(f"  Helper I (each): {In_help_comb / Im_ref}")

# ── Figure 1: side-by-side comparison ────────────────────────────────────────
c   = N_slm // 2
ext = int(h_off * 1.25)

fig1, axes1 = plt.subplots(1, 2, figsize=(12, 5))
fig1.suptitle(f"Helper tweezers via RM encoding ({int(f_demo*100)}% of SLM pixels)", fontsize=12)

for ax, I_plot, title in zip(axes1,
    [I_ref, I_comb],
    ["Main SGL hologram only (no helpers)",
     f"Main SGL + {M_help} helper traps (RM, $f={f_demo}$)"]):
    ax.imshow(display_norm(I_plot)[c-ext:c+ext, c-ext:c+ext],
              origin='lower', extent=[-ext,ext,-ext,ext], cmap='inferno', vmin=0, vmax=1)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("bin x"); ax.set_ylabel("bin y")
    for bx, by in zip(bx_main, by_main):
        ax.plot(bx, by, '+', color='cyan', ms=4, mew=0.7)

for bx, by in zip(bx_help, by_help):
    axes1[1].plot(bx, by, 'x', color='yellow', ms=8, mew=1.8)

axes1[0].legend([plt.Line2D([0],[0],marker='+',color='cyan',ls='none',ms=6)],
                ['main traps'], fontsize=8, loc='upper right')
axes1[1].legend([plt.Line2D([0],[0],marker='+',color='cyan',ls='none',ms=6),
                 plt.Line2D([0],[0],marker='x',color='yellow',ls='none',ms=8)],
                ['main traps','helper traps'], fontsize=8, loc='upper right')

plt.tight_layout()
fig1.savefig("/mnt/user-data/outputs/ex11_2_5_fig1_demo.png", dpi=150)
print("Saved fig1")

# ── Figure 2: metrics vs f_helper ────────────────────────────────────────────
print("\nSweeping helper fraction...")
f_vals = [0.0, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50]
us_m, sigs_m, I_main_n, I_help_n = [], [], [], []

for fh in f_vals:
    if fh == 0:
        I_p = I_ref
    else:
        I_p = focal_raw(combined_mask(phi_main, bx_help, by_help, fh))
    In_m = read_traps(I_p, bx_main, by_main)
    In_h = read_traps(I_p, bx_help, by_help)
    Im, u_m, sig_m = metrics(In_m)
    us_m.append(u_m); sigs_m.append(sig_m)
    I_main_n.append(Im / Im_ref)
    I_help_n.append(In_h.mean() / Im_ref)
    print(f"  f={fh:.2f}: u={u_m:.4f}  sigma={sig_m:.1f}%  "
          f"<I_main>/<I_ref>={Im/Im_ref:.3f}  <I_help>/<I_ref>={In_h.mean()/Im_ref:.4f}")

fig2, axes2 = plt.subplots(1, 3, figsize=(13, 4))
fig2.suptitle("Effect of helper pixel fraction $f$ on main array and helper traps", fontsize=11)

axes2[0].plot(f_vals, us_m, 'o-', color='steelblue')
axes2[0].set_xlabel("Helper fraction $f$"); axes2[0].set_ylabel("Uniformity $u$ (main)")
axes2[0].set_ylim(0, 1.05)

axes2[1].plot(f_vals, sigs_m, 'o-', color='tomato')
axes2[1].set_xlabel("Helper fraction $f$"); axes2[1].set_ylabel("$\\sigma$ % (main)")

axes2[2].plot(f_vals, I_main_n, 'o-', color='steelblue', label='main $\\langle I\\rangle$')
axes2[2].plot(f_vals, I_help_n, 's--', color='orange', label='helper $\\langle I\\rangle$')
axes2[2].set_xlabel("Helper fraction $f$")
axes2[2].set_ylabel("Intensity / $\\langle I_{ref}\\rangle$")
axes2[2].legend(fontsize=8)

for ax in axes2: ax.tick_params(labelsize=8)
plt.tight_layout()
fig2.savefig("/mnt/user-data/outputs/ex11_2_5_fig2_metrics_vs_f.png", dpi=150)
print("Saved fig2")

plt.show()
print("Done.")
