"""
ex11_2_4.py
-----------
Exercise 11.2.4 -- Superposition of Gratings and Lenses (SGL)

Standard task: N=100 traps on a 10x10 square lattice, focal plane (z=0).
SLM: 512x512 pixels (book uses 1000x1000; physics is identical).

Phase mask (Eq. 8):
    phi_SGL = arg{ sum_{n=1}^{N} exp(i * phi_s(mx,my; xn,yn)) }

    phi_s = (2*pi/N_slm) * (mx*bxn + my*byn)

where bxn,byn are the trap positions in FFT-bin units.

The arg() operation keeps only the phase, discarding amplitude -- this is
what a phase-only SLM can encode. It gives good efficiency but, as the
simulation confirms, very poor uniformity (sigma ~ 260%). This is the
known trade-off of SGL (see book Table, Section 5).

Performance metrics:
    I_tot  = sum(In)
    <I>    = I_tot / N
    u      = 1 - (max - min) / (max + min)   [1 = perfect]
    sigma  = sqrt(mean((In-<I>)^2)) / <I> * 100  [% std error]
"""

import numpy as np
import matplotlib.pyplot as plt

# ── SLM grid ──────────────────────────────────────────────────────────────────
N_slm = 512
mx = np.arange(N_slm) - N_slm // 2
my = np.arange(N_slm) - N_slm // 2
MX, MY = np.meshgrid(mx, my, indexing='xy')

# ── Trap positions (FFT-bin units) ────────────────────────────────────────────
n_side  = 10
spacing = 8                              # bins between adjacent traps
half    = (n_side - 1) / 2              # = 4.5
bx_1d   = (np.arange(n_side) - half) * spacing   # -36,-28,...,28,36
by_1d   = bx_1d.copy()
TBX, TBY = np.meshgrid(bx_1d, by_1d)
bx_all   = TBX.ravel()    # shape (100,)
by_all   = TBY.ravel()
N_traps  = len(bx_all)    # 100

# ── Phase mask builders ───────────────────────────────────────────────────────
def phi_single(bx, by):
    return (2 * np.pi / N_slm) * (MX * bx + MY * by)

def phi_SGL(bx_arr, by_arr):
    field_sum = np.zeros((N_slm, N_slm), dtype=complex)
    for bx, by in zip(bx_arr, by_arr):
        field_sum += np.exp(1j * phi_single(bx, by))
    return np.angle(field_sum)

# ── FFT ───────────────────────────────────────────────────────────────────────
def focal_raw(phi):
    ft = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(np.exp(1j * phi))))
    return np.abs(ft) ** 2

def display_norm(I):
    nz = I[I > 0]
    hi = np.percentile(nz, 99) if len(nz) else 1.0
    return np.clip(I / hi, 0, 1)

def read_traps(I, bx_arr, by_arr, w=1):
    """Sum intensity in (2w+1)x(2w+1) patch around each trap bin."""
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
    return In.sum(), I_mean, u, sigma

# ── Compute SGL for default spacing ───────────────────────────────────────────
print("Computing SGL hologram (10x10, spacing=8 bins)...")
phi = phi_SGL(bx_all, by_all)
I   = focal_raw(phi)
In  = read_traps(I, bx_all, by_all)
I_tot, I_mean, u, sigma = metrics(In)
print(f"  I_tot={I_tot:.3e}  <I>={I_mean:.3e}  u={u:.4f}  sigma={sigma:.1f}%")
print(f"  (High sigma is expected: SGL trades uniformity for efficiency.)")

# ── Figure 1: hologram + focal plane ──────────────────────────────────────────
fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
fig1.suptitle("SGL hologram: 10x10 trap array (spacing=8 bins)", fontsize=12)

ax1.imshow(phi, cmap='hsv', origin='lower')
ax1.set_title("SGL phase mask $\\phi^{\\mathrm{SGL}}$", fontsize=10)
ax1.set_xlabel("pixel $m_x$"); ax1.set_ylabel("pixel $m_y$")

ext = int(half * spacing * 1.5)
ax2.imshow(display_norm(I)[N_slm//2-ext:N_slm//2+ext,
                            N_slm//2-ext:N_slm//2+ext],
           origin='lower', extent=[-ext,ext,-ext,ext], cmap='inferno', vmin=0, vmax=1)
ax2.set_title("Focal-plane intensity (trap region)", fontsize=10)
ax2.set_xlabel("bin x"); ax2.set_ylabel("bin y")
for bx, by in zip(bx_all, by_all):
    ax2.plot(bx, by, '+', color='cyan', ms=4, mew=0.7)
ax2.text(0.03, 0.97, f"$u={u:.3f}$\n$\\sigma={sigma:.0f}\\%$",
         transform=ax2.transAxes, color='white', fontsize=9, va='top')

plt.tight_layout()
fig1.savefig("/mnt/user-data/outputs/ex11_2_4_fig1_hologram.png", dpi=150)
print("Saved fig1")

# ── Figure 2: per-trap intensity ──────────────────────────────────────────────
fig2, ax = plt.subplots(figsize=(12, 3.5))
ax.bar(range(N_traps), In / In.mean(), color='steelblue', width=0.8)
ax.axhline(1.0, color='tomato', lw=1.2, ls='--', label='mean')
ax.set_xlabel("Trap index"); ax.set_ylabel("$I_n / \\langle I \\rangle$")
ax.set_title(f"Per-trap intensity (SGL)  |  $u={u:.3f}$,  $\\sigma={sigma:.0f}\\%$  "
             f"-- strong non-uniformity is characteristic of SGL", fontsize=10)
ax.legend(fontsize=8)
plt.tight_layout()
fig2.savefig("/mnt/user-data/outputs/ex11_2_4_fig2_per_trap.png", dpi=150)
print("Saved fig2")

# ── Figure 3: metrics vs spacing ──────────────────────────────────────────────
print("\nSweeping trap spacing...")
spacings = [4, 6, 8, 10, 14, 20]
us, sigmas, I_tots = [], [], []

for sp in spacings:
    bx = (np.arange(n_side) - half) * sp
    by = bx.copy()
    TBX2, TBY2 = np.meshgrid(bx, by)
    bxr, byr = TBX2.ravel(), TBY2.ravel()
    if np.abs(bxr).max() >= N_slm // 2:
        print(f"  spacing={sp}: exceeds Nyquist, skipped"); continue
    I_sp  = focal_raw(phi_SGL(bxr, byr))
    In_sp = read_traps(I_sp, bxr, byr)
    It, Im, u_sp, sig_sp = metrics(In_sp)
    us.append(u_sp); sigmas.append(sig_sp); I_tots.append(It)
    print(f"  spacing={sp:3d}: u={u_sp:.4f}  sigma={sig_sp:.1f}%  I_tot={It:.3e}")

sp_used = spacings[:len(us)]

fig3, axes3 = plt.subplots(1, 3, figsize=(13, 4))
fig3.suptitle("SGL metrics vs. trap spacing (10x10 array)", fontsize=11)
axes3[0].plot(sp_used, us, 'o-', color='steelblue')
axes3[0].set_xlabel("Spacing (bins)"); axes3[0].set_ylabel("Uniformity $u$")
axes3[0].set_ylim(0, 1.05)
axes3[1].plot(sp_used, sigmas, 'o-', color='tomato')
axes3[1].set_xlabel("Spacing (bins)"); axes3[1].set_ylabel("$\\sigma$ (%)")
I_norm = [x/I_tots[0] for x in I_tots]
axes3[2].plot(sp_used, I_norm, 'o-', color='seagreen')
axes3[2].set_xlabel("Spacing (bins)"); axes3[2].set_ylabel("$I_{tot}$ (normalised)")
for ax in axes3: ax.tick_params(labelsize=8)
plt.tight_layout()
fig3.savefig("/mnt/user-data/outputs/ex11_2_4_fig3_metrics_vs_spacing.png", dpi=150)
print("Saved fig3")

plt.show()
print("Done.")
