"""
ex11_2_7.py
-----------
Exercise 11.2.7 -- Superposition of Gratings and Lenses: parameter study

Eq. 9 (book) / Eq. 8 (notes):
    phi_SGL = arg{ sum_{n=1}^N exp(i * phi_s_n) }

Standard task: 10x10 lattice, N=100, 512x512 SLM.

Parameters studied:
  1. Number of traps N (1x1 up to 12x12 = 144 traps)
  2. Trap spacing (lateral density of the array)
  3. Lattice geometry: square vs hexagonal vs random
"""

import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(7)

# ── SLM grid ──────────────────────────────────────────────────────────────────
N_slm = 512
mx = np.arange(N_slm) - N_slm // 2
my = np.arange(N_slm) - N_slm // 2
MX, MY = np.meshgrid(mx, my, indexing='xy')

def phi_single(bx, by):
    return (2 * np.pi / N_slm) * (MX * bx + MY * by)

def phi_SGL(bx_arr, by_arr, weights=None):
    """
    Weighted SGL: exp(i*phi_s_n) terms weighted by w_n before summing.
    weights=None -> uniform (standard SGL).
    """
    field_sum = np.zeros((N_slm, N_slm), dtype=complex)
    W = np.ones(len(bx_arr)) if weights is None else np.asarray(weights)
    for w, bx, by in zip(W, bx_arr, by_arr):
        field_sum += w * np.exp(1j * phi_single(bx, by))
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

def metrics(In):
    Im = In.mean()
    u  = 1 - (In.max() - In.min()) / (In.max() + In.min())
    sg = np.sqrt(((In - Im)**2).mean()) / Im * 100
    return Im, u, sg

# ── Trap grid generators ───────────────────────────────────────────────────────
def square_grid(n_side, spacing):
    half = (n_side - 1) / 2
    bx_1d = (np.arange(n_side) - half) * spacing
    TBX, TBY = np.meshgrid(bx_1d, bx_1d)
    return TBX.ravel(), TBY.ravel()

def hex_grid(n_side, spacing):
    """Hexagonal close-packed lattice."""
    bxs, bys = [], []
    for row in range(n_side):
        for col in range(n_side):
            x = (col - (n_side - 1) / 2) * spacing
            y = (row - (n_side - 1) / 2) * spacing * np.sqrt(3) / 2
            if row % 2 == 1:
                x += spacing / 2
            bxs.append(x); bys.append(y)
    return np.array(bxs), np.array(bys)

def random_grid(N, radius, seed=42):
    """N traps scattered randomly within a circle of given radius (bins)."""
    rng2 = np.random.default_rng(seed)
    bxs, bys = [], []
    while len(bxs) < N:
        x = rng2.uniform(-radius, radius)
        y = rng2.uniform(-radius, radius)
        if x**2 + y**2 < radius**2:
            bxs.append(x); bys.append(y)
    return np.array(bxs[:N]), np.array(bys[:N])

# ── Study 1: metrics vs number of traps (square grid, spacing=8) ─────────────
print("Study 1: metrics vs N (square grid, spacing=8)...")
n_sides  = [2, 3, 4, 5, 6, 8, 10, 12]
spacing  = 8
us, sgs, I_tots, Ns = [], [], [], []

for ns in n_sides:
    bx, by = square_grid(ns, spacing)
    if np.abs(bx).max() >= N_slm // 2: continue
    phi = phi_SGL(bx, by)
    I   = focal_raw(phi)
    In  = read_traps(I, bx, by)
    Im, u, sg = metrics(In)
    Ns.append(ns**2); us.append(u); sgs.append(sg); I_tots.append(Im)
    print(f"  n_side={ns:2d}  N={ns**2:4d}  u={u:.4f}  sigma={sg:.1f}%  <I>={Im:.3e}")

fig1, axes1 = plt.subplots(1, 3, figsize=(13, 4))
fig1.suptitle("SGL: metrics vs number of traps $N$ (square lattice, spacing=8 bins)", fontsize=11)

axes1[0].plot(Ns, us, 'o-', color='steelblue')
axes1[0].set_xlabel("$N$ (traps)"); axes1[0].set_ylabel("Uniformity $u$")
axes1[0].set_ylim(0, 1.05)

axes1[1].plot(Ns, sgs, 'o-', color='tomato')
axes1[1].set_xlabel("$N$ (traps)"); axes1[1].set_ylabel("$\\sigma$ (%)")

# normalise I_tot by N to get efficiency per trap
I_norm = [x / I_tots[0] * Ns[0] / n for x, n in zip(I_tots, Ns)]
axes1[2].plot(Ns, I_norm, 'o-', color='seagreen')
axes1[2].set_xlabel("$N$ (traps)"); axes1[2].set_ylabel("$\\langle I\\rangle / \\langle I\\rangle_{N=4}$")
axes1[2].set_title("Mean per-trap intensity (efficiency proxy)", fontsize=9)

for ax in axes1: ax.tick_params(labelsize=8)
plt.tight_layout()
fig1.savefig("/mnt/user-data/outputs/ex11_2_7_fig1_N_sweep.png", dpi=150)
print("Saved fig1")

# ── Study 2: metrics vs spacing (N=100 fixed) ─────────────────────────────────
print("\nStudy 2: metrics vs spacing (N=100, square)...")
spacings = [4, 6, 8, 10, 14, 20]
us2, sgs2, I_tots2, sp_used = [], [], [], []

for sp in spacings:
    bx, by = square_grid(10, sp)
    if np.abs(bx).max() >= N_slm // 2: continue
    phi = phi_SGL(bx, by)
    I   = focal_raw(phi)
    In  = read_traps(I, bx, by)
    Im, u, sg = metrics(In)
    us2.append(u); sgs2.append(sg); I_tots2.append(Im); sp_used.append(sp)
    print(f"  spacing={sp:3d}: u={u:.4f}  sigma={sg:.1f}%")

fig2, axes2 = plt.subplots(1, 3, figsize=(13, 4))
fig2.suptitle("SGL: metrics vs trap spacing ($N=100$ fixed)", fontsize=11)

axes2[0].plot(sp_used, us2, 'o-', color='steelblue')
axes2[0].set_xlabel("Spacing (bins)"); axes2[0].set_ylabel("Uniformity $u$")
axes2[0].set_ylim(0, 1.05)

axes2[1].plot(sp_used, sgs2, 'o-', color='tomato')
axes2[1].set_xlabel("Spacing (bins)"); axes2[1].set_ylabel("$\\sigma$ (%)")

axes2[2].plot(sp_used, [x/I_tots2[0] for x in I_tots2], 'o-', color='seagreen')
axes2[2].set_xlabel("Spacing (bins)"); axes2[2].set_ylabel("$\\langle I\\rangle$ (norm.)")

for ax in axes2: ax.tick_params(labelsize=8)
plt.tight_layout()
fig2.savefig("/mnt/user-data/outputs/ex11_2_7_fig2_spacing_sweep.png", dpi=150)
print("Saved fig2")

# ── Study 3: geometry comparison (square vs hex vs random, N~100) ─────────────
print("\nStudy 3: geometry comparison...")
c   = N_slm // 2
ext = 60

geoms = {
    'Square\n(10x10, sp=8)':    square_grid(10, 8),
    'Hexagonal\n(10x10, sp=8)': hex_grid(10, 8),
    'Random\n(N=100, r=45)':    random_grid(100, 45),
}

fig3, axes3 = plt.subplots(2, 3, figsize=(13, 8))
fig3.suptitle("SGL: lattice geometry comparison", fontsize=12)

for col, (name, (bx, by)) in enumerate(geoms.items()):
    # clip any traps outside Nyquist
    ok = (np.abs(bx) < N_slm//2) & (np.abs(by) < N_slm//2)
    bx, by = bx[ok], by[ok]

    phi = phi_SGL(bx, by)
    I   = focal_raw(phi)
    In  = read_traps(I, bx, by)
    Im, u, sg = metrics(In)
    print(f"  {name.split(chr(10))[0]:12s}: N={len(bx):3d}  u={u:.4f}  sigma={sg:.1f}%")

    # top: trap positions
    axes3[0, col].scatter(bx, by, s=12, color='steelblue')
    axes3[0, col].set_title(name, fontsize=9)
    axes3[0, col].set_xlabel("bin x"); axes3[0, col].set_ylabel("bin y")
    axes3[0, col].set_xlim(-ext, ext); axes3[0, col].set_ylim(-ext, ext)
    axes3[0, col].set_aspect('equal')

    # bottom: focal plane
    axes3[1, col].imshow(display_norm(I)[c-ext:c+ext, c-ext:c+ext],
                         origin='lower', extent=[-ext,ext,-ext,ext],
                         cmap='inferno', vmin=0, vmax=1)
    axes3[1, col].set_xlabel("bin x"); axes3[1, col].set_ylabel("bin y")
    axes3[1, col].set_title(f"$u={u:.3f}$  $\\sigma={sg:.0f}\\%$", fontsize=9)

plt.tight_layout()
fig3.savefig("/mnt/user-data/outputs/ex11_2_7_fig3_geometry.png", dpi=150)
print("Saved fig3")

plt.show()
print("Done.")
