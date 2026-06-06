"""
Problem 11.1
Genetic Algorithm for Hologram Generation.

Reference: Jones, Maragò & Volpe, Optical Tweezers: Principles and Applications, Ch. 7
Notes: SSPC 2026, Sneha Chakrabarti

Overview
--------
A genetic algorithm (GA) maintains a population of candidate SLM phase masks
and evolves them toward higher fitness using operators from evolutionary biology:
selection, crossover, and mutation. The fitness function can target either:

  (a) Point-trap arrays: fitness = F_gain = <I> - w*sigma   (Eq. 11, theory notes)
  (b) Continuous potentials: fitness = -C  where C = sum(|E_focus| - A_target)^2

GA is compared against GS and AA on the standard benchmark.

Algorithm
---------
Initialisation:
  Generate P random SLM phase masks (population).

Per generation:
  1. Evaluate: compute fitness F_i for each individual i.
  2. Select: retain the top-K elite individuals unchanged ("elitism").
     Draw parents for the remaining P-K slots using tournament selection.
  3. Crossover: combine two parent phase masks. Two modes:
     - Pixel crossover: each pixel taken from parent A with prob 0.5,
       else from parent B.
     - Segment crossover: SLM divided into N_seg x N_seg blocks; each
       block taken from one parent.
  4. Mutate: each pixel independently flipped to a random grey level
     with probability p_mut. Mutation rate decays over generations
     (simulated annealing schedule).
  5. Repair: enforce phase range [0, 2pi).
  Repeat until max_gen reached or fitness converges.

Comparison with GS / AA / DS
------------------------------
All algorithms run on the same standard benchmark:
  SLM  : 1000 x 1000 pixels
  Traps: N = 100, 10x10 lattice, z = 0
  Seed : fixed (rng(42))
  Metric: uniformity u and sigma tracked per iteration/generation.

GA is slower per generation than GS but can escape local optima that
GS stagnates in. The comparison shows the trade-off between computational
cost and solution quality.

Implementation notes
--------------------
Full 1000x1000 population-based GA is extremely expensive: each fitness
evaluation requires one FFT (~1 ms); a population of 30 over 50 generations
= 1500 FFTs + overhead. We use:
  - Population size: P = 20
  - Generations: 80
  - Elite: K = 4
  - Tournament size: 3
  - Mutation rate: p_mut decays from 0.05 to 0.005
  - Grey levels: 32 (5-bit discretisation)
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).parent.parent / "ex11_2_11"))
from gs_algorithm import (gaussian_laser, make_trap_mask, trap_intensities,
                           metrics, gerchberg_saxton)
sys.path.insert(0, str(Path(__file__).parent.parent / "ex11_2_15"))
from aa_algorithm import adaptive_additive

rng = np.random.default_rng(42)

# ── standard benchmark parameters ────────────────────────────────────────────
Mx, My   = 1000, 1000
LASER_W  = 400
MARGIN   = 0.15
N_GREY   = 32          # phase grey levels (5-bit)

# ── GA hyperparameters ────────────────────────────────────────────────────────
POP_SIZE  = 20
N_GEN     = 50         # generations (reduced for speed)
N_ELITE   = 4
TOURN_K   = 3
P_MUT_0   = 0.04
P_MUT_END = 0.004
W_GAIN    = 0.5

N_ITER_COMPARE = 50    # GS/AA iterations for comparison

FIGURES = Path(__file__).parent / "figures"
FIGURES.mkdir(exist_ok=True)

GREY_LEVELS = np.linspace(0, 2 * np.pi, N_GREY, endpoint=False)


# ── fitness function ──────────────────────────────────────────────────────────

def fitness_traps(slm_phase, laser_amp, trap_coords, w=W_GAIN):
    """
    F_gain = <I> - w * sigma  (Eq. 11, theory notes, Section 5.2).
    Higher is better.
    """
    slm_field   = laser_amp * np.exp(1j * slm_phase)
    focal_field = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(slm_field)))
    I_traps     = trap_intensities(focal_field, trap_coords)
    _, I_mean, u, sigma = metrics(I_traps)
    return I_mean - w * sigma / 100.0, u, sigma


# ── GA operators ──────────────────────────────────────────────────────────────

def tournament_select(fitnesses, k):
    """Return index of best individual from a random tournament of size k."""
    idx = rng.choice(len(fitnesses), size=k, replace=False)
    return idx[np.argmax(fitnesses[idx])]


def pixel_crossover(parent_a, parent_b):
    """Each pixel independently from A (prob 0.5) or B."""
    mask = rng.random(parent_a.shape) < 0.5
    child = np.where(mask, parent_a, parent_b)
    return child


def segment_crossover(parent_a, parent_b, n_seg=10):
    """Each n_seg x n_seg block taken from A or B randomly."""
    Mx, My = parent_a.shape
    bx, by = Mx // n_seg, My // n_seg
    child  = parent_a.copy()
    for i in range(n_seg):
        for j in range(n_seg):
            if rng.random() < 0.5:
                child[i*bx:(i+1)*bx, j*by:(j+1)*by] = \
                    parent_b[i*bx:(i+1)*bx, j*by:(j+1)*by]
    return child


def mutate(individual, p_mut):
    """
    Each pixel independently replaced by a random grey level with prob p_mut.
    Grey levels discretised to N_GREY values in [0, 2pi).
    """
    mask    = rng.random(individual.shape) < p_mut
    randoms = rng.choice(GREY_LEVELS, size=individual.shape)
    return np.where(mask, randoms, individual)


# ── genetic algorithm ─────────────────────────────────────────────────────────

def genetic_algorithm(laser_amp, trap_coords, pop_size, n_gen,
                      n_elite, tourn_k, p_mut_0, p_mut_end,
                      crossover='pixel'):
    """
    GA for hologram optimisation targeting point-trap arrays.

    Parameters
    ----------
    laser_amp   : (Mx, My) SLM amplitude
    trap_coords : (N, 2) trap pixel coords
    pop_size    : population size P
    n_gen       : number of generations
    n_elite     : number of elite individuals kept each generation
    tourn_k     : tournament size for parent selection
    p_mut_0     : initial mutation probability
    p_mut_end   : final mutation probability
    crossover   : 'pixel' or 'segment'

    Returns
    -------
    best_phase : (Mx, My) best hologram phase found
    history    : dict with per-generation 'best_F', 'mean_F', 'u', 'sigma'
    """
    Mx, My = laser_amp.shape

    # initialise population with random grey-level phases
    population = np.array([
        rng.choice(GREY_LEVELS, size=(Mx, My))
        for _ in range(pop_size)
    ])

    history = {'best_F': [], 'mean_F': [], 'u': [], 'sigma': []}
    best_phase  = population[0].copy()
    best_F_ever = -np.inf

    for gen in range(n_gen):
        # ── evaluate fitness ──────────────────────────────────────────────────
        F_vals  = np.zeros(pop_size)
        u_vals  = np.zeros(pop_size)
        sig_vals = np.zeros(pop_size)
        for i, ind in enumerate(population):
            F, u, sig       = fitness_traps(ind, laser_amp, trap_coords)
            F_vals[i]       = F
            u_vals[i]       = u
            sig_vals[i]     = sig

        # track best ever
        best_idx = np.argmax(F_vals)
        if F_vals[best_idx] > best_F_ever:
            best_F_ever = F_vals[best_idx]
            best_phase  = population[best_idx].copy()

        history['best_F'].append(best_F_ever)
        history['mean_F'].append(float(F_vals.mean()))
        history['u'].append(float(u_vals[best_idx]))
        history['sigma'].append(float(sig_vals[best_idx]))

        if (gen + 1) % 10 == 0:
            print(f"    gen {gen+1:3d}: best_F={best_F_ever:.4f}, "
                  f"u={u_vals[best_idx]:.3f}, σ={sig_vals[best_idx]:.1f}%")

        # ── mutation rate schedule ────────────────────────────────────────────
        frac  = gen / max(n_gen - 1, 1)
        p_mut = p_mut_0 * (p_mut_end / p_mut_0) ** frac

        # ── build next generation ─────────────────────────────────────────────
        # sort descending by fitness
        order      = np.argsort(F_vals)[::-1]
        new_pop    = [population[i].copy() for i in order[:n_elite]]

        while len(new_pop) < pop_size:
            pa = tournament_select(F_vals, tourn_k)
            pb = tournament_select(F_vals, tourn_k)
            if crossover == 'pixel':
                child = pixel_crossover(population[pa], population[pb])
            else:
                child = segment_crossover(population[pa], population[pb])
            child = mutate(child, p_mut)
            new_pop.append(child)

        population = np.array(new_pop)

    return best_phase, history


# ── run and plot ──────────────────────────────────────────────────────────────

def run_and_plot():
    print(f"Setting up {Mx}×{My} SLM, N=100 traps...")
    laser_amp   = gaussian_laser(Mx, My, LASER_W)
    trap_coords = make_trap_mask(Mx, My, N_side=10, margin=MARGIN)
    N           = len(trap_coords)
    I_target    = 1.0 / N
    seed_phase  = rng.uniform(0, 2 * np.pi, (Mx, My))

    # ── run GA (pixel crossover) ──────────────────────────────────────────────
    print(f"Running GA (pixel crossover, P={POP_SIZE}, {N_GEN} gen)...")
    t0 = time.perf_counter()
    best_phase_ga, hist_ga = genetic_algorithm(
        laser_amp, trap_coords,
        POP_SIZE, N_GEN, N_ELITE, TOURN_K, P_MUT_0, P_MUT_END,
        crossover='pixel'
    )
    t_ga = time.perf_counter() - t0
    print(f"  GA done in {t_ga:.1f}s")

    # ── run GS for comparison ─────────────────────────────────────────────────
    print(f"Running GS ({N_ITER_COMPARE} iterations)...")
    t0 = time.perf_counter()
    slm_gs, hist_gs = gerchberg_saxton(
        laser_amp, trap_coords, I_target, N_ITER_COMPARE,
        seed_phase=seed_phase.copy()
    )
    t_gs = time.perf_counter() - t0
    print(f"  GS done in {t_gs:.1f}s")

    # ── run AA for comparison ─────────────────────────────────────────────────
    print(f"Running AA a=0.5 ({N_ITER_COMPARE} iterations)...")
    t0 = time.perf_counter()
    slm_aa, hist_aa = adaptive_additive(
        laser_amp, trap_coords, I_target, N_ITER_COMPARE,
        a=0.5, seed_phase=seed_phase.copy()
    )
    t_aa = time.perf_counter() - t0
    print(f"  AA done in {t_aa:.1f}s")

    # ── final metrics for all ─────────────────────────────────────────────────
    def final_metrics(slm_phase):
        sf = laser_amp * np.exp(1j * slm_phase)
        ff = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(sf)))
        I  = trap_intensities(ff, trap_coords)
        return metrics(I), ff

    (It_ga,  Im_ga,  u_ga,  s_ga),  ff_ga  = final_metrics(best_phase_ga)
    (It_gs,  Im_gs,  u_gs,  s_gs),  ff_gs  = final_metrics(slm_gs)
    (It_aa,  Im_aa,  u_aa,  s_aa),  ff_aa  = final_metrics(slm_aa)

    print(f"\n{'':─<68}")
    print(f"{'Algorithm':<22} {'u':>8} {'σ (%)':>8} {'<I>':>12} {'Time(s)':>9}")
    print(f"{'':─<68}")
    for label, u, s, Im, t in [
        ("GA (pixel xover)",   u_ga,  s_ga,  Im_ga,  t_ga),
        ("GS",                 u_gs,  s_gs,  Im_gs,  t_gs),
        ("AA (a=0.5)",         u_aa,  s_aa,  Im_aa,  t_aa),
    ]:
        print(f"{label:<22} {u:>8.4f} {s:>8.2f} {Im:>12.6f} {t:>9.1f}")
    print(f"{'':─<68}")

    gens  = np.arange(1, N_GEN + 1)
    iters = np.arange(1, N_ITER_COMPARE + 1)
    crop   = 300
    crop_f = 180
    cx, cy = Mx // 2, My // 2
    sl     = np.s_[cx-crop:cx+crop, cy-crop:cy+crop]

    # ── Figure 1: convergence — uniformity ────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(
        f"Genetic Algorithm vs GS vs AA: Convergence\n"
        f"(1000×1000 SLM, N=100 traps, "
        f"GA: P={POP_SIZE} pop, {N_GEN} gen; GS/AA: {N_ITER_COMPARE} iter)",
        fontsize=12, fontweight='bold'
    )
    for ax, key, ylabel in [
        (axes[0], 'u',     r"Uniformity $u$"),
        (axes[1], 'sigma', r"$\sigma$ (%)"),
    ]:
        ax.plot(gens,  hist_ga[key],     lw=1.8, color='darkorange',
                label=f'GA (pixel xover, P={POP_SIZE})')
        ax.plot(iters, hist_gs[key],     lw=1.8, color='steelblue', ls='--',
                label='GS')
        ax.plot(iters, hist_aa[key],     lw=1.8, color='seagreen', ls=':',
                label='AA (a=0.5)')
        ax.set_xlabel("Generation / Iteration", fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(FIGURES / "fig1_ga_convergence.png", dpi=150,
                bbox_inches='tight')
    plt.close(fig)
    print("  fig1 saved.")

    # ── Figure 2: per-trap intensity bars — all algorithms ────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle(
        "Per-Trap Intensity: GA vs GS vs AA\n"
        f"(normalised to mean; {N_GEN} gen / {N_ITER_COMPARE} iter)",
        fontsize=12, fontweight='bold'
    )
    items = [
        (best_phase_ga, ff_ga,  u_ga,  s_ga,  f"GA (pixel xover, P={POP_SIZE}, {N_GEN}gen)", 'darkorange'),
        (slm_gs,        ff_gs,  u_gs,  s_gs,   f"GS ({N_ITER_COMPARE} iter)",                'steelblue'),
        (slm_aa,        ff_aa,  u_aa,  s_aa,   f"AA a=0.5 ({N_ITER_COMPARE} iter)",          'seagreen'),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    for ax, (_, ff, u, s, label, col) in zip(axes.flat, items):
        I = trap_intensities(ff, trap_coords)
        ax.bar(np.arange(N), I / I.mean(), color=col, alpha=0.8, width=1.0)
        ax.axhline(1.0, color='k', lw=1.2, ls='--')
        ax.set_title(f"{label}\nu={u:.4f},  σ={s:.2f}%", fontsize=9)
        ax.set_xlabel("Trap index"); ax.set_ylabel(r"$I_n/\langle I\rangle$")
        ax.set_ylim(0, max(2.2, (I / I.mean()).max() * 1.1))
        ax.grid(True, alpha=0.25, axis='y')
    plt.tight_layout()
    fig.savefig(FIGURES / "fig2_per_trap_comparison.png", dpi=150,
                bbox_inches='tight')
    plt.close(fig)
    print("  fig2 saved.")

    # ── Figure 3: SLM phase holograms ────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    fig.suptitle("SLM Phase Holograms: GA vs GS (central 600×600 crop)",
                 fontsize=12, fontweight='bold')
    for ax, phase, title in [
        (axes[0], best_phase_ga, f"GA (pixel xover, {N_GEN} gen)"),
        (axes[1], slm_gs,        f"GS ({N_ITER_COMPARE} iter)"),
    ]:
        im = ax.imshow(phase[sl].T % (2*np.pi), cmap='hsv',
                       vmin=0, vmax=2*np.pi, origin='lower')
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("SLM pixel x"); ax.set_ylabel("SLM pixel y")
        plt.colorbar(im, ax=ax, label='Phase (rad)', fraction=0.046, pad=0.04)
    plt.tight_layout()
    fig.savefig(FIGURES / "fig3_slm_holograms.png", dpi=150,
                bbox_inches='tight')
    plt.close(fig)
    print("  fig3 saved.")

    # ── Figure 4: focal intensity maps ────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    fig.suptitle("Focal-Plane Intensity (log scale): GA vs GS vs AA",
                 fontsize=12, fontweight='bold')
    for ax, (_, ff, u, s, label, _col) in zip(axes, items):
        intens = np.abs(ff)**2
        log_i  = np.log1p(intens / intens.max() * 1e4)
        sl_f   = np.s_[cx-crop_f:cx+crop_f, cy-crop_f:cy+crop_f]
        ax.imshow(log_i[sl_f].T, cmap='inferno', origin='lower')
        ax.scatter(trap_coords[:, 0] - (cx - crop_f),
                   trap_coords[:, 1] - (cy - crop_f),
                   s=4, c='cyan', marker='x', linewidths=0.6)
        ax.set_title(label.split(' (')[0], fontsize=10)
        ax.set_xlabel("px"); ax.set_ylabel("px")
    plt.tight_layout()
    fig.savefig(FIGURES / "fig4_focal_maps.png", dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("  fig4 saved.")

    print(f"\nFigures saved to {FIGURES}")


if __name__ == "__main__":
    run_and_plot()
