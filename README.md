# Holographic Optical Tweezers-exercises

Computational exercises and problem-set solutions for **Chapter 7: Wavefront Engineering and Holographic Optical Tweezers** from:

> Jones, Maragò & Volpe, *Optical Tweezers: Principles and Applications*, Cambridge University Press (2015)

Completed as part of **SSPC 2026** under Dr Vijayakumar Chikkadi.

**Author:** Sneha Chakrabarti, IISER Kolkata (3rd Year BS-MS)

---

## Repository structure

```
hot-exercises/
│
├── ex11_2_11/   Standard Gerchberg-Saxton algorithm
├── ex11_2_12/   Sparse GS — field evaluated only at trap sites
├── ex11_2_13/   Weighted GS — adaptive target for uniformity
├── ex11_2_14/   Multi-plane and generalised 3D GS
├── ex11_2_15/   Adaptive-Additive (AA) algorithm
├── ex11_2_16/   Direct search for optimal focusing (Cizmar 2010)
├── ex11_2_17/   Zernike aberration correction
│
├── ex11_3_1/    Laguerre-Gaussian beam generation
├── ex11_3_2/    Hermite-Gaussian beam generation
├── ex11_3_4/    Counter-rotating optical traps
│
├── ex11_4_1/    Continuous optical potentials — GS (single & multi-plane)
├── ex11_4_2/    Proof: GS convergence error C is non-increasing
├── ex11_4_3/    Continuous optical potentials — AA (single & multi-plane)
│
├── prob11_1/    Problem: Genetic algorithm for hologram generation
├── prob11_2/    Problem: Optical vortex pair — merging and topology
├── prob11_3/    Problem: Fractional optical vortices (Berry 2004)
│
├── README.md
├── report.tex / report.pdf
├── requirements.txt
└── .gitignore
```

Each folder contains one Python script and a `figures/` subdirectory of output plots. Scripts import shared helpers from `ex11_2_11/gs_algorithm.py`; run everything from the repo root.

---

## Physical background

Holographic optical tweezers (HOT) encode a computer-generated hologram on a spatial light modulator (SLM) conjugated to the back focal plane of a microscope objective. The trapping plane sees the Fourier transform of the SLM exit field — so hologram design is a **Fourier phase-retrieval problem**: find the SLM phase that produces a desired intensity at the focal plane.

The normalised complex efficiency at trap site `n` is:

```
V(xn, yn, zn) = (1/M) * sum_m  exp(i [phi_m - Delta_m(xn, yn, zn)])
```

where the steering phase `Delta_m` encodes a lateral blazed grating (linear ramp) and axial Fresnel lens (quadratic). Trap power: `Pn / Ptotal = |Vn|^2 <= 1`.

### Standard benchmark (all 2D exercises)

| Parameter | Value |
|-----------|-------|
| SLM grid | 1000 x 1000 pixels |
| Trap geometry | N = 100, 10x10 square lattice, z = 0 |
| Laser | Gaussian, 1/e^2 radius = 400 px |
| Seed | `numpy.random.default_rng(42)` |
| Lattice margin | 15% from SLM edge |

### Performance metrics

| Metric | Formula | Target |
|--------|---------|--------|
| Total intensity | `I_tot = sum(In)` | high |
| Mean intensity | `<I> = I_tot / N` | high |
| Uniformity | `u = 1 - (max - min) / (max + min)` | → 1 |
| % std error | `sigma = 100 * std(In) / <I>` | → 0 |

### Convergence error (continuous potentials)

```
C = sum_{x,y}  (|E_focus(x,y)| - A_target(x,y))^2
```

Full mathematical details with properly typeset equations are in `report.pdf`.

---

## Exercises

### Section 11.2 — Multiple-trap algorithms

---

#### Ex 11.2.11 — Standard Gerchberg-Saxton

**File:** `ex11_2_11/gs_algorithm.py`
Also exports `gaussian_laser`, `make_trap_mask`, `trap_intensities`, `metrics`, `gerchberg_saxton` — used as helpers by all subsequent exercises.

**Algorithm (per iteration):**
1. Forward FFT → focal field
2. Replace amplitude at each trap site with `sqrt(I_target)`, keep focal phase
3. Inverse FFT → SLM plane
4. Replace SLM amplitude with Gaussian laser profile, keep phase

| Parameter | Value | Reason |
|-----------|-------|--------|
| Grid | 1000 x 1000 | Standard benchmark |
| Iterations | 60 | Convergence well within 60 for 10x10 lattice |
| Laser w | 400 px | ~80% aperture fill |
| Margin | 15% | Avoids boundary aliasing |
| FFT convention | `fftshift(fft2(ifftshift()))` | Centres DC at pixel (Mx/2, My/2) |

**Figures:** convergence (4 metrics), SLM hologram phase, focal intensity (log scale), per-trap bar chart.

---

#### Ex 11.2.12 — Sparse GS

**File:** `ex11_2_12/gs_sparse.py`

For a point-trap target, all non-trap focal pixels are zeroed before back-propagation — computing them via FFT is wasted work. Sparse GS replaces the full FFT with a direct DFT matrix evaluated only at the N trap frequencies:

```
E_focal[n] = W @ slm_field.ravel()
W[n, m]    = exp(-i * 2*pi * (mx*nx/Mx + my*ny/My))
```

Back-propagation uses the adjoint `W^H`.

**Complexity crossover:** sparse GS is faster than FFT only when `N < log2(M) ≈ 20` for a 1000x1000 grid. At N=100 the FFT wins in FLOPs, but the DFT approach is exact and alias-free.

| Parameter | Value | Reason |
|-----------|-------|--------|
| Convergence grid | 1000 x 1000 | Standard benchmark |
| Timing study grid | 512 x 512 | Full DFT matrix at 1000x1000 is ~1.6 GB |
| Timing iterations | 20 | Enough to measure per-iteration cost |
| N sweep | 4, 9, 25, 49, 100 | Perfect squares for clean lattices |

---

#### Ex 11.2.13 — Weighted GS

**File:** `ex11_2_13/gs_weighted.py`

GS cannot simultaneously enforce amplitude constraints in two planes — some traps are systematically dim. Weighted GS compensates by rescaling the per-trap target amplitude after each forward pass:

```
A_target[k](n)  =  A_target[k-1](n)  *  (<I> / In[k-1])^alpha
```

Traps that are too weak get a boosted target; the algorithm over-corrects for them on the next iteration.

| Parameter | Value | Reason |
|-----------|-------|--------|
| Grid | 1000 x 1000 | Standard benchmark |
| Iterations | 80 | Extra 20 vs GS for weight adaptation |
| alpha values | 0.2, 0.5, 1.0 | Slow / moderate / full correction |
| Weight normalisation | Geometric mean fixed | Prevents exponential weight growth |

---

#### Ex 11.2.14 — Multi-plane and 3D GS

**File:** `ex11_2_14/gs_3d.py`

**Grid:** 256 x 256 (reduced from 1000 x 1000 — 3D GS requires one FFT pair per plane per iteration; P=3 planes x 50 iterations = 300 FFT pairs on CPU).

Propagation to an axially displaced plane uses the paraxial Fresnel transfer function:

```
H(kx, ky; z) = exp(i*pi*z * (kx^2 + ky^2) / (Mx*My))
```

Per iteration: propagate SLM field to each plane, replace amplitude, back-propagate, then **sum all back-propagated fields at the SLM** before applying the amplitude constraint. This sum is the key step — one hologram shapes the field at all planes simultaneously.

| Parameter | Value | Reason |
|-----------|-------|--------|
| Grid | 256 x 256 | CPU tractability |
| Multi-plane demo | P=3 planes, z_norm in {-0.06, 0, +0.06}, 3x3 traps each | Mild defocus within Rayleigh range |
| 3D cube demo | 8 traps on cube vertices, Dz=0.04 | Symmetric geometry |

---

#### Ex 11.2.15 — Adaptive-Additive Algorithm

**File:** `ex11_2_15/aa_algorithm.py`

Extends GS with mixing parameter `a` in (0, 1]. Instead of hard-replacing the focal amplitude, mix a fraction `a` of the target into the current field:

```
E_new[n]  =  (1-a) * E_current[n]  +  a * A_target * exp(i * phi_current[n])
```

Setting `a=1` reduces exactly to standard GS.

| Parameter | Value | Reason |
|-----------|-------|--------|
| Grid | 1000 x 1000 | Standard benchmark |
| Iterations | 60 | Same as GS for fair comparison |
| a values tested | 0.1, 0.25, 0.5, 0.75, 1.0 | Spans slow-converging to GS-equivalent |
| Shared seed | `rng(42)` | Isolates algorithmic differences |

**Key result:** for the standard 10x10 lattice, `a >= 0.5` converges to `u=1`, `sigma=0` within 60 iterations.

---

#### Ex 11.2.16 — Direct Search for Optimal Focusing (Cizmar 2010)

**File:** `ex11_2_16/ds_optimal_focus.py`

The SLM is divided into `S = Nseg^2` macro-segments, each corresponding to one plane-wave mode. Optimal focus requires all modes to arrive at the target point co-phasally. The direct search algorithm (Cizmar et al. 2010) cycles each segment through K grey levels and retains the phase that maximises intensity at the target — no wavefront sensor needed.

Simulated annealing (Metropolis criterion) is added to allow escape from local optima.

| Parameter | Value | Reason |
|-----------|-------|--------|
| Grid | 256 x 256 | DS cost = Nseg^2 x K x Nsweeps FFTs; 1000x1000 too slow on CPU |
| Segments | 8 x 8 = 64 | Coarse but demonstrates the principle |
| Grey levels K | 16 | Sufficient phase resolution |
| Sweeps | 6 deterministic + 3 annealing | Metropolis acceptance for escape |
| Aberration | Random Zernike, RMS ~1.5 rad | Realistic wavefront distortion |

**Result:** 94% recovery of diffraction-limited intensity without a wavefront sensor.

---

#### Ex 11.2.17 — Zernike Aberration Correction

**File:** `ex11_2_17/zernike_correction.py`

Any smooth aberration can be expanded in the orthonormal Zernike basis. The conjugate correction is added to the SLM hologram modulo 2pi:

```
phi_corrected  =  [phi_hologram  -  sum_j  cj * Zj(rho, phi)]  mod 2*pi
```

| Parameter | Value |
|-----------|-------|
| Grid | 1000 x 1000 |
| Aberration | Defocus (c=1.2), oblique astig (0.8), vertical astig (-0.6), h-coma (0.5), v-coma (-0.4), spherical (0.3) |
| Zernike terms | (n,m): (2,0), (2,±2), (3,±1), (4,0) — Noll j=4 to 11 |

**Key result:** Strehl ratio recovers from 0.215 (aberrated, uncorrected) to 1.000 (fully corrected). Progressive correction is monotonically improving — each Zernike term added to the hologram increases the Strehl.

**Note:** `zernike_aberration` in ex11.2.16 uses unnormalised radial polynomials; `zernike_map` in ex11.2.17 uses orthonormal Zernike polynomials. Both are internally self-consistent but should not be mixed across exercises.

---

### Section 11.3 — Structured beams

---

#### Ex 11.3.1 — Laguerre-Gaussian Beams

**File:** `ex11_3_1/lg_beams.py`

The SLM phase mask for LG_p^l (Eq. 12, theory notes):

```
phi(rho, phi)  =  l * phi  +  pi * Heaviside{ L_p^|l|(2*rho^2 / w0^2) }
```

The azimuthal term `l*phi` creates the helical wavefront and phase singularity. The Heaviside term encodes pi phase flips across the p zeros of the Laguerre polynomial, producing p+1 concentric rings. A blazed grating is superimposed to shift the beam off the DC spot.

| Parameter | Value |
|-----------|-------|
| Grid | 1000 x 1000 |
| Beam waist w0 | 300 px |
| Grating period | 30 px |
| Cases | (l, p) in {(±1,0), (±2,0), (±3,0), (1,1), (3,2)} |

---

#### Ex 11.3.2 — Hermite-Gaussian Beams

**File:** `ex11_3_2/hg_beams.py`

The HG field at the beam waist is `E_G * Hmx(sqrt(2)*x/w0) * Hmy(sqrt(2)*y/w0)`. The corresponding SLM phase mask is:

```
phi(x, y)  =  pi * Heaviside{ Hmx(sqrt(2)*x/w0) * Hmy(sqrt(2)*y/w0) }
```

This is a binary (0 or pi) mask matching the nodal sign structure of the HG mode. The product `Hmx * Hmy` partitions the pupil into `(mx+1)*(my+1)` rectangular lobes of alternating sign.

| Parameter | Value |
|-----------|-------|
| Grid | 1000 x 1000 |
| w0 | 250 px |
| Modes shown | All (mx, my) with mx + my <= 4 |

**Verification:** focal-plane cross-sections confirm mx nodal lines in x and my nodal lines in y.

---

#### Ex 11.3.4 — Counter-Rotating Optical Traps

**File:** `ex11_3_4/counter_rotating_traps.py`

Multiple LG beams at distinct focal positions with charges `ln` are combined via the SGL algorithm:

```
phi_SGL  =  arg( sum_n  exp(i * phi_n) )
```

where each `phi_n` includes the LG phase and a lateral steering grating. Two LG beams with charges `+l` and `-l` at symmetric positions create a counter-rotating trap pair — one particle orbits clockwise, the other counterclockwise. The OAM sign is directly visible in the focal-field phase as opposite winding directions.

| Parameter | Value |
|-----------|-------|
| Grid | 1000 x 1000 |
| w0 | 300 px |
| Trap separation | 150 px |
| Configurations | Pairs (l=1,2,3); 2x2 quad (l=±2) |

---

### Section 11.4 — Continuous optical potentials

---

#### Ex 11.4.1 — Continuous Potentials via GS

**File:** `ex11_4_1/continuous_gs.py`

The only change from point-trap GS: amplitude replacement in step 2 is applied over the **entire** focal field, not just N discrete sites. The target amplitude is normalised via Parseval's theorem so both amplitudes are on the same absolute scale:

```
A_t  =  A_target  *  sqrt(||E_SLM||^2 * Mx*My  /  ||A_target||^2)
```

This fixed-scale normalisation is required for the projection theorem to guarantee monotone convergence.

| Parameter | Value | Reason |
|-----------|-------|--------|
| Single-plane grid | 1000 x 1000 | Standard benchmark |
| Single-plane iterations | 80 | More needed for continuous vs discrete |
| Multi-plane grid | 256 x 256 | CPU tractability (3 planes x 60 iters) |
| Target patterns | Ring, lemniscate, cross, phi symbol | Variety of spatial bandwidths |

---

#### Ex 11.4.2 — Proof: C is Non-Increasing Under GS

**File:** `ex11_4_2/gs_convergence_proof.py`

Each GS iteration consists of two alternating projections:
- **Step 2:** project focal field onto the target amplitude set — sets C = 0 exactly.
- **Step 4:** project back onto the SLM amplitude set via IFFT + laser constraint.

By the alternating-projections theorem, the distance to the intersection of two closed convex sets is non-increasing. This guarantees `C[k+1] <= C[k]` at every iteration.

**Numerical verification:** zero violations across 4 target patterns and 3 random seeds, 100 iterations each (1000 x 1000 SLM). The Parseval normalisation of `A_t` is essential — a per-iteration rescaling breaks the projection property and introduces violations.

| Parameter | Value |
|-----------|-------|
| Grid | 1000 x 1000 |
| Iterations | 100 |
| Seeds | 3 |
| Targets | Ring, cross, square frame, lemniscate |

---

#### Ex 11.4.3 — Continuous Potentials via AA

**File:** `ex11_4_3/continuous_aa.py`

AA mixing applied pixelwise over the full focal field:

```
E_new(x, y)  =  (1-a) * E_current(x, y)  +  a * A_t(x, y) * exp(i * phi_current(x, y))
```

| Parameter | Value |
|-----------|-------|
| Single-plane grid | 1000 x 1000 |
| Iterations | 60 |
| a values | 0.2, 0.5, 0.75, 1.0 |
| Multi-plane grid | 256 x 256 |
| Multi-plane iterations | 40 |

**Key result:** for continuous potentials, GS (a=1) converges to lower C than AA with a<1 at equal iteration count. This is the opposite of the discrete trap case — the continuous projection covers every focal pixel so there are no local traps for mixing to help escape.

---

### Problem set

---

#### Problem 11.1 — Genetic Algorithm

**File:** `prob11_1/genetic_algorithm.py`

A population-based GA evolves SLM phase masks toward higher fitness `F_gain = <I> - w*sigma` (w=0.5) using tournament selection, pixel-wise crossover, and decaying-rate mutation.

| Algorithm | u | sigma (%) | <I> | Wall time |
|-----------|---|-----------|-----|-----------|
| GA (P=20, 50 gen) | 0.005 | 69 | 2.57 | 100 s |
| GS (50 iter) | 1.000 | 0.00 | 0.010 | 6 s |
| AA a=0.5 (50 iter) | 1.000 | 0.00 | 0.010 | 6 s |

GS achieves perfect uniformity 17x faster. The GA maximises `F_gain` but at w=0.5 under-penalises non-uniformity. GA's natural advantage — escaping local optima — is not needed here because the GS projection landscape for this lattice is essentially convex.

---

#### Problem 11.2 — Optical Vortex Pair

**File:** `prob11_2/optical_vortex_pair.py`

Two vortices with charges `l1`, `l2` embedded in a Gaussian beam at positions `(+d, 0)` and `(-d, 0)`:

```
E(x, y)  =  E_G(x, y)  *  exp(i*l1*atan2(y, x-d))  *  exp(i*l2*atan2(y, x+d))
```

Vortex positions are detected by scanning for non-zero phase winding numbers on a grid.

| Configuration | Behaviour as d → 0 | Total charge Q |
|---|---|---|
| Same-sign (+l, +l) | Singularities merge into one charge-2l vortex | Q = +2 always |
| Opposite-sign (+l, -l) | Singularities annihilate; no dark core remains | Q = 0 always |

**Answer:** total topological charge is conserved as vortices move toward the high-intensity region. It is a topological invariant of a smooth field and cannot change unless a vortex leaves the aperture or a vortex-antivortex pair is created or annihilated.

---

#### Problem 11.3 — Fractional Optical Vortices (Berry 2004)

**File:** `prob11_3/fractional_vortex.py`

For non-integer `l = n + eps` (0 < eps < 1), the SLM phase mask:

```
phi_SLM(x, y)  =  [ l * atan2(y, x) ]  mod  2*pi
```

is discontinuous along the `+x` axis (branch cut) with a phase jump of `2*pi*eps`. Rather than a single fractional-charge vortex, Berry (2004) showed the beam contains a **chain of unit-charge (+1) vortices** along the branch cut. As eps increases from 0 to 1, vortices appear near the beam centre and migrate outward.

Simulation reproduces all features reported by Leach et al. (2004) and Lee et al. (2004):
- Asymmetric ring intensity with a gap along the branch cut for fractional l
- Chain of unit-charge vortices detectable in the focal-field phase
- Vortex count approximately floor(l) + 1 at half-integer l
- Symmetric doughnut restored at integer l

---

## Dependencies

```
numpy >= 1.24
matplotlib >= 3.7
scipy >= 1.10
```

```bash
pip install -r requirements.txt
```

---

## Running

Run from the **repo root** so that relative imports resolve correctly.

```bash
# Section 11.2
python ex11_2_11/gs_algorithm.py         # ~8 s
python ex11_2_12/gs_sparse.py            # ~35 s
python ex11_2_13/gs_weighted.py          # ~30 s
python ex11_2_14/gs_3d.py               # ~20 s  (256x256 grid)
python ex11_2_15/aa_algorithm.py         # ~25 s
python ex11_2_16/ds_optimal_focus.py     # ~90 s  (DS sweeps)
python ex11_2_17/zernike_correction.py   # ~15 s

# Section 11.3
python ex11_3_1/lg_beams.py              # ~10 s
python ex11_3_2/hg_beams.py              # ~10 s
python ex11_3_4/counter_rotating_traps.py  # ~15 s

# Section 11.4
python ex11_4_1/continuous_gs.py         # ~50 s
python ex11_4_2/gs_convergence_proof.py  # ~90 s
python ex11_4_3/continuous_aa.py         # ~90 s

# Problem set
python prob11_1/genetic_algorithm.py     # ~115 s  (GA is slow by design)
python prob11_2/optical_vortex_pair.py   # ~30 s
python prob11_3/fractional_vortex.py     # ~60 s
```

All figures are saved to `<exercise_folder>/figures/`.

---

## Attribution

All algorithms and physical models are based on or derived from:

> Jones, P. H., Maragò, O. M., & Volpe, G. (2015). *Optical Tweezers: Principles and Applications*. Cambridge University Press.

Additional references:
- Cizmar, T. et al. (2010). In situ wavefront correction and its application to micromanipulation. *Nature Photonics* 4, 388–394.
- Berry, M. V. (2004). Optical vortices evolving from helicoidal integer and fractional phase steps. *J. Opt. A* 6, 259–268.
- Leach, J. et al. (2004). Vortex knots in light. *New J. Phys.* 7, 55.
- Lee, W. M. et al. (2004). Observation of three-dimensional optical trapping. *Opt. Lett.* 29, 2467.
