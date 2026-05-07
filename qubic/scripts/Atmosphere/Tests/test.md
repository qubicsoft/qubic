# Wind Reconstruction — Full Pipeline Analysis & Weak-Point Report

---

## 1. Pipeline Overview

The goal is to jointly reconstruct an atmospheric sky map and a 2D constant wind vector
`(wx, wy)` from a QUBIC time-ordered data (TOD) stream. Because the two unknowns depend
on each other non-linearly, an **alternating optimisation** is used: iterate
`wind → maps → wind → maps → …` for `n_loop = 10` outer iterations.

---

## 2. Data and Noise Model

### TOD generation
```
tod = H_tod(atm_maps) + noise
```
- `H_tod`: QUBIC pointing operator built from the *wind-perturbed* sampling `q_sampling_local`
  (true constant wind `[0.3, 0.5]` baked in at the pointing level)
- `atm_maps`: shape `(nsub_in=8, npix, 3)` — multi-frequency atmospheric intensity maps
- `noise`: white-noise realisation from `QubicTotNoise`; shape `(99200,)`, std ~`5e-17`

### Noise inverse operator
```
invN = Pchunk * R_chunk * BlockDiagonal(invN_rec) * R_chunk^T * Pchunk^T
```
`invN_rec[i]` is the per-chunk inverse noise covariance from `get_invntt_operator`.

---

## 3. Forward Model (Reconstruction Side)

The simulated TOD during reconstruction is:
```
tod_sim(wind; maps) = Pchunk * H_ * W(wind) * Amm * P(maps_rec)
```

| Operator | Shape in → out | Role |
|---|---|---|
| `P` (PackOperator.T) | `(49152,) → (1, 49152, 3)` | Unpack masked free parameters |
| `Amm` (DenseOperator) | `(1, npix, 3) → (8, npix, 3)` | Apply mixing matrix: project 1 sky component to 8 freq bands |
| `W(wind)` (BlockColumnOperator) | `(8, npix, 3) → (80, npix, 3)` | Apply per-chunk sky rotation for wind displacement |
| `H_` (BlockDiagonalOperator) | `(80, npix, 3) → (99200,)` | QUBIC pointing + beam (no wind) |
| `Pchunk` (PermuteChunksOperator) | `(99200,) → (99200,)` | Reorder axes: (n_chunks, ndet, N_buffer) → (ndet, Npointings) |

### Wind operator detail
Wind `(wx, wy)` is a scalar-per-axis constant speed (m/step). Converted to cumulative
displacement `wind_x[t] = wx * t`. The 100 pointings are split into `n_chunks = 10`
buffers of `N_buffer = 10` each. Each buffer gets a single **mean** displacement and
applies it as a HEALPix map rotation via `shift_healpy_map`.

---

## 4. Chi2 Objective

```
chi2(wind; maps) = 0.5 * r^T * invN * r,    r = tod - tod_sim(wind; maps)
```

This is `−log L` (the Gaussian negative log-likelihood, up to constants). Under the
correct noise model and at the true parameters:
```
E[chi2_min] ≈ (N_data − n_params) / 2    →    chi2_min / ndf ≈ 0.5
```

Observed values:
- At initialisation (wind=[0,0], maps_rec = scaled true maps): `chi2 ≈ 2.19e15`
- After 1 wind fit + 10 PCG iterations: `chi2_min ≈ 2.63e14`,  `chi2_min / ndf ≈ 2.65e9`

The factor `~5e9` excess over the expected `0.5` is the central quantitative indicator
that the noise model / forward model has a large inconsistency.

---

## 5. Wind Fitting Sub-problem

### Algorithm
L-BFGS-B minimisation of `chi2(wind) / scale` over `wind ∈ [−5, 5]²`, where
`scale = chi2(x0)` (the chi2 at the starting point, making the objective O(1)).

```python
result = minimize(
    lambda x, t: get_chi2(x, t) / scale,
    x0=wind_rec,
    jac=lambda x, t: _chi2_grad(x, t, eps=1e-3, scale=scale),
    method="L-BFGS-B",
    args=tod,
    bounds=[(-5, 5), (-5, 5)],
    options={"ftol": 1e-30, "gtol": 1e-10, "maxiter": 100},
)
```

Jacobian: explicit central differences with `eps = 1e-3`.

### Observed result (iteration 0)
```
x = [0.30096, 0.49256]   (true: [0.3, 0.5])   ← excellent recovery
chi2_min = 2.63e14
chi2_min / ndf = 2.65e9
status: ABNORMAL_TERMINATION_IN_LNSRCH
```

The wind is recovered with < 0.01 error on both components in one call, despite the
abnormal termination.

---

## 6. Map Fitting Sub-problem

### Normal equations
With `wind_op` fixed at the current estimate:
```
H = Pchunk * H_ * wind_op * Amm
A = P^T * H^T * invN * H * P
b = P^T * H^T * invN * (tod − H * x_masked)
```
Solved by `PCGAlgorithm` for `n_iter = 10` iterations.

### Starting point
`maps_rec` carries over from the previous iteration. At `iloop=0` it is the scaled
true map (see Initialisation below).

---

## 7. Initialisation

```python
maps_rec = P.T(true_maps) / mixing_matrix[:, 1].mean()
wind_rec  = (0, 0)
```

**`maps_rec` is initialised from the true sky map.** This is intentional in this
simulation context: it ensures that the chi2 surface at iteration 0 has its minimum
at the true wind, so the wind fit succeeds immediately. For real data this initialisation
would not be available.

---

## 8. Outer Loop

```
for iloop in range(10):
    wind_rec, sigma2_eff = fit_wind(tod)          # L-BFGS-B
    H = Pchunk * H_ * wind_op * Amm
    maps_rec = PCG(A, b, maps_rec, n_iter=10)     # CG solve
```

---

## 9. Weak Points

### 9.1 — chi2/ndf >> 1: the noise model is wrong (most critical)

After fitting, `chi2_min / ndf ≈ 2.65e9` (expected: `0.5`). The excess is a factor
`~5e9`. This means the forward model does not explain the data at the noise level.

**Root cause candidates:**
- **Forward model mismatch**: The true TOD was generated with a continuous wind
  perturbation at the *pointing level* (`H_tod` uses `q_sampling_local` with real
  pointing deviations). The reconstruction applies wind as a *HEALPix sky rotation*
  (pixelised, chunked). These two are not mathematically identical. The chunking into
  `N_buffer=10` averages the displacement within each buffer — a systematic error.
- **PCG not converged**: Only 10 CG iterations. The convergence plot shows the PCG
  residual is still decreasing at `iloop=0` (last value ~4.8e-3). The map solution
  is approximate, leaving structured residuals.
- **Noise model calibration**: `invN` is built from `get_invntt_operator(wdet, wpho150,
  wpho220)`. If these weights do not match the actual noise in the simulated TOD, the
  chi2 scale is wrong.

**Consequence for error bars**: `sigma2_eff = chi2_min / ndf ≈ 2.65e9` is used to
normalise the chi2 for the PDG Delta-chi2 intervals. The resulting error bars are
statistically correct given this noise model, but they absorb all the model mismatch
into a single rescaling factor. The error bars measure *fitting precision*, not
*systematic accuracy*.

---

### 9.2 — Initialisation with true maps

`maps_rec = P.T(true_maps) / mixing_matrix[:,1].mean()` uses the ground-truth
atmospheric map. In simulation this is fair as a "warm start" benchmark, but it means
the quality of iteration-0 wind recovery is artificially good. In practice the
initialisation would have to come from a blind map-making step without wind
correction, and the iteration-0 chi2 surface minimum might not be at the true wind.

---

### 9.3 — Gradient precision and catastrophic cancellation

From the finite-difference gradient test:

| eps    | grad_wy (unscaled) | grad (scaled) |
|--------|--------------------|---------------|
| 1e-1   | −9.36e67           | −5.44e-1      |
| 1e-3   | −8.84e67           | −5.14e-1      |
| **1e-5** | **0**            | **0**         |
| 1e-8   | 0                  | 0             |

For `eps < 1e-5`, the central difference `chi2(x+eps) − chi2(x−eps)` becomes
smaller than the floating-point rounding noise of `chi2 ~ 1e68` (`~1e52` in absolute
terms), and the gradient collapses to zero. The chosen `eps = 1e-3` avoids this, but:
- The gradient estimate at `eps=1e-3` is less accurate than one at `eps~1e-6` would
  be in a well-conditioned problem.
- L-BFGS-B's internal finite differences (default `eps ~ 1.5e-8`) would give zero
  gradients — hence the explicit `jac=` is essential and correct.
- The large `eps` means the gradient points in the right direction but with some
  noise, which may contribute to the `ABNORMAL_TERMINATION_IN_LNSRCH`.

---

### 9.4 — Abnormal termination of L-BFGS-B

Despite recovering the correct wind, the optimizer exits with
`ABNORMAL_TERMINATION_IN_LNSRCH`. This means the line search failed to find a
sufficient decrease along the search direction. Likely causes:
- The chi2 surface has slight non-smoothness from the pixelised HEALPix rotation
  (discrete map shifts produce tiny discontinuities in chi2 as a function of wind).
- The gradient estimate at `eps=1e-3` is noisy enough to occasionally point slightly
  off from the true descent direction, causing the line search to overstep.

The optimizer still produces a good result (wind error < 1%), but convergence is not
guaranteed in general and will depend on the noise realisation and map quality.

---

### 9.5 — Wind operator approximation (chunking)

The 100 pointings are divided into 10 chunks of 10. Each chunk applies a single mean
wind displacement. The true displacement is cumulative within each chunk, so the mean
is only exact for the middle of the chunk. The error per chunk is:
```
Δ_displacement = wx * (N_buffer / 2) * N_buffer / N_buffer ≈ wx * N_buffer / 2
```

For `wx = 0.3` and `N_buffer = 10`, the maximum displacement error within a chunk is
`~1.5` steps worth of wind. For a small enough pixel scale this is negligible, but at
high signal-to-noise it becomes a systematic floor on chi2/ndf.

---

### 9.6 — Error bars: statistical vs systematic

The profile chi2 error bars and Hessian errors are **statistical** (they describe the
curvature of the chi2 surface around the minimum). Because `chi2_min / ndf >> 1`, these
are rescaled by `sigma2_eff`, which inflates them to be consistent with the data.

However, they do not account for:
- **Systematic bias** from the forward model mismatch (section 9.1, 9.5)
- **Convergence bias** from incomplete PCG (section 9.1)
- **Degeneracy** between wind and map: the alternating optimisation does not explore
  the joint (wind, maps) uncertainty, so the error bars on wind are conditioned on the
  current map estimate and will be underestimated if the wind-map degeneracy is
  significant.

---

### 9.7 — Alternating optimisation convergence

The alternating scheme (block coordinate descent) converges to a stationary point of
the joint chi2 only under certain convexity conditions. Here:
- chi2 is linear in maps (for fixed wind) → the map step always reaches the global
  minimum in the map subspace (given sufficient CG iterations).
- chi2 is non-linear and non-convex in wind (for fixed maps) → the wind step may
  converge to a local minimum, especially in later iterations when maps_rec has drifted
  from the true maps.

With only `n_iter = 10` CG iterations in the map step, the map estimate never fully
converges, which slightly perturbs the chi2 surface for the next wind step.

---

## 10. Summary Table

| Issue | Severity | Evidence | Fix direction |
|---|---|---|---|
| chi2/ndf >> 1 (noise model mismatch) | **Critical** | chi2_min/ndf = 2.65e9 | Identify source: PCG convergence vs forward model mismatch |
| Forward model: pointing vs rotation | **High** | Structural: H_tod vs H_rec | Verify with noiseless TOD |
| Initialisation with true maps | Medium | Simulation only | OK for benchmarking, not for real data |
| Gradient catastrophic cancellation | Medium | eps < 1e-5 gives zero grad | eps=1e-3 avoids it; already handled |
| L-BFGS-B abnormal termination | Medium | Logged in output | Investigate noise in chi2 surface |
| Chunking approximation in W(wind) | Low–Medium | N_buffer=10 | Reduce N_buffer or use continuous integration |
| Only 10 PCG iterations | Medium | PCG not converged at iter end | Increase n_iter or check convergence criterion |
| Error bars ignore wind-map degeneracy | Medium | Structural | Joint Fisher matrix needed |

---

## 11. Proposed Solutions

Solutions are ordered by impact and diagnosis-first principle: the most critical
issue (9.1) should be diagnosed before spending time on secondary fixes.

---

### S1 — Diagnose chi2/ndf source (prerequisite for everything else)

Before changing any code, run a noiseless TOD experiment:

```python
tod_noiseless = H_tod(atm_maps)  # no noise
```

Then run the full fitting loop on `tod_noiseless` and check `chi2_min / ndf`:

- **If chi2_min/ndf ≈ 0.5** → the excess in the noisy run comes from noise calibration
  (invN does not match the actual noise variance). Fix: rescale `invN` so that
  `diag(N) ≈ var(noise)` measured from the noise realisation.

- **If chi2_min/ndf >> 0.5** → the forward model cannot reproduce its own input data.
  The mismatch is structural (pointing deviation vs sky rotation). Fix: **S2** below.

This single test costs one TOD simulation and one fitting run.

---

### S2 — Align the forward model with TOD generation (critical fix)

**Problem**: TOD is generated with wind baked into the *pointing direction* (perturbed
`q_sampling_local`), but reconstruction applies wind as a *sky map rotation* via
`ShiftOperator`. These are not equivalent.

**Fix**: generate TOD using the same reconstruction pipeline:
```python
# Use the same H_rec-style operator but with wind-perturbed sampling
q_wind = wind_class.get_deviated_qubic_sampling()  # already exists
H_tod_consistent = QubicInstrumentType(
    qubic_dict, nsub_in, nsub_in, sampling=q_wind
).get_operator()
tod = H_tod_consistent(atm_maps).ravel() + noise
```

This makes the two sides of the chi2 exactly consistent by construction, and
chi2_min/ndf should collapse to ≈ 0.5 in the noiseless limit.

**Alternative if you want to keep the current TOD**: replace `ShiftOperator` with a
proper wind-aware pointing operator that perturbs each pointing direction at the
map-level by the cumulative wind displacement, matching what `WindPerturbation` does.
This is more work but keeps the physical interpretation.

---

### S3 — Increase PCG convergence (easy, immediate gain)

Replace the fixed-iteration PCG with a tolerance-based stopping criterion:

```python
# In call_pcg, and in the loop:
n_iter = 200         # upper bound
tol_pcg = 1e-6      # stop when relative residual < tol_pcg
```

The `PCGAlgorithm` already accepts `tol=1e-12` but the iteration count `maxiter` caps
it at 10. With `maxiter=200` and `tol=1e-6`, the PCG will converge to the map solution
and the chi2 at the map step will be at its true minimum.

**Expected impact**: directly reduces chi2_min and therefore sigma2_eff. If chi2/ndf
drops from 2.65e9 to ≈ 0.5, the error bars become physically meaningful.

---

### S4 — Fix the gradient precision issue

Once chi2 values are O(N_data) ≈ 5e4 instead of O(1e68), the catastrophic
cancellation threshold shifts from `eps ~ 1e-5` down to `eps ~ 1e-12`. Then:

- Use `eps = 1e-5` (or even let L-BFGS-B compute its own finite differences without
  explicit `jac=`, which would then work correctly).
- Alternatively, derive the **analytical gradient**:

```
d chi2 / d wind_k = -r^T * invN * d(tod_sim)/d(wind_k)
```

where `d(tod_sim)/d(wind_k) = Pchunk * H_ * (d W/d wind_k) * Amm * P(maps_rec)`.

`d W/d wind_k` is the derivative of the HEALPix rotation operator with respect to
displacement, which can be approximated as the pixel-shifted gradient of the map.
This would give a noise-free gradient at zero extra cost per parameter, but requires
implementing the derivative of `shift_healpy_map`.

---

### S5 — Fix the chunking approximation

**Problem**: each N_buffer=10 chunk uses a single mean wind displacement, introducing
an O(wx * N_buffer) intra-chunk error.

**Option A — Reduce N_buffer to 1** (exact, costly):
```python
N_buffer = 1   # one operator per pointing, exact displacement
```
This creates 100 ShiftOperators instead of 10. Compute cost increases 10x for the
wind step. Memory is the bottleneck.

**Option B — Quadratic intra-chunk correction**:
Instead of the mean displacement, use a finer sub-chunking or apply the displacement
at each pointing separately by splitting `get_wind_operator` into individual per-pointing
blocks when N_buffer is large.

**Option C — Accept the approximation, test its impact**:
Run both `N_buffer=1` and `N_buffer=10` on the noiseless TOD and compare chi2_min/ndf.
If the difference is negligible relative to other error sources, N_buffer=10 is fine.

---

### S6 — Robustify the L-BFGS-B optimizer

Even after fixing chi2 scale, the pixelised HEALPix rotation introduces small
discontinuities in chi2 as a function of wind, which can cause line-search failures.

**Option A — Add gradient noise tolerance**: increase `gtol` slightly (e.g., 1e-8
instead of 1e-10) to accept convergence earlier, before the line search struggles with
non-smoothness.

**Option B — Switch to Nelder-Mead for 2 parameters**:
```python
result = minimize(
    lambda x: get_chi2(x, tod) / scale,
    x0=x0,
    method="Nelder-Mead",      # gradient-free, robust to non-smooth surfaces
    bounds=[(-5,5), (-5,5)],
    options={"xatol": 1e-4, "fatol": 1e-8, "maxiter": 500},
)
```
For only 2 parameters, Nelder-Mead is competitive in cost and much more robust to
discontinuities. No Jacobian needed.

**Option C — Grid search warm-start**:
Before calling the optimizer, do a coarse 2D grid scan (e.g., 5×5 over [−1,1]²) to
find a good starting point close to the minimum. This avoids the line search starting
from a bad direction.

---

### S7 — Error bars: account for wind-map degeneracy

The current profile errors fix `maps_rec` and vary wind. They underestimate the true
uncertainty if wind and maps are degenerate (i.e., a slight change in wind can be
compensated by a change in maps).

**Proper treatment — profile over maps**:
For each wind value `w` on the grid, re-solve the map problem (PCG) and use the
resulting minimum chi2:
```python
def chi2_profile(w, tod):
    """Profile chi2: minimise over maps for fixed wind."""
    wind_op = get_wind_operator(w)
    H_w = Pchunk * H_ * wind_op * Amm
    A_w = P.T * H_w.T * invN * H_w * P
    b_w = P.T * H_w.T * invN * tod
    maps_w = PCG(A_w, b_w, maps_rec, max_iter=100)
    return get_chi2(w, tod, maps_override=maps_w)
```

This gives the true marginalised chi2 for wind, independent of maps_rec. It is 10–20×
more expensive (one PCG solve per chi2 evaluation) but gives honest error bars.

---

### S8 — Diagnostic cell: noiseless residual test

Add a cell that computes the "ideal residual" — the residual you'd get with perfect
wind and maps, using the reconstruction forward model:

```python
wind_true_op = get_wind_operator(params["wind_cst"])
tod_rec_true = (Pchunk * H_ * wind_true_op * Amm)(P.T(true_maps) / mixing_matrix[:,1].mean())
res_ideal = tod - tod_rec_true
chi2_ideal = 0.5 * np.dot(res_ideal, invN(res_ideal))
print(f"Ideal chi2 / ndf = {chi2_ideal / (tod.size - 2):.4e}  (forward model floor)")
```

If `chi2_ideal / ndf >> 0.5`, the gap is the structural forward model mismatch floor
(S2). This single number quantifies how much chi2/ndf can ever improve.

---

## 12. Recommended Priority Order

| Priority | Action | Expected impact |
|---|---|---|
| 1 | **S8**: Measure ideal chi2 floor (1 cell, read-only) | Diagnoses root cause immediately |
| 2 | **S1**: Noiseless TOD test | Separates noise calibration from model mismatch |
| 3 | **S2**: Align forward model with TOD generation | Eliminates structural floor if S1/S8 confirm mismatch |
| 4 | **S3**: Increase PCG n_iter to 200 | Reduces chi2 floor from PCG convergence |
| 5 | **S4**: Fix gradient eps after chi2 scale is corrected | Cleaner optimizer behaviour |
| 6 | **S6B**: Switch to Nelder-Mead (2 params) | Eliminates abnormal termination |
| 7 | **S5C**: Test N_buffer=1 vs N_buffer=10 | Quantifies chunking error |
| 8 | **S7**: Profile chi2 over maps for honest error bars | Only meaningful once chi2/ndf ≈ 0.5 |
