# Convolution Handling in the CMM Pipeline — Complete Change Log

Covers all changes since commit `7d70df14 Remove outdated comment` (2026-05-26) through
2026-05-28, across `preset_acquisition.py`, `Qcmm.py`, and `params.yml`.

---

## Mathematical Setup

The CMM pipeline generates synthetic data as:

$$\vec{d} = H_\text{in}(\vec{s}_\text{true}) + \vec{n}$$

where $H_\text{in} = [H_{\text{QUBIC},1}, \ldots, H_{\text{QUBIC},N}, H_{\text{Planck},1}, \ldots]$
is the joint acquisition operator. For each QUBIC sub-band $j$:

$$H_{\text{QUBIC},j} = P_j \circ B_j$$

$P_j$ is the pointing operator (synthesized beam + sampling), $B_j$ is a Gaussian convolution
at FWHM $\theta_j$ (`allfwhm[j]`). The reconstruction solves:

$$\hat{x} = \arg\min_x \;\|H_\text{out}(x) - \vec{d}\|^2_{N^{-1}}$$

using PCG, where $H_\text{out}$ is the reconstruction operator controlled by the
`convolution_out` flag. To compute residuals, $\hat{x}$ must be compared to the input sky
at the **same effective resolution** — this is `components_in_convolved`.

---

## Change 1 — `get_convolution()`: four-branch restructuring

**File**: `preset_acquisition.py`, `get_convolution()`

### Bugs fixed

**Bug 1 (crash):** A `[..., 0]` indexing was applied to `fwhm_planck_tod` after building
it as a Python list. For `conv_out=True`, the list contained scalars → `np.array()` → 1D
array → `[..., 0]` extracted a scalar → `np.concatenate((fwhm_qubic_tod, scalar))` crashed
with `ValueError: zero-dimensional arrays cannot be concatenated`.

**Bug 2 (crash):** The `conv_in=False, conv_out=True` branch was missing entirely from the
if-elif chain → `self.fwhm_planck_tod` was never assigned → `AttributeError` at `np.concatenate`.

**Bug 3 (latent, `Qchi2MM.py`):** In `compute_chi_square_fix_beta` and
`compute_chi_square_varying_beta`, the Planck operator was built without a `fwhm` argument
even though the TOD was generated with a beam convolution applied. With `fit_mixing_matrix=True`
this mismatch biased spectral-index estimates.

### Fix

Replace the entire if-elif block with an exhaustive four-branch structure. Each branch
computes `fwhm_qubic_rec`, `fwhm_qubic_mapmaking`, and `components_in_convolved` consistently.
Pass `fwhm=self.preset.acquisition.fwhm_mapmaking[self.nfreq:]` to the Planck operator in both
chi-square methods (Bug 3).

### Mathematical description per case

**`conv_in=True`, `conv_out=True`**

Partial beam removal per sub-band in reconstruction:
$\theta_j^\text{map} = \sqrt{\theta_j^2 - \theta_\text{min}^2}$, $\theta_\text{min} = \min_k \theta_k$.
All sub-bands contribute at resolution $\theta_\text{min}$ after composition.
Reference: $\vec{s}_\text{ref} = B_{\theta_\text{min}} * \vec{s}_\text{true}$.

**`conv_in=True`, `conv_out=False`**

No beam in reconstruction ($\theta_j^\text{map} = 0$). The PCG absorbs per-sub-band beams
into the pixel solution. Effective reconstruction FWHM is the noise-precision-weighted mean
of sub-band FWHMs (see Change 4). Reference: $\vec{s}_\text{ref} = B_{\theta_\text{eff}} * \vec{s}_\text{true}$.

**`conv_in=False`, `conv_out=True`** and **`conv_in=False`, `conv_out=False`**

No beam in TOD. Reference: $\vec{s}_\text{ref} = \vec{s}_\text{true}$, all FWHMs zero.

### Resolution consistency for `conv_in=True`, `conv_out=True`

| Quantity | Value | Why |
|---|---|---|
| QUBIC TOD beam | `allfwhm[j]` per sub-band | Real synthesized beam width |
| Planck TOD beam | `min(allfwhm)` | Degrade to QUBIC's finest beam |
| Reconstruction target | `min(allfwhm)` | Finest beam achievable |
| QUBIC mapmaking beam | `sqrt(allfwhm[j]² − min²)` | Differential; composed = `allfwhm[j]` ✓ |
| Planck mapmaking beam | `0` | Maps already at `min(allfwhm)`; no extra conv ✓ |
| `components_in_convolved` | `C(min) * s_in` | Correct truth at reconstruction resolution ✓ |

Note: `fwhm_planck_mapmaking = 0` is **required**, not a simplification. The TOD is
`H_planck(C(min)*s)` and the reconstruction model must match: `H_planck(C(0)*m) = H_planck(m)`.

---

## Change 2 — `fwhm_planck_tod`: case-dependent assignment

**File**: `preset_acquisition.py`, `__init__()`

### Description

For consistency, Planck TOD generation must target the same angular resolution as the QUBIC
reconstruction reference. The assignment depends on the convolution flags:

| `conv_in` | `conv_out` | `fwhm_planck_tod` | Reason |
|---|---|---|---|
| True | True | $\theta_\text{min}$ | QUBIC reconstructs at $\theta_\text{min}$ |
| True | False | `0` (see Change 7) | Beam baked into `components_in_convolved` |
| False | * | $0$ | No beam in QUBIC TOD |

The `conv_in=True, conv_out=False` row was initially set to `fwhm_qubic_rec[0]` and later
corrected to `0` in Change 7 (see below).

---

## Change 3 — PCG prior: `components_out` → `components_in_convolved`

**File**: `Qcmm.py`, `update_components()` and `call_pcg()`

### Prior consistency fix

The map-making equation is solved in two parts. With prior
$x_\text{prior} = w \cdot \vec{s}_\text{ref}$, the PCG solves the residual problem:

$$H_\text{out}^\top N^{-1} H_\text{out} \, \hat{x}_\delta = H_\text{out}^\top N^{-1} (\vec{d} - H_\text{out}(x_\text{prior}))$$

and the full solution is $\hat{x} = \hat{x}_\delta + x_\text{prior}$.

Previously, `components_out` (the raw unconvolved sky) was used as prior. This is wrong:
$H_\text{out}(x_\text{prior})$ did not match the actual TOD contribution, leaving a bias.
The prior and the update must both use `components_in_convolved` (the sky at reconstruction
resolution).

---

## Change 4 — `_compute_invn_weighted_fwhm()`: automatic `fwhm_rec` for UWB vs DB

**File**: `preset_acquisition.py`, new methods `_get_subband_weights()` and `_compute_gls_weights()`

### Noise-weighted FWHM formula

For `conv_out=False`, the PCG converges to the inverse-noise-weighted mean of sub-band skies:

$$\theta_\text{eff} = \frac{\sum_j w_j \, \theta_j}{\sum_j w_j}, \qquad w_j = \frac{1}{\sigma_j^2}$$

Noise variance per sub-band differs between instruments:

- **DB** (independent focal planes):
  $\sigma_{150}^2 = n_\text{det}^2 + n_{\text{pho},150}^2$,
  $\sigma_{220}^2 = n_\text{det}^2 + n_{\text{pho},220}^2$

- **UWB** (single shared focal plane, detector noise attributed to 150 GHz only):
  $\sigma_{150}^2 = n_\text{det}^2 + n_{\text{pho},150}^2$,
  $\sigma_{220}^2 = n_{\text{pho},220}^2$

Since $\sigma_{220,\text{UWB}}^2 < \sigma_{220,\text{DB}}^2$, the 220 GHz sub-bands
(narrower beams) carry more weight in UWB, pulling $\theta_\text{eff}$ lower.

`fwhm_rec` in `params.yml` (scalar, list, or `null`) overrides the auto-computation.

---

## Change 5 — `get_tod_comp()`: missing beam convolution in simulated TOD

**File**: `Qcmm.py`, `get_tod_comp()`

### Forward model consistency

`get_tod_comp()` produces simulated TOD from current component maps for mixing matrix
fitting. For `conv_out=True`, $H_{\text{out},j}$ includes partial beam removal with FWHM
$\theta_j^\text{map}$. The simulated TOD must apply the same convolution $C_j$ before the
pointing operator; without it, the forward model used in spectral index fitting was
inconsistent with the reconstructed maps.

---

## 2026-05-28 — Multi-component and synthesized beam fixes

Changes 1–5 were designed for a single sky component. With multiple components (CMB + Dust),
three further issues appeared.

---

## Change 6 — `components_in_convolved`: per-component GLS prediction with cross-component leakage

**File**: `preset_acquisition.py`, `get_convolution()` — `conv_in=True, conv_out=False` branch

### Problem

For multiple components, a single-component Gaussian fails in two ways:

1. **Different effective FWHMs per component.** CMB is dominated by 150 GHz sub-bands
   (wider beams); Dust by 220 GHz (narrower). A shared $\theta_\text{eff}$ is wrong for
   at least one component.

2. **Cross-component leakage is missing.** The GLS weight matrix
   $W = (A^\top N^{-1} A)^{-1} A^\top N^{-1}$ has negative entries: for Dust,
   $W_{\text{Dust},j} < 0$ at 150 GHz sub-bands. The PCG Dust reconstruction therefore
   *subtracts* 150 GHz data — which contains large CMB I signal. A per-component Gaussian
   on $s_\text{in}[\text{Dust}]$ alone cannot capture this leakage, leaving a systematic
   CMB I signal in the Dust residual → $\sigma_I / (\sigma_Q/\sqrt{2}) \gg 1$.

### Resolution

Replace the per-component single Gaussian with the full GLS prediction:

$$\texttt{comp\_in\_conv}[i] = \sum_j W_{ij} \; B_j\!\left(\sum_k A_{jk} \cdot s_\text{in}[k]\right)$$

$W_{ij}$ are the GLS weights (can be negative), $A_{jk}$ is the SED mixing matrix, $B_j$
is the Gaussian at `allfwhm[j]`. Each sub-band beam is applied to the *combined* sky before
accumulation — this correctly captures both the per-component effective beam and the
cross-component leakage.

New helpers: `_build_mixing_matrix(allnus)` and refactored `_compute_gls_weights()`.

---

## Change 7 — Planck TOD from `components_in_convolved` with `fwhm_planck_tod = 0`

**Files**: `preset_acquisition.py`, `__init__()` and `get_tod()` — `conv_in=True, conv_out=False`

### Problem (supersedes Change 2 for this case)

With `fwhm_planck_tod = fwhm_qubic_rec[0]` (CMB FWHM), the inside-patch Planck data was
$B_{\theta_P}(A_\text{Planck} \cdot s_\text{in})$, driving $m_\text{rec} \to B_{\theta_P}(s_\text{in})$.
But the outside-patch prior used `components_in_convolved` at the GLS resolution. Two
different targets → visible resolution discontinuity at the patch boundary.

Additionally: when `level_noise_planck=0`, `get_invntt_operator` returns `IdentityOperator`
unconditionally, bypassing the `weight_planck` mask. Planck then has unit weight *inside*
the patch regardless of `weight_planck`. With the old inconsistent TOD source, this forced
the Dust reconstruction toward the wrong resolution.

### Planck TOD source fix

Set `fwhm_planck_tod = 0` (Planck operator reduces to $A_\text{Planck}$ only, no beam), and
generate Planck TOD from `components_in_convolved` instead of raw $s_\text{in}$:

```python
planck_source = components_in_convolved  # conv_in=True, conv_out=False only
TOD_planck = H_P(planck_source)          # H_P has fwhm=0 → just A_planck
```

Both inside and outside Planck constraints now drive $m_\text{rec} \to \texttt{comp\_in\_conv}$
everywhere — same target, no boundary discontinuity.

**Caution — `level_noise_planck=0` tautology.** When `planck_ntot=0`, Planck invN is
`IdentityOperator` everywhere, so Planck regularizes $m_\text{rec} \to \texttt{comp\_in\_conv}$
regardless of `weight_planck`. The residual appears clean not because the reference is
correct, but because Planck forces the solution to match it. **The only meaningful test of
`components_in_convolved` accuracy is `weight_planck=0` with `level_noise_planck > 0`.**

---

## Change 8 — Reference PCG for `components_in_convolved` (corrects synthesized beam harmonics)

**Files**: `preset_acquisition.py` `get_tod()`, `Qcmm.py` `run()` + new
`compute_reference_components_in_convolved()`

### Synthesized beam harmonic contamination

Changes 6 and 7 assume $P_j \approx c \cdot I$ (uniform coverage, Gaussian beam only). In
reality the QUBIC acquisition operator is:

$$H_j = \underbrace{P_j}_{\text{synth. beam + pointing}} \circ \underbrace{B_j}_{\text{Gaussian}} \circ A_j$$

where $P_j$ is the full synthesized beam operator (9 peaks for `synthbeam_kmax=1`). The PCG
normal matrix $M = P^\top N^{-1} P$ has **off-diagonal cross-pixel terms** linking pixels
at harmonic-peak separations. The GLS formula assumes $M \approx c \cdot I$, so it misses
these terms.

The actual PCG output versus the GLS prediction:

$$m_\text{rec} = (A^\top M A)^{-1} A^\top M \cdot B \cdot A \cdot s_\text{in} \qquad \text{(includes off-diagonal }M\text{ terms)}$$

$$\texttt{comp\_in\_conv}_\text{GLS} = (A^\top A)^{-1} A^\top \cdot B \cdot A \cdot s_\text{in} \qquad \text{(ignores them)}$$

The difference is a signal-shaped residual. For Dust (negative GLS weights at 150 GHz), the
large CMB I signal leaks in via the harmonic terms → $\sigma_I / (\sigma_Q/\sqrt{2}) \approx 3$.

**Empirical confirmation**: setting `synthbeam_kmax=0` (single Gaussian beam, $M = c \cdot I$
exactly) reduces the ratio to $\approx 1$.

### Reference PCG implementation

The only exact solution: run the PCG on the *signal-only* TOD. This uses the same operator
(synthesized beam included) and produces $E[m_\text{pcg}]$ directly.

**In `get_tod()`**, signal TOD is extracted before noise addition:

```python
self.TOD_qubic_signal = H_Q(s_in)           # signal only, no noise
self.TOD_qubic = self.TOD_qubic_signal + noise
self.TOD_planck_signal_zeroed = H_P(comp_in_conv_zeroed)
self.TOD_obs_signal = concatenate(TOD_qubic_signal, TOD_planck_signal_zeroed)
```

**In `Qcmm.run()`**, before the main loop:

```python
self._steps = 0
self.compute_reference_components_in_convolved(seenpix)  # runs once
while self._info:  # main loop unchanged
    ...
```

`compute_reference_components_in_convolved()` temporarily swaps `TOD_obs` → `TOD_obs_signal`
and forces `weight_planck=0` (to avoid regularization toward the approximate GLS prior),
calls `update_components()` (`n_init_iter_pcg` iterations, `_steps=0`), stores the PCG output
as `components_in_convolved[:, seenpix, :]`, then restores everything and resets
`components_iter`.

**Cost**: one extra PCG run at startup (same iteration count as the first main PCG call).
For `conv_in=False` or `conv_out=True` the method returns immediately.

---

## Empirical Validation (single-component CMB, `conv_out=True`)

DB instrument, nsub=10, nside=256, npointings=3000, 143+217 GHz Planck.
σ in µK (CMB). Residual = `components_in_convolved − components_iter`.

| Run | Iters | PCG conv | σ_I | σ_Q/√2 | σ_I/(σ_Q/√2) | corr(resid, input) |
| --- | --- | --- | --- | --- | --- | --- |
| conv_out=True, w=0, 30 iter | 30 | 6.56e-4 | 3.43 | 2.63 | 1.31 | ~0 |
| conv_out=True, w=0, 50 iter | 50 | 2.68e-4 | 3.44 | 3.13 | 1.10 | 0.0003 |
| conv_out=True, w=1, 100 iter | 100 | 5.88e-9 | 1.51 | 1.98 | 0.76 | 0.003 |
| conv_out=False, w=0, 50 iter | 50 | 6.27e-6 | 3.06 | 1.76 | 1.74 | ~0 |

Key observations:

1. **Signal correlation ≈ 0** in all cases → residuals are pure noise, no systematic leakage.

2. **`conv_out=True` needs more iterations.** Partial beam deconvolution makes the system
   harder to condition. Recommend `n_init_iter_pcg ≥ 100`.

3. **σ_I/(σ_Q/√2) ≈ 1 is expected but not exact.** QUBIC's non-uniform polarization angle
   coverage introduces ~10% intrinsic asymmetry.

4. **`weight_planck=1` gives σ_I < σ_Q/√2 — correct physics, not a bug.** Planck constrains
   intensity much better than polarization, so σ_I drops faster.

5. **`conv_out=False, w=0` gives σ_I/σ_Q = 1.74 before Change 8.** This excess comes from
   synthesized beam harmonic cross-pixel contamination (Change 8). With Change 8 (reference
   PCG) and `synthbeam_kmax=0`, the ratio approaches 1.

---

## Note on `maps_noise`

`maps_noise` stored in the HDF5 output (`Qcmm.py`) is:

```python
"maps_noise": components_in_convolved - components_iter
```

This is **the same quantity as the residual** — the reconstruction error at the current
iteration. It is not an independent noise realization.

---

## Verification Checklist

1. Run with `conv_in=True, conv_out=True`. Confirm no `np.concatenate` crash and that
   `fwhm_planck_tod.shape = (n_planck_subs,)`.
2. Confirm `fwhm_tod.shape = (nsub_qubic + n_planck_subs,)`.
3. Check `corr(residual, input) ≈ 0` for all Stokes — confirms no signal leakage.
4. For `conv_out=True, weight_planck=0`, use `n_init_iter_pcg ≥ 100`; expect
   σ_I/(σ_Q/√2) closer to 1.
5. With `fit_mixing_matrix=True`, verify Planck residuals are near zero at `beta_true`.
6. For multi-component (`conv_out=False`): with `synthbeam_kmax=0` and Change 6, expect
   σ_I/(σ_Q/√2) ≈ 1 for both CMB and Dust. With `kmax > 0`, Change 8 (reference PCG)
   should achieve the same.
7. Validate that the inside/outside patch boundary shows no resolution discontinuity when
   `weight_planck > 0` (Change 7).

---

## Files Changed

| File | Changes |
| --- | --- |
| `preset/preset_acquisition.py` | Four-branch `get_convolution()`; `_build_mixing_matrix()`, `_compute_gls_weights()`; full GLS `components_in_convolved`; `fwhm_planck_tod=0` for `(True,False)`; Planck TOD from `comp_in_conv`; store `TOD_qubic_signal`, `TOD_planck_signal_zeroed`, `TOD_obs_signal` |
| `Qcmm.py` | `components_out` → `components_in_convolved` in PCG prior and `call_pcg`; beam in `get_tod_comp()`; new `compute_reference_components_in_convolved()`; call in `run()` |
| `Qchi2MM.py` | Pass `fwhm` to Planck operator in both chi-square methods |
| `params.yml` | `fwhm_rec: null` (scalar / list / null) |
