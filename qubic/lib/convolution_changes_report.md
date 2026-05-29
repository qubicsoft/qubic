# Convolution Handling in the CMM Pipeline — Complete Change Log

Covers all changes since commit `7d70df14 Remove outdated comment` (2026-05-26) through
2026-05-29, across `preset_acquisition.py`, `Qcmm.py`, `Qacquisition.py`, and `params.yml`.

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

For consistency, Planck TOD generation must target the same angular resolution as the QUBIC
reconstruction reference:

| `conv_in` | `conv_out` | `fwhm_planck_tod` | Reason |
|---|---|---|---|
| True | True | $\theta_\text{min}$ | QUBIC reconstructs at $\theta_\text{min}$ |
| True | False | `0` (see Change 7) | Beam baked into `components_in_convolved` |
| False | * | $0$ | No beam in QUBIC TOD |

---

## Change 3 — PCG prior: `components_out` → `components_in_convolved`

**File**: `Qcmm.py`, `update_components()` and `call_pcg()`

The prior and the update must both use `components_in_convolved` (the sky at reconstruction
resolution). Previously `components_out` (raw unconvolved sky) was used, leaving a bias in
$H_\text{out}(x_\text{prior})$.

---

## Change 4 — UWB noise model: consistent `invN` and GLS weights

**Files**: `Qacquisition.py` `QubicInstrumentType.get_invntt_operator()`,
`preset_acquisition.py` `_get_subband_weights()`

### Problem

For UWB (single focal plane of 992 detectors observing both bands simultaneously),
`QubicTotNoise.total_noise` correctly generates noise as:

```python
return ndet[0] + npho[0] + npho[1]  # variance = sigma_det² + sigma_pho_150² + sigma_pho_220²
```

But `get_invntt_operator` was computing:

```python
self.invN = np.sum(invn_list)  # = 1/(sigma_det² + sigma_pho_150²) + 1/sigma_pho_220²
```

This is **not** equal to the correct `1/(sigma_det² + sigma_pho_150² + sigma_pho_220²)`.
For equal noise levels (`ndet = npho150 = npho220 = 1`) the error is a factor of **4.5×**,
making the PCG treat UWB data as 4.5× more precise than it is → over-fits noise → wrong maps.

Additionally, `_get_subband_weights` assigned different weights to 150 and 220 GHz sub-bands
in UWB, which is wrong — all sub-bands share the same focal plane and the same combined noise.

### Fix

**`Qacquisition.py`** — save the sigma from each per-band sub-acquisition, combine them into
the correct total sigma, and use `forced_sigma` to build a single consistent invN:

```python
subacqs = []
for iband, band in enumerate(self.used_bands):
    ...
    invn_list.append(subacq.get_invntt_operator(...))
    subacqs.append(subacq)

if self.dict["instrument_type"] == "UWB":
    sigma_combined = np.sqrt(sum(np.atleast_1d(s)**2 for s in [a.sigma for a in subacqs]))
    subacqs[0].forced_sigma = sigma_combined
    self.invN = subacqs[0].get_invntt_operator(det_noise=0, photon_noise=0)
else:
    self.invN = BlockDiagonalOperator(invn_list, axisout=0)
```

This gives `invN = 1/(sigma_det² + sigma_pho_150² + sigma_pho_220²)` — matching `total_noise`. `Qnoise.py` is unchanged.

**`preset_acquisition.py` `_get_subband_weights()`** — for UWB, return uniform weights across
all sub-bands (all share the same combined noise), giving standard unweighted LS separation:

```python
if is_uwb:
    sigma_combined_sq = max(ndet**2 + npho150**2 + npho220**2, 1e-30)
    return np.ones(len(allfwhm)) / sigma_combined_sq
else:
    # DB: band-specific noise
    ...
```

`fwhm_rec` in `params.yml` (scalar, list, or `null`) overrides the auto-computation.

---

## Change 5 — `get_tod_comp()`: missing beam convolution in simulated TOD

**File**: `Qcmm.py`, `get_tod_comp()`

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

For multiple components, a single-component Gaussian fails in two ways:

1. **Different effective FWHMs per component.** CMB is dominated by 150 GHz sub-bands
   (wider beams); Dust by 220 GHz (narrower). A shared $\theta_\text{eff}$ is wrong for
   at least one component.

2. **Cross-component leakage is missing.** The GLS weight matrix
   $W = (A^\top N^{-1} A)^{-1} A^\top N^{-1}$ has negative entries: for Dust,
   $W_{\text{Dust},j} < 0$ at 150 GHz sub-bands. A per-component Gaussian on
   $s_\text{in}[\text{Dust}]$ alone cannot capture this leakage.

### Resolution

Replace the per-component single Gaussian with the full GLS prediction:

$$\texttt{comp\_in\_conv}[i] = \sum_j W_{ij} \; B_j\!\left(\sum_k A_{jk} \cdot s_\text{in}[k]\right)$$

New helpers: `_build_mixing_matrix(allnus)` and `_compute_gls_weights()`.

---

## Change 7 — Planck TOD from `components_in_convolved` with `fwhm_planck_tod = 0`

**Files**: `preset_acquisition.py`, `__init__()` and `get_tod()` — `conv_in=True, conv_out=False`

Set `fwhm_planck_tod = 0` and generate Planck TOD from `components_in_convolved` instead
of raw $s_\text{in}$. Both inside and outside Planck constraints now drive
$m_\text{rec} \to \texttt{comp\_in\_conv}$ everywhere — same target, no boundary discontinuity.

**Caution — `level_noise_planck=0` tautology.** When `planck_ntot=0`, Planck invN is
`IdentityOperator` everywhere, so Planck regularizes $m_\text{rec} \to \texttt{comp\_in\_conv}$
regardless of `weight_planck`. **The only meaningful test of `components_in_convolved` accuracy
is `weight_planck=0` with `level_noise_planck > 0`.**

---

## Change 8 — Reference PCG for `components_in_convolved` (corrects synthesized beam harmonics)

**Files**: `preset_acquisition.py` `get_tod()`, `Qcmm.py` `run()` + `compute_reference_components_in_convolved()`

### Problem

For `conv_in=True, conv_out=False` with `synthbeam_kmax > 0`, the GLS formula misses the
off-diagonal cross-pixel terms in $M = P^\top N^{-1} P$ arising from harmonic side-lobes.
For Dust (negative GLS weights at 150 GHz), the large CMB I signal leaks in via these
harmonic terms → $\sigma_I / (\sigma_Q/\sqrt{2}) \approx 3$.

### Solution

Run the PCG once on the *signal-only* TOD before the main reconstruction loop. This uses
the same operator (synthesized beam included) and produces $E[m_\text{pcg}]$ directly,
correctly accounting for harmonic cross-pixel terms.

### Active conditions (all three must hold)

- `conv_in=True, conv_out=False`
- `synthbeam_kmax > 0` — when `kmax=0` the beam is a pure Gaussian, the GLS is exact, no
  reference PCG is needed
- `use_reference_pcg=True` in `params.yml` QUBIC section (default `True`; set `False` to
  disable explicitly)

### Additional correctness fix: use the true mixing matrix

The reference PCG temporarily replaces `Amm_iter` with `Amm_in` (the true mixing matrix):
the signal-only TOD was generated with `Amm_in`, so using `Amm_iter` (which may be a wrong
initial guess when `fit_mixing_matrix=True`) would bias the reference.

Three state variables are swapped for the duration of the reference PCG and restored afterward:
`TOD_obs`, `weight_planck`, and `Amm_iter`.

---

## Known Issue — `conv_out=True, weight_planck=0` boundary bleed

**Files**: `Qcmm.py` `update_components()` — **not yet fixed**

### Problem

In `update_components`, the shift map `x_planck_full` is constructed by masking
`components_in_convolved` before applying the reconstruction operator:

```python
weight_mask = np.where(seenpix[None,:,None], w, 1.0)   # 0 inside, 1 outside for w=0
x_planck_full = components_in_convolved * weight_mask   # mask THEN convolve via H_i
b = U.T(H_i.T * invN * (TOD_obs - H_i(x_planck_full)))
```

For `conv_out=True`, `H_i` includes Gaussian convolutions `B_j_map` (the partial
deconvolution beams). When applied to the discontinuous `x_planck_full` (0 inside the patch,
non-zero outside), these convolutions **bleed the outside-patch signal into the patch**,
creating a systematic bias in the RHS `b`.

The bias is absent for `conv_out=False` because `B_j_map = Identity` (no convolution across
the boundary), and absent for `weight_planck > 0` because `x_planck_full` is smooth
everywhere (no discontinuity).

### Fix direction

Convolve first, then mask: apply `H_i` to the full (unmasked) `components_in_convolved`,
then restrict the result to the patch via `U.T`. Concretely, use
`x_planck_full = components_in_convolved` (unmasked) and add the inside-patch shift back
in `call_pcg`. Both `update_components` and `call_pcg` need to be updated together for
the shift accounting to remain consistent.

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
2. **`conv_out=True` needs more iterations.** Recommend `n_init_iter_pcg ≥ 100`.
3. **σ_I/(σ_Q/√2) ≈ 1 is expected but not exact.** QUBIC's non-uniform polarization angle
   coverage introduces ~10% intrinsic asymmetry.
4. **`weight_planck=1` gives σ_I < σ_Q/√2 — correct physics.** Planck constrains intensity
   much better than polarization.
5. **`conv_out=False, w=0` gives σ_I/σ_Q = 1.74 before Change 8.** With Change 8 and
   `synthbeam_kmax=0`, the ratio approaches 1.

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
4. For `conv_out=True, weight_planck=1`, use `n_init_iter_pcg ≥ 100`.
5. With `fit_mixing_matrix=True`, verify Planck residuals are near zero at `beta_true`.
6. For multi-component (`conv_out=False`): with `synthbeam_kmax=0` and Change 6, expect
   σ_I/(σ_Q/√2) ≈ 1 for both CMB and Dust. With `kmax > 0`, Change 8 (reference PCG)
   should achieve the same.
7. Validate that the inside/outside patch boundary shows no resolution discontinuity when
   `weight_planck > 0` (Change 7).
8. For UWB: confirm invN value is `1/(sigma_det² + sigma_pho_150² + sigma_pho_220²)` and
   that `fwhm_rec` uses the arithmetic mean of all sub-band FWHMs (uniform weights).

---

## Files Changed

| File | Changes |
| --- | --- |
| `preset/preset_acquisition.py` | Four-branch `get_convolution()`; `_build_mixing_matrix()`, `_compute_gls_weights()`; full GLS `components_in_convolved`; `fwhm_planck_tod=0` for `(True,False)`; Planck TOD from `comp_in_conv`; store `TOD_qubic_signal`, `TOD_planck_signal_zeroed`, `TOD_obs_signal`; UWB uniform GLS weights in `_get_subband_weights()` |
| `Qcmm.py` | `components_out` → `components_in_convolved` in PCG prior and `call_pcg`; beam in `get_tod_comp()`; new `compute_reference_components_in_convolved()` with `kmax > 0` guard, `use_reference_pcg` param, and `Amm_in` swap; call in `run()` |
| `Qacquisition.py` | UWB `get_invntt_operator()`: correct combined sigma via `forced_sigma` |
| `Qchi2MM.py` | Pass `fwhm` to Planck operator in both chi-square methods |
| `params.yml` | `fwhm_rec: null`; `use_reference_pcg: true` |
