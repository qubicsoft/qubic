# Convolution Changes Report — since commit `7d70df14 Remove outdated comment`

## Summary

Five substantive bugs or inconsistencies were fixed across two files
(`qubic/lib/MapMaking/ComponentMapMaking/preset/preset_acquisition.py` and
`qubic/lib/MapMaking/ComponentMapMaking/Qcmm.py`), all related to the angular
resolution handling in the CMM pipeline. The rest of the diff is formatting only.

---

## Mathematical Setup

The CMM pipeline generates synthetic data as:

$$\vec{d} = H_\text{in}(\vec{s}_\text{true}) + \vec{n}$$

where $H_\text{in} = [H_{\text{QUBIC},1}, \ldots, H_{\text{QUBIC},N}, H_{\text{Planck},1}, \ldots]$
is the joint acquisition operator. For each QUBIC sub-band $j$:

$$H_{\text{QUBIC},j} = P_j \circ B_j$$

$P_j$ is the pointing operator (sampling), $B_j$ is a Gaussian convolution at
full-width half-maximum $\theta_j$ (`allfwhm[j]`). The reconstruction solves:

$$\hat{x} = \argmin_x \;\|H_\text{out}(x) - \vec{d}\|^2_{N^{-1}}$$

using PCG, where $H_\text{out}$ is the reconstruction operator controlled by the
`convolution_out` flag. To compute residuals, one must compare $\hat{x}$ to the
input sky at the **same effective resolution** — this quantity is
`components_in_convolved`.

---

## Change 1 — `get_convolution()`: four-branch restructuring

**File**: `preset_acquisition.py`, `get_convolution()`

### What changed

The old code had two independent `if` statements for `convolution_in` and
`convolution_out`, with a single `_get_scalar_acquisition_operator()` path for
`conv_out=False` that applied the pointing operator to a unit vector to estimate
the effective FWHM. The `conv_in=False, conv_out=True` branch was missing entirely
(→ `AttributeError` at runtime). `components_in_convolved` was always computed in
`__init__` at `min(allfwhm)`, ignoring the convolution flags.

The new code is a clean exhaustive `if/elif/elif/else` on all four
`(conv_in, conv_out)` combinations. Each branch computes three quantities
consistently: `fwhm_qubic_rec`, `fwhm_qubic_mapmaking`, and
`components_in_convolved`.

### Mathematical description per case

**Case (`conv_in=True`, `conv_out=True`)**

The reconstruction operator applies a partial beam removal per sub-band:

$$\theta_j^\text{map} = \sqrt{\theta_j^2 - \theta_min^2}, \quad \theta_min = min_k \theta_k$$

This deconvolves the sub-band-specific part of the beam so that all sub-bands,
after reconstruction, contribute at the same resolution $\theta_min$. The
reference input map is:

$$\vec{s}_\text{ref} = B_{\theta_min} * \vec{s}_\text{true}, \qquad \texttt{fwhm\_rec} = \theta_min$$

**Case (`conv_in=True`, `conv_out=False`)**

No beam is applied in reconstruction ($\theta_j^\text{map} = 0$). The PCG absorbs
the per-sub-band beams directly into the pixel solution. The reconstructed map
converges to an effective beam $B_\text{eff}$ which is a noise-precision-weighted
mean of the sub-band beams (see Change 4 for the exact formula). The reference
map is:

$$\vec{s}_\text{ref} = B_{\theta_\text{eff}} * \vec{s}_\text{true}, \qquad \texttt{fwhm\_rec} = \theta_\text{eff}$$

**Case (`conv_in=False`, `conv_out=True`)**

No beam enters the TOD. The reconstruction operator would formally apply a
beam-removal kernel even though no beam was introduced — this is an unphysical but
valid test case. The reference map is simply
$\vec{s}_\text{ref} = \vec{s}_\text{true}$ (no convolution needed).

**Case (`conv_in=False`, `conv_out=False`)**

No beam anywhere. $\vec{s}_\text{ref} = \vec{s}_\text{true}$, all FWHMs are zero.

---

## Change 2 — `fwhm_planck_tod`: case-dependent assignment

**File**: `preset_acquisition.py`, `__init__()`, lines setting `fwhm_planck_tod`

### What changed

Previously, `fwhm_planck_tod` was always set to `min(fwhm_qubic_tod)` regardless
of the convolution flags. This was the correct value only for the `(True, True)`
case.

### Why it matters

The Planck external data is generated as:

$$\vec{d}_\text{Planck} = H_\text{Planck}(\vec{s}_\text{ref}) + \vec{n}_\text{Planck}$$

For consistency, the Planck convolution in TOD generation must match the QUBIC
reconstruction reference resolution. Otherwise, the Planck prior provides sky maps
at a different angular resolution than QUBIC recovers, introducing a systematic
bias in the joint reconstruction. The new assignment is:

| `conv_in` | `conv_out` | `fwhm_planck_tod` | Reason |
|---|---|---|---|
| True | True | $\theta_min = min_j \theta_j$ | QUBIC reconstructs at $\theta_min$; Planck must also target $\theta_min$ |
| True | False | $\theta_\text{eff}$ (`fwhm_rec[0]`) | QUBIC PCG converges to $\theta_\text{eff}$; Planck must match |
| False | * | $0$ | No beam in QUBIC TOD; Planck contributes unconvolved sky |

---

## Change 3 — `components_in_convolved` consistency fix in PCG

**File**: `Qcmm.py`, `update_components()` and `call_pcg()`

### What changed

In `update_components` and `call_pcg`, the Planck prior and the PCG output update
previously used `components_out` (the raw, unconvolved input sky). They now use
`components_in_convolved` (the sky convolved to the reconstruction resolution).

**In `update_components`** (the map-making right-hand side):

```python
# Before:
x_planck_full = self.preset.comp.components_out * weight_mask
# After:
x_planck_full = self.preset.acquisition.components_in_convolved * weight_mask
```

**In `call_pcg`** (adding back the prior after PCG):

```python
# Before:
components_iter[seenpix] = x_pcg + w * components_out[seenpix]
# After:
components_iter[seenpix] = x_pcg + w * components_in_convolved[seenpix]
```

### Mathematical description

The map-making equation is solved in two parts. Defining
$x_\text{prior} = w \cdot \vec{s}_\text{ref}$ (the Planck prior at reconstruction
resolution), the PCG solves the residual problem:

$$\hat{x}_\delta = \argmin_{x} \; \| H_\text{out}(x + x_\text{prior}) - \vec{d} \|^2_{N^{-1}}$$

which in practice is expanded as:

$$H_\text{out}^\top N^{-1} H_\text{out} \, \hat{x}_\delta = H_\text{out}^\top N^{-1} (\vec{d} - H_\text{out}(x_\text{prior}))$$

The full solution is then $\hat{x} = \hat{x}_\delta + x_\text{prior}$. For this to
be self-consistent:

- The prior $x_\text{prior}$ must be at the **same resolution** as what
  $H_\text{out}$ models (i.e., `components_in_convolved`, not `components_out`).
- Using the unconvolved sky `components_out` as prior introduced a systematic: the
  subtracted term $H_\text{out}(x_\text{prior})$ did not match the actual TOD
  contribution, leaving a residual that biased the PCG solution.

---

## Change 4 — `_compute_invn_weighted_fwhm()`: automatic `fwhm_rec` for UWB vs DB

**File**: `preset_acquisition.py`, new method `_compute_invn_weighted_fwhm()`

### What changed

The old code used `_get_scalar_acquisition_operator()` — it applied `H[j]` to a
unit vector and used the mean result as a proxy for the sub-band weight. This was
numerically wrong because it captured only pointing effects, not the noise
structure. It was replaced by an analytical invN-weighted formula.

### Mathematical description

For `conv_out=False`, the PCG reconstruction converges to a map that is the
inverse-noise-precision-weighted mean of the per-sub-band convolved skies:

$$\hat{s}(x) = \frac{\sum_j w_j \, B_j * s(x)}{\sum_j w_j}, \qquad w_j = \frac{1}{\sigma_j^2}$$

The effective FWHM of the reconstruction reference map is therefore:

$$\theta_\text{eff} = \frac{\sum_j w_j \, \theta_j}{\sum_j w_j}$$

The noise variance per sub-band differs between instruments because of how detector
noise is attributed in the acquisition model:

- **DB** — two independent focal planes, each with its own detectors:

$$\sigma_{150,j}^2 = n_\text{det}^2 + n_{\text{pho},150}^2, \qquad \sigma_{220,j}^2 = n_\text{det}^2 + n_{\text{pho},220}^2$$

- **UWB** — a single shared focal plane whose detectors observe the full bandwidth.
  Since both 150 GHz and 220 GHz are observed by the same detectors, the total
  detector noise budget is one set (not two). In the model this is captured by
  attributing detector noise to the 150 GHz half only
  (`det_noise = [n_det, 0]`), avoiding double-counting:

$$\sigma_{150,j}^2 = n_\text{det}^2 + n_{\text{pho},150}^2, \qquad \sigma_{220,j}^2 = n_{\text{pho},220}^2$$

Since $\sigma_{220,\text{UWB}}^2 < \sigma_{220,\text{DB}}^2$, the 220 GHz
sub-bands (narrower beams, higher frequency) carry more weight in UWB than in DB.
This pulls $\theta_\text{eff}$ below the arithmetic mean for UWB.

**Numerical example** with `ndet = npho150 = npho220 = 1`,
$\bar\theta_{150} \approx 0.00692$ rad, $\bar\theta_{220} \approx 0.00472$ rad:

| Instrument | $w_{150}$ | $w_{220}$ | $\theta_\text{eff}$ |
|---|---|---|---|
| UWB | $1/(1+1) = 0.5$ | $1/1 = 1.0$ | $\approx 0.0054$ rad ✓ |
| DB | $1/(1+1) = 0.5$ | $1/(1+1) = 0.5$ | $\approx 0.0058$ rad ✓ |

The `fwhm_rec` parameter in `params.yml` allows a manual override if the empirical
PCG output FWHM is known from calibration:

```yaml
fwhm_rec: null  # null = auto-compute from noise weights; set to e.g. 0.0054 to override
```

---

## Change 5 — `get_tod_comp()`: missing beam convolution in simulated TOD

**File**: `Qcmm.py`, `get_tod_comp()`

### What changed

`get_tod_comp()` computes simulated TOD from the current component maps for use in
mixing matrix fitting. Previously it applied `H[j]` (pointing only) without any
beam convolution. Now it applies the reconstruction-mode beam `C_j` before `H[j]`:

```python
# Before:
tod_comp[i, j] = H[j](components_iter[i]).ravel()

# After:
C_j = HealpixConvolutionGaussianOperator(fwhm=fwhm_mapmaking[j])
tod_comp[i, j] = H[j](C_j(components_iter[i])).ravel()
```

### Why it matters

The TOD model used in mixing matrix fitting must match the same forward operator
as the map-making step. In the `conv_out=True` case, $H_{\text{out},j}$ includes
partial beam removal with FWHM $\theta_j^\text{map} = \sqrt{\theta_j^2 - \theta_min^2}$.
`get_tod_comp()` must include this same convolution; without it, the simulated TOD
used for spectral index fitting was inconsistent with the reconstructed maps, which
would bias the spectral index estimates.

---

## Files changed

| File | Nature of change |
|---|---|
| `qubic/lib/MapMaking/ComponentMapMaking/preset/preset_acquisition.py` | Full rewrite of `get_convolution()`, new `_compute_invn_weighted_fwhm()`, corrected `fwhm_planck_tod` |
| `qubic/lib/MapMaking/ComponentMapMaking/Qcmm.py` | `components_out` → `components_in_convolved` in PCG prior; added beam convolution in `get_tod_comp()` |
| `qubic/scripts/MapMaking/src/CMM/params.yml` | Added `fwhm_rec: null` parameter |
