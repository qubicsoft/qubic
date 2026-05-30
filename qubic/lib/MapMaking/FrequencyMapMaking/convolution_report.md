# Convolution Handling in the FMM Pipeline

Reference implementation: `Qfmm.py`, `PipelineFrequencyMapMaking`.

---

## Mathematical Setup

The FMM pipeline reconstructs `nrec` frequency-averaged sky maps $\hat{s} = (\hat{s}_1, \ldots, \hat{s}_{N_\text{rec}})$.
Each reconstructed band covers `fsub_out = nsub_out / nrec` sub-acquisitions.

The simulated data are:

$$\vec{d} = H_\text{in}(\vec{s}) + \vec{n}$$

where $H_\text{in} = [H_{\text{QUBIC},1}, \ldots, H_{\text{QUBIC},N_\text{sub\_in}}, H_{\text{Planck},1}, \ldots]$.

For each QUBIC sub-band $j$ (in the *input* operator):

$$H_{\text{QUBIC},j} = P_j \circ B_j$$

$P_j$ is the pointing operator (synthesized beam + sampling), $B_j$ is a Gaussian convolution at FWHM
$\theta_j$ (`allfwhm[j]`). Planck is modelled as a direct-space operator — its "TOD" is the
band-averaged sky map at the reconstruction reference resolution plus noise:

$$\text{TOD}_{\text{Planck},k} = B_{\theta_\text{rec}[k]}(s_k) + n_k$$

The reconstruction solves:

$$\hat{s} = \arg\min_x \;\|H_\text{out}(x) - \vec{d}\|^2_{N^{-1}}$$

via PCG, where $H_\text{out}$ uses `fwhm_out` (controlled by `convolution_out`).
The reference truth at reconstruction resolution is `maps_input_convolved`:

$$\text{maps\_input\_convolved}[k] = B_{\theta_\text{rec}[k]}(\text{maps\_input}[k])$$

**Key difference from CMM:** FMM produces frequency maps, not physical component maps.
There is no mixing matrix or spectral index fitting.

---

## Sub-band Bookkeeping

| Symbol | Meaning |
|---|---|
| `nsub_in` | Sub-bands for TOD generation |
| `nsub_out` | Sub-bands for reconstruction |
| `nrec` | Number of reconstructed frequency maps |
| `fsub_in = nsub_in / nrec` | Input sub-bands per rec band |
| `fsub_out = nsub_out / nrec` | Output sub-bands per rec band |

`nsub_in` and `nsub_out` may differ. When they are equal and `path_tod is None`, the
operator $H$ is shared between TOD generation and reconstruction (`H = joint_tod.qubic.H`).

The input sky maps are averaged over `fsub_in` sub-frequencies per rec band:

$$\text{maps\_input}[k] = \frac{1}{f_\text{sub\_in}} \sum_{j=k \cdot f_\text{sub\_in}}^{(k+1) \cdot f_\text{sub\_in}-1} m_\nu[j]$$

All FWHM computations for `fwhm_rec[k]` use only the `allfwhm` entries belonging to the
reconstruction sub-bands of band $k$ (indices $k \cdot f_\text{sub\_out}$ to $(k+1) \cdot f_\text{sub\_out} - 1$).

---

## The Four Convolution Branches

`get_convolution()` returns three arrays: `fwhm_in`, `fwhm_out`, `fwhm_rec`.

### Branch 1 — `conv_in=True`, `conv_out=True`

Partial beam removal per sub-band in reconstruction. Within each rec band $k$, define
$\theta_\text{min}^{(k)} = \min_j \theta_j$ (minimum over its `fsub_out` sub-bands). Then:

$$\theta_j^\text{out} = \sqrt{\theta_j^2 - (\theta_\text{min}^{(k)})^2}, \qquad
\theta_\text{rec}^{(k)} = \theta_\text{min}^{(k)}$$

After composition $B_j \circ B_j^\text{out} = B_{\theta_\text{min}^{(k)}}$, all sub-bands in
band $k$ contribute at the same resolution $\theta_\text{min}^{(k)}$.
The reference truth is `maps_input_convolved[k] = C(θ_min^(k)) * maps_input[k]`.

| Quantity | Value | Why |
|---|---|---|
| `fwhm_in[j]` | `allfwhm[j]` | Real sub-band beam |
| `fwhm_out[j]` | `sqrt(allfwhm[j]² − θ_min^(k)²)` | Differential beam; composition = `allfwhm[j]` ✓ |
| `fwhm_rec[k]` | `min(allfwhm in band k)` | Finest beam achievable per band |
| Planck TOD | `C(θ_min^(k)) * maps_input[k] + noise` | Consistent with reconstruction target ✓ |

### Branch 2 — `conv_in=True`, `conv_out=False`

Beam in the TOD, no additional convolution in reconstruction. The PCG absorbs the per-sub-band
beams into the pixel solution, but the effective output resolution is not uniform across
sub-bands. The reconstruction converges to a map at an effective FWHM determined by the
signal-weighted average across sub-bands.

$$\theta_j^\text{out} = 0, \qquad
\theta_\text{rec}^{(k)} = \frac{\sum_{j \in k} \alpha_j \cdot w_j \cdot \theta_j}{\sum_{j \in k} \alpha_j \cdot w_j}$$

where $\alpha_j$ = `scalar_acquisition_operators[j]` (effective pointing coverage) and
$w_j$ = `weight_factor[j]` (SED evaluated at `allnus[j]`; see §4 for the formula).

The reference truth is `maps_input_convolved[k] = C(θ_rec^(k)) * maps_input[k]`.

### Branch 3 — `conv_in=False`, `conv_out=True`

**This combination is physically inconsistent.** No beam is applied during TOD generation
(`fwhm_in = 0`), yet the reconstruction operator includes non-zero partial deconvolution
beams (`fwhm_out = sqrt(allfwhm² − min²)`). Applying a deconvolution beam to beam-free data
has no physical meaning. The code does not guard against this case; see §7.

### Branch 4 — `conv_in=False`, `conv_out=False`

No convolution anywhere. All FWHMs are zero. The reference truth is `maps_input` itself,
and `maps_input_convolved = maps_input`.

---

## Signal-Weighted `fwhm_rec` (Branch 2)

`_get_scalar_acquisition_operator()` computes the effective pointing weight of each sub-band
by applying its acquisition operator to a ones vector:

```python
scalar_acquisition_operators[j] = np.mean(H_j(np.ones(H_j.shapein)))
```

The SED weight is chosen based on `params["Foregrounds"]["Dust"]`:

```python
# Dust sky
weight_factor = Dust(nu0=353, beta_d=1.54, temp=20).eval(allnus)

# CMB sky (default)
weight_factor = CMB().eval(allnus)
```

The effective FWHM and the corresponding effective frequency for each rec band are:

```python
fwhm_rec[k] = sum(scalar[j] * weight[j] * fwhm_in[j]) / sum(scalar[j] * weight[j])
fraction    = sum(scalar[j] * weight[j]) / sum(scalar[j])
nus_Q[k]    = argmin_ν |fraction − SED(ν)|   # corrected effective frequency
```

**Limitation:** The weight uses a single component SED. For a CMB+Dust sky, the dominant
component at each sub-band frequency drives the weight, and `fwhm_rec` will be systematically
wrong for the subdominant component.

---

## Planck Data Integration

Planck contributes to the joint TOD as band-averaged maps at the reconstruction reference
resolution plus noise. There is no explicit beam operator for Planck — the beam is baked into
`maps_input_convolved` via `HealpixConvolutionGaussianOperator(fwhm_rec[k])`.

```python
TOD_PLANCK[irec] = maps_input_convolved[irec] + noise_planck[band]
```

The noise band assignment follows the Planck frequency split at `nrec/2`:

| Rec band index | Noise used |
|---|---|
| `irec < nrec / 2` | `noise_planck[0]` (150 GHz noise level) |
| `irec >= nrec / 2` | `noise_planck[1]` (220 GHz noise level) |

Parameters controlling Planck noise:

| Parameter | Role |
|---|---|
| `level_noise_planck` | Multiplicative scale on Planck noise; `0` → noiseless Planck |
| `weight_planck` | Weight on Planck constraints within the QUBIC patch |
| `seed_noise` | Noise realization seed |

**Caution — `level_noise_planck=0` tautology.** When `planck_ntot=0`, the Planck invN is an
`IdentityOperator` everywhere, so Planck pins $\hat{s} \to \text{maps\_input\_convolved}$ regardless
of `weight_planck`. The only meaningful test of reconstruction quality (with Planck) is
`weight_planck=0, level_noise_planck > 0`.

---

## Shifted PCG Formulation and Boundary Bleed Fix

### The shift

When `external_data=True`, the PCG solves for $\delta m = \hat{s} - \text{maps\_input\_convolved}$
rather than $\hat{s}$ directly:

```python
x0 = np.zeros(maps_input[:, seenpix, :].shape)   # start at maps_input_convolved
b  = H_out.T * invN * (d - H_out_all_pix(x_planck))   # x_planck = maps_input_convolved (unmasked)
```

After convergence:

```python
solution[:, seenpix, :] = pcg_output["x"]["x"]                   # delta_m
solution[:, seenpix, :] += maps_input_convolved[:, seenpix, :]   # recover m
```

### Boundary bleed

For `conv_out=True`, `H_out` contains Gaussian convolution operators $B_j^\text{out}$.
If `x_planck` were masked (set to zero inside the QUBIC patch) before applying $H_\text{out}$,
the convolution would spread outside-patch signal across the patch boundary, biasing the
RHS of the PCG. The fix is to apply $H_\text{out}$ to the full, **unmasked**
`maps_input_convolved` and let $N^{-1}_\text{Planck} = 0$ inside the patch (when
`weight_planck=0`) suppress any contribution there in exact arithmetic.

---

## External TOD Loading (`path_tod`)

When `path_tod` is not `None`, the pipeline loads a pre-computed TOD from HDF5:

```python
data = HDF5Dict().load_dict(self.path_tod)
return data["tod"]
```

In this case:
- `joint_tod` is never built.
- `fwhm_in` is forced to `np.zeros(nsub_out)` — the pipeline has no record of what beam
  was applied during external TOD generation.
- `npointings` is automatically adjusted if the file contains multiple combined simulations
  (`n_sims * npointings_per_sim`).

The reconstruction parameters (`fwhm_out`, `fwhm_rec`) must be set to match whatever
convolution was used when the external TOD was generated. There is no validation for this.

---

## Known Inconsistencies

### 1. `conv_in=False, conv_out=True` is physically invalid

The code computes non-zero `fwhm_out` and `fwhm_rec` even when no beam was applied to the
TOD. The reconstruction operator then applies partial deconvolution beams to beam-free data,
and `maps_input_convolved` is convolved at `fwhm_rec = min(allfwhm)` even though no
convolution should be needed. `FMM_errors_checking.py` does not guard against this.

**Fix direction:** Add a check in `ErrorChecking` or `get_convolution()` that raises
`ValueError` when `conv_in=False` and `conv_out=True`.

### 2. Single-component SED weight for multi-component sky

The `fwhm_rec` formula in Branch 2 uses either Dust or CMB as the SED weight, selected by
`params["Foregrounds"]["Dust"]`. For a mixed sky (CMB + Dust), the effective resolution
after reconstruction differs between components, but a single scalar `fwhm_rec` is used for
`maps_input_convolved`. The convolved reference will be correct for the dominant component
and biased for the subdominant one.

### 3. External TOD convolution provenance

`path_tod` loads an opaque HDF5 file. The `fwhm_in` logged in the output (and used to build
`maps_input_convolved`) is zero, not the true value used when the external TOD was generated.
The saved `fwhm_in` in the output HDF5 is therefore misleading in this mode.

---

## Verification Checklist

1. **Branch 1** (`conv_in=True, conv_out=True`): confirm
   - `fwhm_out[j] = sqrt(allfwhm[j]² − min_irec²)` for each sub-band
   - `fwhm_rec[k] = min(allfwhm in band k)` for each rec band
   - `maps_input_convolved` is visibly smoother than `maps_input`

2. **Branch 2** (`conv_in=True, conv_out=False`): confirm
   - `fwhm_rec[k]` lies between `min(allfwhm)` and `max(allfwhm)` for each band
   - `nus_Q` shifts slightly from the arithmetic mean of sub-band frequencies

3. **Signal correlation** (all branches): compute `corr(s_hat − maps_input_convolved, maps_input_convolved)` for each Stokes parameter — should be $\approx 0$, indicating residuals are pure noise with no signal leakage.

4. **Boundary bleed** (`conv_out=True, weight_planck=0`): plot `s_hat` near the patch boundary — no ring or edge artefact from Planck outside-patch data bleeding in.

5. **Planck tautology**: with `level_noise_planck=0`, `s_hat` should be identical to `maps_input_convolved` regardless of `weight_planck`. This confirms the invN = Identity path.

6. **External TOD**: when using `path_tod`, print `fwhm_in` at startup — should be all zeros. Manually verify `conv_out` and `fwhm_rec` match the external generation settings.

7. **`conv_in=False, conv_out=True`**: this should raise an error (currently does not). Flag in `FMM_errors_checking.py`.

---

## Resolution Summary Table

| Branch | `fwhm_in[j]` | `fwhm_out[j]` | `fwhm_rec[k]` | `maps_input_convolved` |
|---|---|---|---|---|
| `(True, True)` | `allfwhm[j]` | `sqrt(θ_j² − θ_min^(k)²)` | `min_k(allfwhm)` | `C(θ_min^(k)) * maps_input[k]` |
| `(True, False)` | `allfwhm[j]` | `0` | signal-weighted avg | `C(θ_rec^(k)) * maps_input[k]` |
| `(False, True)` | `0` | `sqrt(θ_j² − θ_min²)` | `min(allfwhm)` | `C(θ_min) * maps_input[k]` — **inconsistent** |
| `(False, False)` | `0` | `0` | `0` | `maps_input[k]` |

---

## Relevant Parameters (`params.yaml`)

| Parameter | Section | Role |
|---|---|---|
| `convolution_in` | QUBIC | Apply beam to TOD generation |
| `convolution_out` | QUBIC | Apply partial deconvolution in reconstruction |
| `nsub_in` | QUBIC | Sub-bands for TOD generation |
| `nsub_out` | QUBIC | Sub-bands for reconstruction |
| `nrec` | QUBIC | Number of reconstructed maps |
| `synthbeam_kmax` | QUBIC.SYNTHBEAM | Synthesized beam harmonic order (TOD) |
| `synthbeam_kmax_out` | QUBIC.SYNTHBEAM | Synthesized beam harmonic order (reconstruction) |
| `external_data` | PLANCK | Include Planck in joint acquisition |
| `weight_planck` | PLANCK | Planck weight inside QUBIC patch |
| `level_noise_planck` | PLANCK | Planck noise scale (`0` = noiseless) |
| `path_tod` | top-level | Load pre-computed TOD from HDF5 |
