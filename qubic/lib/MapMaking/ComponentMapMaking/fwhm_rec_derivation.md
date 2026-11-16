# Derivation of `fwhm_rec` in the QUBIC CMM pipeline

**Context**: case `convolution_in = True`, `convolution_out = False`.

---

## 1. Sub-band beam structure

QUBIC observes in `n` spectral sub-bands. Sub-band `j` has a synthesized Gaussian beam of
FWHM `φⱼ` (in radians). In harmonic space, the beam transfer function is:

```
Bⱼ(ℓ) = exp( −ℓ(ℓ+1) σⱼ² / 2 ),    σⱼ = φⱼ / (2√(2 ln 2))
```

For QUBIC UWB with `n = 16` sub-bands (150–220 GHz), the FWHMs span
`φ₁ ≈ 0.00421 rad` (highest frequency) to `φ₁₆ ≈ 0.00771 rad` (lowest frequency).

---

## 2. Forward model

With `convolution_in = True`, sub-band `j` produces the TOD:

```
yⱼ = Hⱼ Bⱼ m_true + nⱼ
```

where `Hⱼ` is the per-sub-band pointing + mixing-matrix operator, `m_true` is the true
sky, and `nⱼ` is noise.

With `convolution_out = False`, the reconstruction operator `H_recon` includes **no beam**:
the PCG solves for a map that, when projected forward without beam smoothing, fits the
beam-smoothed data `yⱼ`.

---

## 3. Effective beam of the reconstruction

In the low-noise limit, the PCG solution converges to:

```
x_rec  ≈  (H_recon^T N⁻¹ H_recon)⁻¹ H_recon^T N⁻¹  Σⱼ yⱼ
       =  Σⱼ wⱼ Bⱼ m_true
       ≡  B_eff m_true
```

where the weights `wⱼ` (normalised, `Σⱼ wⱼ = 1`) are determined by the normal equations.
For equal noise per sub-band and a flat SED, `wⱼ = 1/n` (uniform).

The **exact** effective beam is therefore:

```
B_eff(ℓ) = Σⱼ wⱼ Bⱼ(ℓ) = Σⱼ wⱼ exp( −ℓ(ℓ+1) σⱼ² / 2 )
```

This is a **weighted sum of Gaussians**, not itself a Gaussian.

In the code, the exact pixel-space effective-beam map is stored in `components_in_convolved`:

```python
# weights = 1/n for CMB (uniform); SED-weighted for dust/synchrotron
components_in_convolved[comp] = Σⱼ wⱼ  Bⱼ( m_true[comp] )
```

No approximation is made here. This is the true reference that `x_rec` should converge to.

---

## 4. Single-Gaussian approximation: deriving `fwhm_rec`

For downstream analysis (comparison maps in the notebook, metadata in the data file),
a single effective FWHM `φ_eff` is needed. The question is: which `φ_eff` best
approximates `B_eff`?

### 4.1 Harmonic-space derivation (low-ℓ matching)

Expand `B_eff(ℓ)` to first order in `ℓ(ℓ+1)`:

```
B_eff(ℓ) ≈ 1  −  (ℓ(ℓ+1)/2) Σⱼ wⱼ σⱼ²
```

A single Gaussian with FWHM `φ_eff` has:

```
B_φ_eff(ℓ) ≈ 1  −  ℓ(ℓ+1) σ_eff² / 2
```

Matching the first-order term gives the **weighted RMS**:

```
σ_eff²  =  Σⱼ wⱼ σⱼ²    →    φ_eff^(RMS)  =  √( Σⱼ wⱼ φⱼ² )
```

### 4.2 Why arithmetic mean is used instead

The low-ℓ expansion justifies RMS, but the pipeline validation test compares residuals
across the full multipole range `ℓ ∈ [40, 3×nside]`. For those modes, neither formula
is exact. Numerically, with `n = 16` QUBIC sub-bands and uniform weights:

| Formula                    | Value (rad)  |
|----------------------------|-------------|
| Weighted RMS (uniform `wⱼ`)  | ≈ 0.00547   |
| Arithmetic mean (uniform `wⱼ`) | ≈ 0.00582  |
| Empirical χ² optimum        | ≈ 0.0057    |

The arithmetic mean is closer to the empirical optimum. Intuitively, the sum
`Σⱼ (1/n) Bⱼ` peaks lower and decays more slowly than the narrowest Gaussian in the
set; the arithmetic mean captures this broadening better than the RMS at intermediate ℓ.

### 4.3 Code implementation

```python
# preset_acquisition.py — get_convolution(), case conv_in=True, conv_out=False
weights = self._get_component_weights(comp_name)   # uniform 1/n for CMB
fwhm_qubic_rec[comp] = np.sum(weights * fwhm_qubic_tod)   # arithmetic mean
```

With uniform weights `wⱼ = 1/n`:

```
φ_eff  =  (1/n) Σⱼ φⱼ
```

For dust/synchrotron, `_get_component_weights` scales `wⱼ` by the component SED squared
evaluated at each sub-band frequency `νⱼ`, then renormalises:

```
wⱼ(dust) ∝ f_dust(νⱼ)²,    wⱼ(sync) ∝ f_sync(νⱼ)²
```

This shifts `φ_eff` towards the FWHMs of the bands where the component is bright.

---

## 5. Planck consistency requirement

The PCG right-hand side has two contributions:

```
b  =  H_QUBIC^T N_QUBIC⁻¹ ( y_QUBIC  −  H_QUBIC  x_ref )
   +  H_Planck^T N_Planck⁻¹ ( y_Planck −  H_Planck x_ref )
```

where `x_ref = components_in_convolved`.

For the reconstruction to converge to noise only (`b = noise term`), both terms must
have zero signal mean.

**QUBIC term**: `y_QUBIC − H_QUBIC x_ref` averages to zero by construction (the sum
of sub-band residuals cancels when `x_ref = B_eff m_true`).

**Planck term**: if Planck data is generated from a different reference `m_in ≠ x_ref`,
then:

```
b_Planck  =  N_Planck⁻¹ H_Planck ( m_in − x_ref )  ≠  0
```

This is a **signal-level systematic bias** that accumulates across PCG iterations. UWB
is more sensitive to it than DB because the stronger QUBIC normal matrix (more sub-bands
contributing coherently) causes the PCG to converge faster, so more of `b_Planck`
is integrated into the solution before the stopping criterion is met.

**Fix**: generate the Planck TOD from `components_in_convolved` directly, and set
`fwhm_planck_tod = 0` so `H_Planck` applies only the mixing matrix (no additional beam).
Then:

```
y_Planck  =  H_Planck( components_in_convolved )
           =  H_Planck  x_ref

→  b_Planck  =  N_Planck⁻¹ ( H_Planck x_ref − H_Planck x_ref )  =  0  exactly
```

The PCG correction `Δx = A⁻¹ b` is then driven by noise alone, guaranteeing
`σ_I / (σ_Q / √2) → 1` for both DB and UWB.

---

## Summary

| Quantity | Definition | Code |
|----------|-----------|------|
| `components_in_convolved` | Exact weighted sum `Σⱼ wⱼ Bⱼ m_true` | `get_convolution()` lines 230–234 |
| `fwhm_rec` | Arithmetic mean `Σⱼ wⱼ φⱼ` (single-Gaussian approximation) | `get_convolution()` line 229 |
| `fwhm_planck_tod` | Set to 0 to enforce Planck/prior consistency | `__init__` lines 126–130 |
| Planck TOD input | `components_in_convolved` (not `m_true`) | `get_tod()` lines 333–340 |

The `fwhm_rec` formula is an approximation used only for metadata and notebook comparisons.
The pipeline itself operates on the exact `components_in_convolved`, so it is robust to the
residual gap between the arithmetic mean (0.00582 rad) and the true effective-beam FWHM
(~0.0057 rad empirically).
