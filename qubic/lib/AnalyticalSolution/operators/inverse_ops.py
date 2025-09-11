# qubic/lib/AnalyticalSolution/operators/inverse_ops.py
import math

import numpy as np
import torch
import torch.nn as nn
from scipy.constants import c, h, k


def _as_tensor(x, dtype=torch.float32, device=None):
    t = torch.as_tensor(x, dtype=dtype)
    return t.to(device) if device is not None else t


def _broadcast_det(x, det_vector):
    """
    x:  (..., D, Nt)  or  (D, Nt)
    det_vector: (D,)
    returns x * det_vector[None,:,None] broadcasting over batch/time.
    """
    if det_vector.ndim != 1:
        raise ValueError("det_vector must be 1D (D,).")
    view = (1, -1, 1)
    return x * det_vector.view(*view)


# 1) TRANSMISSION ( product(optics) * efficiency)


class InverseTransmissionDeterministic(nn.Module):
    """
    Multiply TOD by 1 / T_det where
    T_det = prod(optics.transmission) * efficiency[det].

    Inputs
    ------
    qubic_instrument: object with
      - optics.components['transmission']  (array-like, length ~18)
      - detector.efficiency                (array-like, length D)

    Forward
    -------
    x: (N, D, Nt) or (D, Nt)
    returns: same shape
    """

    def __init__(self, qubic_instrument, dtype=torch.float32, device=None):
        super().__init__()
        T_optics = float(np.prod(qubic_instrument.optics.components["transmission"]))
        eta = np.array(qubic_instrument.detector.efficiency, dtype=float)  # (D,)
        invT_det = 1.0 / (T_optics * eta)  # (D,)
        self.register_buffer("invT_det", _as_tensor(invT_det, dtype=dtype, device=device))

    def forward(self, x):
        return _broadcast_det(x, self.invT_det)


class InverseTransmissionTrainable(nn.Module):
    """
    Trainable inverse throughput.

    Modes
    -----
    - mode="global_eta": single scalar efficiency shared by all detectors.
        invT_det = 1 / (T_optics * eta)
    - mode="per_detector_eta": one efficiency per detector.
        invT_det[d] = 1 / (T_optics * eta_vec[d])
    - mode="direct_invT": learn invT_det[d] directly (no constraint).

    Initialize from instrument, then call .fit(...) with paired (after,before) TOD.

    Shapes
    ------
    x: (N, D, Nt) or (D, Nt)
    """

    def __init__(self, qubic_instrument, mode="global_eta", dtype=torch.float32, device=None):
        super().__init__()
        if mode not in ("global_eta", "per_detector_eta", "direct_invT"):
            raise ValueError("mode must be 'global_eta' | 'per_detector_eta' | 'direct_invT'.")

        self.mode = mode
        self.dtype = dtype

        self.T_optics = float(np.prod(qubic_instrument.optics.components["transmission"]))
        eta = np.array(qubic_instrument.detector.efficiency, dtype=float)  # (D,)
        D = eta.size

        if mode == "global_eta":
            # initialize at mean efficiency
            self.eta = nn.Parameter(_as_tensor(float(eta.mean()), dtype=dtype, device=device))
        elif mode == "per_detector_eta":
            self.eta_vec = nn.Parameter(_as_tensor(eta, dtype=dtype, device=device))  # (D,)
        else:  # direct_invT
            invT0 = 1.0 / (self.T_optics * eta)
            self.invT_vec = nn.Parameter(_as_tensor(invT0, dtype=dtype, device=device))  # (D,)

    def forward(self, x):
        if self.mode == "global_eta":
            invT_det = 1.0 / (self.T_optics * self.eta)  # scalar
            return x * invT_det
        elif self.mode == "per_detector_eta":
            invT_det = 1.0 / (self.T_optics * self.eta_vec)  # (D,)
            return _broadcast_det(x, invT_det)
        else:
            return _broadcast_det(x, self.invT_vec)

    # ----- trainer ---------
    @torch.no_grad()
    def _infer_device(self, *tensors):
        for t in tensors:
            if isinstance(t, torch.Tensor):
                return t.device
        return None

    def fit(self, tod_after, tod_before, lr=5e-3, epochs=200, weight_decay=0.0, print_every=50):
        """
        Fit parameters to minimize MSE( forward(x_after), x_before )

        tod_after : (N,D,Nt) or (D,Nt)
        tod_before: same shape as tod_after
        """
        device = self._infer_device(tod_after, tod_before) or torch.device("cpu")
        self.to(device)
        tod_after = tod_after.to(device)
        tod_before = tod_before.to(device)

        params = [p for p in self.parameters() if p.requires_grad]
        opt = torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
        loss_hist = []

        for e in range(1, epochs + 1):
            opt.zero_grad()
            pred = self.forward(tod_after)
            loss = torch.mean((pred - tod_before) ** 2)
            loss.backward()
            opt.step()
            loss_hist.append(loss.item())
            if (print_every is not None) and (e % print_every == 0):
                print(f"[InverseTransmissionTrainable] epoch {e:4d}  MSE={loss.item():.3e}")

        return {"loss_history": loss_hist}

    @torch.no_grad()
    def current_invT(self):
        """Return (D,) inverse throughput vector implied by current params."""
        if self.mode == "global_eta":
            return torch.full((1,), 1.0 / (self.T_optics * float(self.eta))).squeeze(0)
        if self.mode == "per_detector_eta":
            return (1.0 / (self.T_optics * self.eta_vec)).detach()
        return self.invT_vec.detach()


# 2) DETECTOR INTEGRATION (solid angle & secondary beam)


class InverseDetectorIntegration(nn.Module):
    """
    Multiply TOD by the inverse of detector-integration factor:
        inv = 1 / (sr_det / sr_beam * sec(theta,phi))   per detector.

    Forward accepts (N,D,Nt) or (D,Nt).
    """

    def __init__(self, qubic_instrument, dtype=torch.float32, device=None):
        super().__init__()
        pos = qubic_instrument.detector.center
        area = qubic_instrument.detector.area
        secb = qubic_instrument.secondary_beam

        theta = np.arctan2(np.sqrt((pos[..., :2] ** 2).sum(-1)), pos[..., 2])
        phi = np.arctan2(pos[..., 1], pos[..., 0])
        sr_det = -area / pos[..., 2] ** 2 * np.cos(theta) ** 3
        sr_beam = secb.solid_angle
        gain = secb(theta, phi)
        inv = 1.0 / (sr_det / sr_beam * gain)  # (D,)
        self.register_buffer("inv_vec", _as_tensor(inv, dtype=dtype, device=device))

    def forward(self, x):
        return _broadcast_det(x, self.inv_vec)


# 3) FILTER


class InverseFilter(nn.Module):
    """
    Divide by bandwidth. If bandwidth=0, act as identity.
    """

    def __init__(self, qubic_instrument, dtype=torch.float32, device=None):
        super().__init__()
        bw = float(qubic_instrument.filter.bandwidth)
        inv = 1.0 if bw == 0.0 else 1.0 / bw
        self.register_buffer("inv_bw", _as_tensor(inv, dtype=dtype, device=device))

    def forward(self, x):
        return x * self.inv_bw


# 4) APERTURE INTEGRATION  — sky space


class InverseApertureIntegration(nn.Module):
    """
    Multiply sky by 1 / (nhorns * pi r_eff^2).
    """

    def __init__(self, qubic_instrument, dtype=torch.float32, device=None):
        super().__init__()
        nhorns = int(np.sum(qubic_instrument.horn.open))
        inv = 1.0 / (nhorns * math.pi * float(qubic_instrument.horn.radeff) ** 2)
        self.register_buffer("inv_ap", _as_tensor(inv, dtype=dtype, device=device))

    def forward(self, x):
        return x * self.inv_ap


# 5) ATMOSPHERE (simple scalar transmission) — TOD space or sky, depending on usage


class InverseAtmosphere(nn.Module):
    """
    Divide by atmospheric transmission (currently just scalar).
    """

    def __init__(self, qubic_acquisition, dtype=torch.float32, device=None):
        super().__init__()
        tau = float(qubic_acquisition.scene.atmosphere.transmission)
        self.register_buffer("inv_tau", _as_tensor(1.0 / tau, dtype=dtype, device=device))

    def forward(self, x):
        return x * self.inv_tau


# 6) UNIT CONVERSION (brightness temp into power) — sky space


class InverseUnitConversion(nn.Module):
    """
    Inverse of scene.get_unit_conversion_operator(nu).
    For differential (non-absolute) scenes, uses linearized Planck law.
    For absolute scenes, provide a small NN placeholder (if you wish to fit).
    """

    def __init__(self, qubic_instrument, qubic_scene, dtype=torch.float32, device=None):
        super().__init__()
        nu = float(qubic_instrument.filter.nu)  # Hz
        Ω = float(qubic_scene.solid_angle)
        self.absolute = bool(qubic_scene.absolute)

        a = 2.0 * Ω * h * nu**3 / c**2  # common factor
        if not self.absolute:
            T = float(qubic_scene.temperature)
            x = h * nu / (k * T)

            # dP/dT (µK_CMB → W m^-2 Hz^-1) linearized; inverse below:
            val = 1e-6 * a * x * np.exp(x) / ((np.expm1(x)) ** 2 * T)
            inv = 1.0 / val
            self.register_buffer("inv_uc", _as_tensor(inv, dtype=dtype, device=device))
            self.nn = None
        else:
            self.nn = _InversePlanckNN().to(device)
            self.register_buffer("inv_uc", _as_tensor(1.0, dtype=dtype, device=device))

    def forward(self, x):
        if self.nn is not None:
            return self.nn(x)  # absolute (nonlinear) case
        return x * self.inv_uc


class _InversePlanckNN(nn.Module):
    """Nonlinear absolute Planck inversion (optional)."""

    def __init__(self, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, x):
        return self.net(x)
