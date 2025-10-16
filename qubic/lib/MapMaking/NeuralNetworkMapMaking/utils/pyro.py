import os

import matplotlib.pyplot as plt
import numpy as np
import pyro
import pyro.distributions as dist
import torch
from pyro.infer import SVI, Predictive, Trace_ELBO
from pyro.nn import PyroModule, PyroSample
from pyro.optim import Adam

from qubic.lib.Instrument.Qacquisition import QubicAcquisition
from qubic.lib.Instrument.Qinstrument import QubicInstrument
from qubic.lib.MapMaking.NeuralNetworkMapMaking.operators.forward_ops import ForwardOps
from qubic.lib.Qsamplings import get_pointing
from qubic.lib.Qscene import QubicScene


class InvTransmissionPyro(PyroModule):
    """Pyro module defining per-sample inverse transmission efficiency."""

    def __init__(self, prior_mean=0.0, prior_sigma=0.2):
        super().__init__()
        self.log_eta = PyroSample(dist.Normal(prior_mean, prior_sigma))

    def forward(self, tod_det, transmission):
        eta = torch.exp(self.log_eta)[:, None, None]
        Tprod = transmission.to(tod_det.device, dtype=tod_det.dtype).prod()
        return tod_det / (Tprod * eta)


class InvTransmissionEstimator:
    """
    Bayesian inference for inverse transmission operator calibration using Pyro.
    """

    def __init__(self, qubic_dict, eta=None, transmission=None, prior_mean=0.0, prior_sigma=0.2, lr=3e-3, sigma_noise=1e-18, checkpoint="invT_svi_ckpt.pt", device=None, dtype=torch.float32):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = dtype

        self.qubic_dict = qubic_dict
        self.pointing = get_pointing(qubic_dict)
        self.scene = QubicScene(qubic_dict)
        self.instrument = QubicInstrument(qubic_dict)
        self.acquisition = QubicAcquisition(self.instrument, self.pointing, self.scene, self.qubic_dict)

        # If the user provides an eta array / scalar, set detector efficiency
        if eta is not None:
            eta_arr = np.asarray(eta) * np.ones(self.instrument.detector.efficiency.shape)
            self.instrument.detector.efficiency = eta_arr

        # Transmission as a torch tensor on the requested device/dtype
        if transmission is not None:
            T = np.asarray(transmission) * np.ones(self.instrument.optics.components["transmission"].shape)
            self.transmission = torch.tensor(T, dtype=self.dtype, device=self.device) if isinstance(T, np.ndarray) else torch.tensor([T], dtype=self.dtype, device=self.device)
        else:
            self.transmission = torch.tensor(np.asarray(self.instrument.optics.components["transmission"]), dtype=self.dtype, device=self.device)

        print("True eta (mean) =", float(self.instrument.detector.efficiency.mean()))
        print("Transmission =", self.transmission)

        # hyperparams
        self.prior_mean = float(prior_mean)
        self.prior_sigma = float(prior_sigma)
        self.lr = float(lr)
        self.sigma_noise = float(sigma_noise)
        self.ckpt_file = checkpoint

        # forward ops and deterministic layer
        self.forward_ops = ForwardOps(self.instrument, self.acquisition, self.scene)
        self.layer = InvTransmissionPyro()

        # placeholders set in build_model
        self.model_built = False

    def convolve_map(self, sky_map):
        """Apply convolutions on Sky map.

        Apply convolutions from QUBIC instrument on a sky map.

        Parameters
        ----------
        sky_map : ndarray
            Map of the sky, shape : (Npix, Nstk)
        """

        convolution = self.acquisition.get_convolution_peak_operator()
        return convolution(sky_map)

    def compute_data_from_sky(self, sky_map, N_samples=2):
        """Compute learning datasets.

        Compute learning datasets from a sky map by applying the Qubic operators to obtain tod before and after applying the TransmissionOperator.

        Parameters
        ----------
        sky_map : ndarray
            Map of the sky, shape : (Npix, Nstk)

        Returns
        -------
        tod_before : ndarray
            TOD before TransmissionOperator, shape : (Ndet, Npointing, Nstk)
        tod_after : ndarray
            TOD after TransmissionOperator, shape : (Ndet, Npointing)
        """

        Us = self.forward_ops.op_unit_conversion()(sky_map)
        TUs = Us
        ATUs = self.forward_ops.op_aperture_integration()(TUs)
        FATUs = self.forward_ops.op_filter()(ATUs)
        PFATUs = self.acquisition.get_operator().operands[-1](FATUs)
        HPFATUs = self.forward_ops.op_hwp()(PFATUs)
        PHPFATUs = self.forward_ops.op_polarizer()(HPFATUs)
        APHPFATUs = self.forward_ops.op_detector_integration()(PHPFATUs)
        TAPHPFATUs = self.forward_ops.op_transmission()(APHPFATUs)

        tod_before_list = [APHPFATUs for _ in range(N_samples)]
        tod_after_list = [TAPHPFATUs for _ in range(N_samples)]

        tod_before = torch.tensor(tod_before_list, dtype=self.dtype, device=self.device)
        tod_after = torch.tensor(tod_after_list, dtype=self.dtype, device=self.device)

        return tod_before, tod_after

    def build_model(self):
        def model(tod_det, tod_sky=None):
            # tod_before: (N, D, Nt)
            batch = tod_det.size(0)
            with pyro.plate("batch", batch):
                sky_hat = self.layer(tod_det, self.transmission)
                pyro.sample("obs", dist.Normal(sky_hat, self.sigma_noise).to_event(2), obs=tod_sky)

        self.model = model
        self.guide = pyro.infer.autoguide.AutoNormal(model)
        self.optim = Adam({"lr": self.lr})
        self.svi = SVI(self.model, self.guide, self.optim, Trace_ELBO())
        self.model_built = True

    def load_checkpoint(self):
        if os.path.exists(self.ckpt_file):
            ckpt = torch.load(self.ckpt_file, map_location=self.device)
            pyro.get_param_store().set_state(ckpt.get("param_store", {}))
            if hasattr(self, "optim"):
                try:
                    self.optim.set_state(ckpt.get("optim_state", {}))
                except Exception:
                    pass
            print(f"✓ Resumed from step {ckpt.get('step', 0)}")
            return ckpt.get("step", 0) + 1
        return 0

    def train(self, tod_det, tod_sky, start=0, n_steps=200, save_every=20, use_checkpoint=True):
        assert self.model_built, "Call build_model() first."
        pyro.set_rng_seed(0)

        # ensure tensors are on correct device/dtype
        tod_det = tod_det.to(dtype=self.dtype, device=self.device)
        tod_sky = tod_sky.to(dtype=self.dtype, device=self.device)

        start = self.load_checkpoint() if use_checkpoint else start
        for step in range(start, start + n_steps):
            loss = self.svi.step(tod_det, tod_sky)

            if save_every != 0 and step % save_every == 0:
                torch.save(
                    {
                        "step": step,
                        "param_store": pyro.get_param_store().get_state(),
                        "optim_state": self.optim.get_state(),
                    },
                    self.ckpt_file,
                )
                print(f"step {step:5d} | ELBO {loss:8.3g}  ➜ checkpoint saved")
        print("Training complete ✔")

    def posterior(self, tod_det, num_samples=100, sample_site="log_eta", return_samples=False):
        """Compute posterior mean, std, and samples."""
        assert self.model_built, "Call build_model() first."
        tod_det = tod_det.to(dtype=self.dtype, device=self.device)

        pred = Predictive(self.model, guide=self.guide, num_samples=num_samples, return_sites=[sample_site])
        post = pred(tod_det, tod_sky=None)
        eta_samples = torch.exp(post["log_eta"])

        eta_mean = eta_samples.mean(0).cpu()
        eta_std = eta_samples.std(0).cpu()
        if return_samples:
            return eta_mean, eta_std, eta_samples.cpu()
        return eta_mean, eta_std

    def plot_results(self, eta_mean, eta_std, N_samples):
        plt.figure()
        print(eta_mean.shape)
        plt.errorbar(range(N_samples), eta_mean, yerr=eta_std, fmt="o", capsize=3, color="tab:blue")

        plt.axhline(self.instrument.detector.efficiency.mean(), ls="--", c="k", lw=1)
        plt.ylabel("mean detector efficiency η")
        plt.xlabel("TOD sample ID")
        plt.title("Posterior mean ±1σ per training TOD")
        plt.show()

    def posterior_mean_std_and_samples(self, model, guide, det_tod, n_draw=60, num_samples=10, sample_site="log_eta", keep_draws=False):
        """
        Same as before, but optionally returns a tensor of all stored draws.
        Set keep_draws=True if you want to keep them (this may be large! depends on the choice of size of sempling sets!).
        """
        assert n_draw % num_samples == 0
        mean = m2 = None
        n_seen = 0
        all_draws = [] if keep_draws else None

        pred = Predictive(model, guide=guide, num_samples=num_samples, return_sites=[sample_site])

        for _ in range(n_draw // num_samples):
            post = pred(det_tod, self.transmission, sky=None)
            draws = torch.exp(post[sample_site]).mean(1)  # (chunk, N_det)

            if keep_draws:
                all_draws.append(draws.cpu())

            for x in draws:
                n_seen += 1
                if mean is None:
                    mean = torch.zeros_like(x)
                    m2 = torch.zeros_like(x)
                delta = x - mean
                mean += delta / n_seen
                m2 += delta * (x - mean)

            del post, draws

        std = torch.sqrt(m2 / (n_seen - 1))
        if keep_draws:
            all_draws = torch.cat(all_draws, dim=0)  # (n_draw, N_det)
            return mean.cpu(), std.cpu(), all_draws
        return mean.cpu(), std.cpu()

    def run(self, sky_map):
        """Should we define a run method ??"""
        return None
