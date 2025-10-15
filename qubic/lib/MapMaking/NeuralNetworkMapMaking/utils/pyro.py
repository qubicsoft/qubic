import gc
import os

import matplotlib.pyplot as plt
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

    def __init__(self, mean=0.0, rel_sigma=0.05):
        super().__init__()
        self.log_eta = PyroSample(dist.Normal(mean, rel_sigma))

    def forward(self, tod_before, transmission):
        eta = torch.exp(self.log_eta)[:, None, None]
        return tod_before / (transmission.prod() * eta)


class InvTransmissionEstimator:
    """
    Bayesian inference for inverse transmission operator calibration using Pyro.
    """

    def __init__(self, qubic_dict, rel_sigma=0.2, lr=3e-3, sigma_noise=1e-18, checkpoint="invT_svi_ckpt.pt", device=None):
        self.qubic_dict = qubic_dict
        self.pointing = get_pointing(qubic_dict)
        self.scene = QubicScene(qubic_dict)
        self.instrument = QubicInstrument(qubic_dict)
        self.acquisition = QubicAcquisition(self.instrument, self.pointing, self.scene, self.qubic_dict)

        self.rel_sigma = rel_sigma
        self.lr = lr
        self.sigma_noise = sigma_noise
        self.ckpt_file = checkpoint
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.transmission = torch.tensor(self.instrument.optics.components["transmission"], dtype=torch.float32, device=self.device)
        pyro.clear_param_store()
        self.forward_ops = ForwardOps(self.instrument, self.acquisition, self.scene)
        self.layer = InvTransmissionPyro(rel_sigma).to(self.device)

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

        tod_before_list, tod_after_list = [], []
        for _ in range(N_samples):
            tod_before_list.append(APHPFATUs)
            tod_after_list.append(TAPHPFATUs)

        tod_before_transmission_torch = torch.tensor(tod_before_list, dtype=torch.float32)
        tod_after_transmission_torch = torch.tensor(tod_after_list, dtype=torch.float32)

        return tod_before_transmission_torch, tod_after_transmission_torch

    def build_model(self):
        def model(tod_before, transmission, tod_after):
            with pyro.plate("batch", tod_before.size(0)):
                sky_hat = self.layer(tod_before, transmission)
                pyro.sample("obs", dist.Normal(sky_hat, self.sigma_noise).to_event(2), obs=tod_after)

        self.model = model
        self.guide = pyro.infer.autoguide.AutoNormal(model)
        self.optim = Adam({"lr": self.lr})
        self.svi = SVI(self.model, self.guide, self.optim, Trace_ELBO())
        self.model_built = True

    def load_checkpoint(self):
        if os.path.exists(self.ckpt_file):
            ckpt = torch.load(self.ckpt_file, map_location=self.device, weights_only=False)
            pyro.get_param_store().set_state(ckpt["param_store"])
            self.optim.set_state(ckpt["optim_state"])
            print(f"✓ Resumed from step {ckpt['step']}")
            return ckpt["step"] + 1
        return 0

    def train(self, tod_before, tod_after, start=0, n_steps=200, save_every=20, resume=True):
        assert self.model_built, "Call build_model() first."
        pyro.set_rng_seed(0)

        start = self.load_checkpoint() if resume else start
        for step in range(start, start + n_steps):
            loss = self.svi.step(tod_before, self.transmission, tod_after)

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
            gc.collect()
        print("Training complete ✔")

    def posterior(self, tod_before, n_draw=100, chunk=10, sample_site="log_eta"):
        """Compute posterior mean, std, and samples."""
        pred = Predictive(self.model, guide=self.guide, num_samples=chunk, return_sites=[sample_site])
        post = pred(tod_before, self.transmission, tod_after=None)
        eta_samples = torch.exp(post["log_eta"])  # (300 , N)

        eta_mean = eta_samples.mean(0).cpu()  # (N,)
        eta_std = eta_samples.std(0).cpu()
        return eta_mean, eta_std

    def plot_results(self, eta_mean, eta_std, N_samples, true_eta=None):
        plt.figure()
        print(eta_mean.shape)
        plt.errorbar(range(N_samples), eta_mean, yerr=eta_std, fmt="o", capsize=3, color="tab:blue")
        if true_eta is not None:
            plt.axhline(true_eta, ls="--", c="k", lw=1)
        plt.ylabel("mean detector efficiency η")
        plt.xlabel("TOD sample ID")
        plt.title("Posterior mean ±1σ per training TOD")
        plt.show()

    def run(self, sky_map):
        """Should we define a run method ??"""
        return None
