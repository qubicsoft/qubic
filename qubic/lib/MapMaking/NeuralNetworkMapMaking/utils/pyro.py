import os

import matplotlib.pyplot as plt
import numpy as np
import pyro
import pyro.distributions as dist
import torch
from pyro.infer import SVI, Predictive, Trace_ELBO
from pyro.nn import PyroModule, PyroSample
from pyro.optim import Adam
from scipy.stats import norm

from qubic.lib.Instrument.Qacquisition import QubicAcquisition
from qubic.lib.Instrument.Qinstrument import QubicInstrument
from qubic.lib.MapMaking.NeuralNetworkMapMaking.operators.forward_ops import ForwardOps
from qubic.lib.Qsamplings import get_pointing
from qubic.lib.Qscene import QubicScene


class InvTransmissionPyro(PyroModule):
    """Pyro module defining per-sample inverse transmission efficiency."""

    def __init__(self, prior_mean, prior_sigma):
        super().__init__()
        self.log_eta = PyroSample(dist.Normal(prior_mean, prior_sigma))

        self.id = "transmission"
        self.param_name = "log_eta"

    def forward(self, tod_det, op_parameters):
        transmission = op_parameters
        eta = torch.exp(self.log_eta)[:, None, None]
        Tprod = transmission.to(tod_det.device, dtype=tod_det.dtype).prod()
        return tod_det / (Tprod * eta)


class InvDetIntegrationPyro(PyroModule):
    def __init__(self, prior_mean, prior_sigma):
        super().__init__()
        self.log_solid_angle = PyroSample(dist.Normal(prior_mean, prior_sigma))

        self.id = "det_integration"
        self.param_name = "log_solid_angle"

    def forward(self, tod_det, op_parameters):
        pos, area, sec_beam = op_parameters

        solid_angle = torch.exp(self.log_solid_angle)[:, None, None]
        theta = np.arctan2(np.sqrt((pos[..., :2] ** 2).sum(-1)), pos[..., 2])
        phi = np.arctan2(pos[..., 1], pos[..., 0])
        sr_det = -area / pos[..., 2] ** 2 * np.cos(theta) ** 3
        gain = sec_beam(theta, phi)

        return tod_det * solid_angle / torch.tensor((sr_det[None, :, None] * gain[None, :, None]))


class InvApertureIntegrationPyro(PyroModule):
    def __init__(self, prior_mean, prior_sigma):
        super().__init__()
        self.log_horn_radius = PyroSample(dist.Normal(prior_mean, prior_sigma))

        self.id = "aperture"
        self.param_name = "log_horn_radius"

    def forward(self, tod_det, op_parameters):
        nhorns = op_parameters
        horn_radius = torch.exp(self.log_horn_radius)[:, None, None]
        return tod_det / (nhorns * torch.pi * horn_radius**2)


class InvEstimatorPyro:
    """
    Bayesian inference for inverse transmission operator calibration using Pyro.

    Goal: pass layer as an argument, to transform the class into a general method to propagate error through any inv operator. I will need to define a method to extract parameters(ex : transmission) and unknowns (ex : eta) for each Inv Operators.
    """

    def __init__(self, qubic_dict, layer, lr=3e-3, sigma_noise=1e-18, checkpoint="inv_ckpt.pt", device=None, dtype=torch.float32):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = dtype

        self.layer = layer
        self.qubic_dict = qubic_dict
        self.pointing = get_pointing(qubic_dict)
        self.scene = QubicScene(qubic_dict)
        self.instrument = QubicInstrument(qubic_dict)
        self.acquisition = QubicAcquisition(self.instrument, self.pointing, self.scene, self.qubic_dict)
        self.forward_ops = ForwardOps(self.instrument, self.acquisition, self.scene)

        # Inverse operator parameter(s)
        if self.layer.id == "transmission":
            self.operators_parameters = torch.tensor(self.instrument.optics.components["transmission"], dtype=self.dtype, device=self.device)
            self.true_value = self.instrument.detector.efficiency.mean()
        elif self.layer.id == "det_integration":
            pos = torch.tensor(self.instrument.detector.center, dtype=self.dtype, device=self.device)
            area = torch.tensor(self.instrument.detector.area, dtype=self.dtype, device=self.device)
            sec_beam = self.instrument.secondary_beam
            self.operators_parameters = [pos, area, sec_beam]
            self.true_value = self.instrument.secondary_beam.solid_angle
        elif self.layer.id == "aperture":
            nhorns = torch.tensor(int(np.sum(self.instrument.horn.open)), dtype=self.dtype, device=self.device)
            self.operators_parameters = nhorns
            self.true_value = float(self.instrument.horn.radeff)
        else:
            raise ValueError(f"Inv Operator {self.layer.id} is not yet implemented.")
        self.op_index = self.id_to_index(self.layer)

        # hyperparams
        self.lr = float(lr)
        self.sigma_noise = float(sigma_noise)
        self.ckpt_file = checkpoint

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

    def id_to_index(self, layer):
        if layer.id == "unit_conversion":
            op_index = 1
        elif layer.id == "atmosphere":
            op_index = 2
        elif layer.id == "aperture":
            op_index = 3
        elif layer.id == "filter":
            op_index = 4
        elif layer.id == "projection":
            op_index = 5
        elif layer.id == "hwp":
            op_index = 6
        elif layer.id == "polarizer":
            op_index = 7
        elif layer.id == "det_integration":
            op_index = 8
        elif layer.id == "transmission":
            op_index = 9
        elif layer.id == "bol_response":
            op_index = 10
        else:
            raise ValueError(f"Inverse operator {layer.id} is not implemented.")

        return op_index

    def compute_tod_list(self, sky_map):
        s = sky_map
        Us = self.forward_ops.op_unit_conversion()(sky_map)
        TUs = Us
        ATUs = self.forward_ops.op_aperture_integration()(TUs)
        FATUs = self.forward_ops.op_filter()(ATUs)
        PFATUs = self.acquisition.get_operator().operands[-1](FATUs)
        HPFATUs = self.forward_ops.op_hwp()(PFATUs)
        PHPFATUs = self.forward_ops.op_polarizer()(HPFATUs)
        APHPFATUs = self.forward_ops.op_detector_integration()(PHPFATUs)
        TAPHPFATUs = self.forward_ops.op_transmission()(APHPFATUs)
        RTAPHPFATUs = self.forward_ops.op_bolometer_response()(TAPHPFATUs)

        return [s, Us, TUs, ATUs, FATUs, PFATUs, HPFATUs, PHPFATUs, APHPFATUs, TAPHPFATUs, RTAPHPFATUs]

    def build_dataset(self, sky_map, N_samples=5):
        tod_list = self.compute_tod_list(sky_map)

        tod_before = torch.tensor([tod_list[self.op_index - 1] for _ in range(N_samples)], device=self.device, dtype=self.dtype)
        tod_after = torch.tensor([tod_list[self.op_index] for _ in range(N_samples)], device=self.device, dtype=self.dtype)

        return tod_before, tod_after

    def build_model(self):
        def model(tod_det, tod_sky=None):
            batch = tod_det.size(0)
            with pyro.plate("batch", batch):
                sky_hat = self.layer(tod_det, self.operators_parameters)
                pyro.sample("obs", dist.Normal(sky_hat, self.sigma_noise).to_event(2), obs=tod_sky)

        self.model = model
        self.guide = pyro.infer.autoguide.AutoNormal(model)
        self.optim = Adam({"lr": self.lr})
        self.svi = SVI(self.model, self.guide, self.optim, Trace_ELBO())
        self.model_built = True

    def load_checkpoint(self):
        if os.path.exists(self.ckpt_file):
            ckpt = torch.load(self.ckpt_file, map_location=self.device, weights_only=False)
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

    def posterior(self, tod_det, num_samples=100, return_samples=False):
        """Compute posterior mean, std, and samples."""
        assert self.model_built, "Call build_model() first."
        tod_det = tod_det.to(dtype=self.dtype, device=self.device)

        pred = Predictive(self.model, guide=self.guide, num_samples=num_samples, return_sites=[self.layer.param_name])
        post = pred(tod_det, tod_sky=None)
        samples = torch.exp(post[self.layer.param_name])

        mean = samples.mean(0).cpu()
        std = samples.std(0).cpu()
        if return_samples:
            return mean, std, samples.cpu()
        return mean, std

    def plot_results(self, samples):
        N_samples = samples.shape[1]
        mean, std = samples.mean(0), samples.std(0)

        plt.figure()
        plt.errorbar(range(N_samples), mean, yerr=std, fmt="o", capsize=3, color="tab:blue")
        plt.axhline(self.true_value, ls="--", c="k", lw=1)
        plt.ylabel("Mean value")
        plt.xlabel("TOD sample ID")
        plt.title("Posterior mean ±1σ per training TOD")
        plt.show()

    def posterior_mean_std_and_samples(self, samples, n_draws=50, keep_draws=False):
        num_samples, _ = samples.shape

        mean = m2 = None
        n_seen = 0
        all_draws = [] if keep_draws else None

        for _ in range(n_draws // num_samples):
            draws = samples.mean(1)  # (chunk, N_det)

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

        std = torch.sqrt(m2 / (n_seen - 1))
        if keep_draws:
            all_draws = torch.cat(all_draws, dim=0)
            return mean.cpu(), std.cpu(), all_draws
        return mean.cpu(), std.cpu()

    def plot_parameter_posteriors(self, draws, nbins=40, credible_interval=0.68, ax=None, show=True, savepath=None):
        """
        Plot posterior samples with Gaussian fit and ±1σ (or custom credible interval).

        Parameters
        ----------
        draws : torch.Tensor or np.ndarray
            Posterior samples for one detector.
        bins : int, optional
            Number of histogram bins.
        credible_interval : float, optional
            Width of shaded credible region (default 68%).
        ax : matplotlib.axes.Axes, optional
            Axis to plot on.
        show : bool, optional
            Whether to display the figure.
        savepath : str, optional
            Path to save the figure.
        """

        if torch.is_tensor(draws):
            draws = draws.detach().cpu().numpy()

        mu, sigma = np.mean(draws), np.std(draws)
        lower, upper = np.quantile(draws, [(1 - credible_interval) / 2, 1 - (1 - credible_interval) / 2])

        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))

        # Histogram of samples
        ax.hist(draws, bins=nbins, density=True, alpha=0.25, color="C0", label="Posterior samples")

        # Normal fit
        xs = np.linspace(draws.min(), draws.max(), 300)
        ax.plot(xs, norm.pdf(xs, mu, sigma), "k--", lw=1.5, label=r"$\mathcal{N}(\mu,\sigma)$ fit")

        # Mean and credible interval
        ax.axvline(self.true_value, ls="-", c="k", lw=1, label=f"True value = {self.true_value:.3f}")
        ax.axvline(mu, color="k", ls="--", lw=1.5, label=f"Mean = {mu:.3f}")
        ax.axvspan(lower, upper, alpha=0.15, color="orange", label=f"{int(credible_interval * 100)}% CI")

        ax.set_xlabel(r"Mean value$")
        ax.set_ylabel("Density")
        ax.set_title("Posterior distribution per detector")
        ax.legend(frameon=False)
        ax.grid(alpha=0.2)
        plt.tight_layout()

        if savepath:
            plt.savefig(savepath, dpi=200, bbox_inches="tight")
        if show:
            plt.show()

    def plot_corrected_tod(self, tod_det, tod_sky, draws, ax=None, show=True, savepath=None):
        """
        Plot Time-Ordered Data (TOD) before and after detector efficiency correction.

        Parameters
        ----------
        tod_det : torch.Tensor
            Observed TOD per detector [batch, det, time].
        tod_sky : torch.Tensor
            True sky TOD per detector [batch, det, time].
        draws : torch.Tensor
            Posterior samples for efficiency η.
        ax : matplotlib.axes.Axes, optional
            Axis to plot on.
        show : bool, optional
            Whether to display the figure.
        savepath : str, optional
            Path to save the figure.
        """

        det_id = 0
        mu_det = draws.mean().item()

        tod_det = tod_det[0, det_id].detach().cpu().numpy()
        tod_true = tod_sky[0, det_id].detach().cpu().numpy()
        tod_corr = tod_det * mu_det

        t = np.arange(len(tod_det))

        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))

        ax.plot(t, tod_det, lw=0.8, alpha=0.5, color="C0", label="Attenuated detector TOD")
        ax.plot(t, tod_corr, lw=1.0, alpha=0.7, color="C3", label=f"Corrected TOD (× {mu_det:.3f})")
        ax.plot(t, tod_true, "--", lw=1.0, color="k", alpha=0.6, label="True sky TOD")

        ax.set_xlabel("Time sample")
        ax.set_ylabel("Power")
        ax.set_title(f"Detector {det_id} – before / after correction")
        ax.legend(frameon=False)
        ax.grid(alpha=0.2)
        plt.tight_layout()

        if savepath:
            plt.savefig(savepath, dpi=200, bbox_inches="tight")
        if show:
            plt.show()

    def plot_corrected_map(self, tod_det, tod_sky, draws, ax=None, show=True, savepath=None):
        stk_id = 0
        mu_det = draws.mean().item()

        tod_det = tod_det[0, :, stk_id].detach().cpu().numpy()
        tod_true = tod_sky[0, :, stk_id].detach().cpu().numpy()
        tod_corr = tod_det * mu_det

        t = np.arange(len(tod_det))

        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))

        ax.plot(t, tod_det, lw=0.8, alpha=0.5, color="C0", label="Attenuated detector TOD")
        ax.plot(t, tod_corr, lw=1.0, alpha=0.7, color="C3", label=f"Corrected TOD (× {mu_det:.3f})")
        ax.plot(t, tod_true, "--", lw=1.0, color="k", alpha=0.6, label="True sky TOD")

        ax.set_xlabel("Time sample")
        ax.set_ylabel("Power")
        ax.set_title("Before / after correction")
        ax.legend(frameon=False)
        ax.grid(alpha=0.2)
        plt.tight_layout()

        if savepath:
            plt.savefig(savepath, dpi=200, bbox_inches="tight")
        if show:
            plt.show()

    def plot_correction(self, tod_det, tod_sky, draws, ax=None, show=True, savepath=None):
        if self.op_index > 5:
            self.plot_corrected_tod(tod_det, tod_sky, draws, ax=ax, show=show, savepath=savepath)
        else:
            self.plot_corrected_map(tod_det, tod_sky, draws, ax=ax, show=show, savepath=savepath)

    def run(self, sky_map, start=0, n_steps=400, save_every=20, use_checkpoint=True, num_samples=200, return_samples=True):
        """
        Run the error propagation from a the given layer over a sky map.

        Parameters
        ----------
        sky_map : np.ndarray
            Input sky map
        start : int, optional
            Starting step (default: 0)
        n_steps : int, optional
            Number of training steps (default: 400)
        save_every : int, optional
            Save checkpoint every n steps (default: 20)
        use_checkpoint : bool, optional
            Whether to use checkpointing (default: True)
        num_samples : int, optional
            Number of posterior samples (default: 200)
        return_samples : bool, optional
            Whether to return samples (default: True)

        Returns
        -------
        mean, std, samples (if return_samples)
        """

        ### Build dataset
        tod_det, tod_sky = self.build_dataset(sky_map)

        ### Build model
        self.build_model()

        ### Train model
        self.train(tod_det, tod_sky, start=start, n_steps=n_steps, save_every=save_every, use_checkpoint=use_checkpoint)

        ### Compute posterior
        post = self.posterior(tod_det, num_samples, return_samples)

        return post
