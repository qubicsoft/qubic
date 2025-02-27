import numpy as np
import matplotlib.pyplot as plt


class Q1D:
    def __init__(self, instrument_type="BI-GPA", params=None, plot_sb=True):
        """
        Instanciation of the class with default parameters.
        If a dictionnary param is given it replaces the default parameters.
        """
        if params is None:
            params = {}

        # Sky Description
        self.npix_sky = params.get("npix_sky", 2**15)
        self.sky_model = params.get("sky_model", [rnd_sky_1d, 1.8])  # Modèle fictif
        self.sky_input_method = params.get("sky_input_method", "BWconv")

        # Pointings
        self.npointings = params.get("npointings", 50000)

        # Instrument
        self.instrument_type = params.get("instrument_type", instrument_type)
        self.detpos = params.get("detpos", np.array([0.0]))
        self.fwhmprim_150 = params.get("fwhmprim_150", 14.0)
        self.nu = params.get("nu", 150e9)
        self.dist_horns = params.get("dist_horns", 14.0e-3)
        self.sqnh = params.get("sqnh", 8)
        self.Df = params.get("Df", 1.0)
        self.kmax_build = params.get("kmax_build", 2)

        # SB sampling
        self.ntheta = params.get("ntheta", 2**15)
        self.minmaxtheta = params.get("minmaxtheta", 30)
        self.th = np.linspace(-self.minmaxtheta, self.minmaxtheta, self.ntheta)

        # TOD fabrication method
        self.TOD_method = params.get("TOD_method", "approx")

        # Noise
        self.noise_rms = params.get("noise_rms", 0.0)

        # Sky Reconstruction
        self.kmax_rec = params.get("kmax_rec", self.kmax_build)
        self.npix = params.get("npix", 360 * 2)
        pixels = np.linspace(-180, 180, self.npix + 1)
        self.rec_pix_centers = 0.5 * (pixels[:-1] + pixels[1:])  # Compute pixel centers

        # Seeds
        self.sky_seed = params.get("sky_seed", 1)
        self.pointing_seed = params.get("pointing_seed", 1)
        self.noise_seed = params.get("noise_seed", None)

        # Plottings
        self.plot_sky = params.get("plot_sky", False)
        self.plot_sb = params.get("plot_sb", plot_sb)
        self.plot_expected = params.get("plot_expected", False)
        self.plot_pointings = params.get("plot_pointings", False)
        self.plot_TOD = params.get("plot_TOD", False)
        self.plot_reconstructed = params.get("plot_reconstructed", True)
        self.plot_H = params.get("plot_H", False)
        self.plot_HtH = params.get("plot_HtH", False)

        # Prepare the synthesized beam
        self.prepare_synthesized_beam()

        # Things that will be filled later
        self.input_sky = None
        self.ptg_deg = None
        self.TOD = None
        self.convolved_sky = None
        self.convolved_sky_pix = None
        self.SynthBeam_convolved = None
        self.SynthBeam_convolved_pix = None
        self.H = None

    def create_sky(self):
        """Create a synthetic sky realization based on input parameters.

        This function generates a sky realization using a specified sky model and
        optional plotting parameters. The generated sky is normalized to have unit
        standard deviation.

        Parameters
        ----------
        None : taken from the class attibutes

        Returns
        -------
        None: updates the class attribute self.input_sky

        """
        np.random.seed(self.sky_seed)  # Set the random seed for reproducibility
        xpix, truey = create_1d_sky(self.npix_sky, self.sky_model[0], self.sky_model[1:])
        if self.plot_sky:
            plt.figure()
            plt.plot(xpix, truey, label="True Sky")
            plt.xlabel("angle")
            plt.ylabel("sky")
            plt.legend()

        self.input_sky = [xpix, truey]

    def prepare_synthesized_beam(self):
        """Prepare the synthesized beam and related quantities for an instrument.

        This function computes the synthesized beam of an instrument based on the detector
        positions and primary beam properties, and optionally plots the results.

        Parameters
        ----------
        None : taken from the class attibutes

        Returns
        -------
        sb : ndarray
            The synthesized beam values at the specified angular positions `th`.
        fwhmpeak : float
            The FWHM of the synthesized beam.
        thetapeaks : ndarray
            The angular positions of the peaks in the synthesized beam.
        amppeaks : ndarray
            The amplitudes of the peaks in the synthesized beam.
        ratio : float
            The ratio of the synthesized beam to a Gaussian peak approximation.

        """
        ndet = len(self.detpos)
        if self.instrument_type not in ["Imager", "BI", "BI-GPA"]:
            print("Wrong instrument type")
            stop

        if self.plot_sb:
            plt.figure()
            plt.plot(
                self.th,
                get_primbeam(self.th, 3e8 / self.nu, fwhmprimbeam_150=self.fwhmprim_150),
                "k--",
                label="Th. Prim. Beam at {0:3.0f} GHz".format(self.nu / 1e9),
            )

        self.b = []
        self.fwhmpeak = []
        self.thetapeaks = []
        self.amppeaks = []
        self.ratio = []
        for d in self.detpos:
            b, fwhmpeak, thetapeaks, amppeaks, ratio = give_sbcut(
                self.th,
                self.dist_horns,
                3e8 / self.nu,
                self.sqnh,
                Df=self.Df,
                detpos=d / 1000,
                fwhmprimbeam_150=self.fwhmprim_150,
                kmax=self.kmax_build,
                type=self.instrument_type,
            )

            self.b.append(b)
            self.fwhmpeak.append(fwhmpeak)
            self.thetapeaks.append(np.array(thetapeaks))
            self.amppeaks.append(np.array(amppeaks))
            self.ratio.append(ratio)

            if self.plot_sb:
                p = plt.plot(
                    self.th,
                    b,
                    lw=2,
                    label="Th. Synth. Beam at {0:3.0f} GHz \n detpos={1:3.1f} mm - Ratio G.P.A. = {2:5.2f} \n FWHM = {3:5.2f} deg".format(
                        self.nu / 1e9, d, ratio, fwhmpeak
                    ),
                )
                plt.plot(thetapeaks, amppeaks / np.max(amppeaks), "o", color=p[0].get_color())
        if self.plot_sb:
            plt.xlabel(r"$\theta$ [deg.]")
            plt.ylabel("Synthesized beam")
            plt.legend(loc="upper left", fontsize=8)
            plt.xlim(np.min(self.th), np.max(self.th))
            plt.ylim(-0.01, 1.4)
            plt.title("Synthesized Beam: " + self.instrument_type)

    def build_TOD(self):
        """
        Build Time-Ordered Data (TOD) from a sky model, instrument configuration, and pointing information.

        This function generates TOD by either exactly convolving a given sky model with the detector's synthesized beam
        and interpolating the result at the pointing locations, or by approximating the process using the "H operator" approach
        with additional steps for angular resolution and pixel window functions.

        Parameters
        ----------
        None : taken from the class attributes

        Returns
        -------
        None : updates the class attributes `TOD` and `convolved_sky`
        """

        # Reform input sky and corresponding angles
        xpix = self.input_sky[0]  # Sky pixel positions
        truey = self.input_sky[1]  # True sky model values

        # Calculate the number of peaks in the synthesized beam based on instrument type
        if self.instrument_type == "Imager":
            npeaks = 1
        else:
            npeaks = 2 * self.kmax_build + 1

        # Initialize the TOD array with zeros (ndet x npointings)
        ndet = len(self.detpos)
        TOD = np.zeros((ndet, self.npointings))

        # Calculate the sampling interval of the input sky
        sampling = xpix[1] - xpix[0]

        # Calculate the width of the reconstructed pixels
        output_pixels_width = self.rec_pix_centers[1] - self.rec_pix_centers[0]

        # Loop over each detector to build its TOD
        for i in range(ndet):

            # Get the synthesized beam for the current detector
            b = self.b[i]
            fwhmpeak = self.fwhmpeak[i]
            thetapeaks = np.zeros(npeaks) + self.thetapeaks[i]
            amppeaks = np.zeros(npeaks) + self.amppeaks[i]

            # Normalize the peak amplitudes of the synthesized beam
            amppeaks = amppeaks / np.sum(amppeaks)

            # Choose the algorithm for TOD generation
            if self.TOD_method == "exact":
                # Step 1: Convolve the true sky with the full synthesized beam
                convolved = conv_circ(truey, norm_filt(xpix, self.th, b))

                # Step 2: If requested, reconvolve the sky with the reconstructed pixels' window function
                if self.sky_input_method == "BWconv":
                    convolved = convolve_fourier_with_rectangular(convolved, output_pixels_width, sampling)

                # Step 3: Interpolate the convolved sky at the pointing locations
                convolved_pix = np.interp(self.rec_pix_centers, xpix, convolved)
                TOD[i, :] = np.interp(self.ptg_deg, xpix, convolved)
            else:
                # Step 1: Convolve the true sky with the detector's angular resolution
                bgauss = np.exp(-0.5 * (self.th / (fwhmpeak / np.sqrt(8 * np.log(2)))) ** 2)
                convolved = conv_circ(truey, norm_filt(xpix, self.th, bgauss))
                # Step 2: If needed, reconvolve the resolution-convolved sky with the reconstructed pixels' window function
                if self.sky_input_method == "BWconv":
                    convolved = convolve_fourier_with_rectangular(convolved, output_pixels_width, sampling)
                # Step 3: Interpolate the convolved sky at the reconstructed pixel center locations
                convolved_pix = np.interp(self.rec_pix_centers, xpix, convolved)

                if self.TOD_method == "approx":
                    # Step 4: Sum the contributions from each peak in the synthesized beam
                    for k in range(npeaks):
                        indices = np.floor(((self.ptg_deg - thetapeaks[k] + 180) * self.npix / 360)).astype(int) % self.npix
                        TOD[i, :] += convolved_pix[indices] * amppeaks[k]
                elif self.TOD_method == "approx_Hd":
                    self.build_H_operator(self.kmax_build)
                    H = self.H
                    for i in range(ndet):
                        TOD[i, :] = np.dot(H[i, :, :], convolved_pix)
                else:
                    print("Wrong algorithm (TOD_method parameter to build_TOD())")
                    stop

            self.SynthBeam_convolved = convolved
            self.SynthBeam_convolved_pix = convolved_pix

            # Step 5: Add noise to the TOD for the current detector
            np.random.seed(self.noise_seed)  # Initialize the random seed for reproducibility
            TOD[i, :] += np.random.randn(self.npointings) * self.noise_rms

            if self.plot_TOD:
                plt.figure()
                plt.plot(xpix, truey, label="True Sky")
                plt.plot(xpix, convolved, label="Convolved Sky")
                plt.errorbar(
                    self.rec_pix_centers,
                    convolved_pix,
                    xerr=(self.rec_pix_centers[1] - self.rec_pix_centers[0]),
                    label="Pixellized convolved sky",
                    fmt="o",
                    capsize=3,
                )
                plt.plot(self.ptg_deg, TOD[i, :], ".", label="TOD")
                plt.xlabel(r"$\theta$ [deg.]")
                plt.ylabel("TOD")
                plt.title("TOD for Detector {0}".format(i))
                plt.legend()

        # We assign the TOD to the class corresponding class attribute
        self.TOD = TOD

    def build_H_operator(self, kmax):
        nptg = self.npointings
        npix = self.npix
        ndet = len(self.detpos)  # Number of detectors
        # Calculate the number of peaks in the synthesized beam based on instrument type
        if self.instrument_type == "Imager":
            npeaks = 1
        else:
            npeaks = 2 * kmax + 1

        H = np.zeros((ndet, nptg, npix))
        for i in range(ndet):
            thetapeaks = np.zeros(npeaks) + self.thetapeaks[i]
            amppeaks = np.zeros(npeaks) + self.amppeaks[i]
            for j in range(npeaks):
                peaks_ptg = (self.ptg_deg - thetapeaks[j] + 180 + 360) % 360 - 180
                peaks_indices = np.floor(((peaks_ptg - (-180)) * npix / 360)).astype(int)
                for k in range(nptg):
                    H[i, k, peaks_indices[k]] = amppeaks[j] / np.sum(amppeaks)

        if self.plot_H:
            plt.figure()
            plt.imshow(H[0], aspect="auto", interpolation="nearest", origin="lower")
            plt.xlabel("Rec pixel")
            plt.ylabel("Time Sample")
            plt.title("H operator for detector 0")
            plt.colorbar()

        self.H = H

    def compute_expected_sky(self):
        # Reform input sky and corresponding angles
        xpix = self.input_sky[0]  # Sky pixel positions
        truey = self.input_sky[1]  # True sky model values

        bgauss = np.exp(-0.5 * (self.th / (self.fwhmpeak / np.sqrt(8 * np.log(2)))) ** 2)
        Bconvy = conv_circ(truey, norm_filt(xpix, self.th, bgauss))
        Bconvy_pix = np.interp(self.rec_pix_centers, xpix, Bconvy)  # Interpolate at pixel centers

        dpix = self.rec_pix_centers[1] - self.rec_pix_centers[0]  # Rec pixel width

        if self.sky_input_method == "Bconv":
            Ctruth, Ctruth_pix = Bconvy, Bconvy_pix
        elif self.sky_input_method == "BWconv":
            BWconvy = convolve_fourier_with_rectangular(Bconvy, dpix, xpix[1] - xpix[0])
            BWconvy_pix = np.interp(self.rec_pix_centers, xpix, BWconvy)
            Ctruth, Ctruth_pix = BWconvy, BWconvy_pix
        else:
            raise ValueError("Incorrect <sky_input_method> in parameters")

        # Plot expected signal if required
        if self.plot_expected:
            plt.figure()
            plt.title("Expected signal: " + self.instrument_type)
            plt.plot(xpix, truey, label="True Sky")
            plt.plot(xpix, Ctruth, label="Beam-convolved sky")
            plt.errorbar(self.rec_pix_centers, Ctruth_pix, xerr=dpix / 2, fmt=".", label="Pixellized convolved sky")
            plt.legend()

        # No return value, we assign the relevant quantities to class attributes
        self.convolved_sky = Ctruth
        self.convolved_sky_pix = Ctruth_pix

    def generate_pointings(self):
        np.random.seed(self.pointing_seed)
        self.ptg_deg = np.random.random(self.npointings) * 360 - 180

        # Plot histogram of pointings if required
        if self.plot_pointings:
            plt.figure()
            plt.hist(self.ptg_deg, range=[-180, 180], bins=90, label="Pointings")
            plt.xlim(-180, 180)
            plt.legend()
            plt.xlabel(r"$\theta$ [deg.]")
            plt.title("Pointings")

    def mapmaking_solution(self, H, TOD):
        npix = self.npix  # Pixels in the sky map
        ndet = len(self.detpos)  # Number of detectors
        # Compute inverse covariance matrix and reconstruct sky map
        HtH = np.zeros((ndet, npix, npix))
        HtHinv = np.zeros((ndet, npix, npix))
        solution = np.zeros((ndet, npix))

        for i in range(ndet):
            HtH[i] = np.dot(H[i].T, H[i])
            HtHinv[i] = np.linalg.inv(HtH[i])
            solution[i] = np.dot(HtHinv[i], np.dot(H[i].T, TOD[i]))

        if self.plot_HtH:
            plt.figure()
            plt.subplot(1, 2, 1)
            plt.imshow(HtH[0, :, :], aspect="equal", interpolation="nearest", origin="lower")
            plt.xlabel("Rec pixel")
            plt.ylabel("Rec pixel")
            plt.title("$H^t\cdot H$ for detector 0")
            plt.colorbar()

            plt.subplot(1, 2, 2)
            plt.imshow(HtHinv[0, :, :], aspect="equal", interpolation="nearest", origin="lower")
            plt.xlabel("Rec pixel")
            plt.ylabel("Rec pixel")
            plt.title("$(H^t\cdot H)^{-1}$" + " for detector 0")
            plt.colorbar()
            plt.tight_layout()

        # Compute average solution across detectors
        solution_all = np.mean(solution, axis=0)

        if self.plot_reconstructed:
            # Compute residuals
            res_all = solution_all - self.convolved_sky_pix
            ss_all = np.std(res_all)

            # Plot reconstructed sky map and residuals if required
            plt.figure()
            plt.subplot(2, 1, 1)
            for i in range(ndet):
                plt.plot(self.rec_pix_centers, solution[i], "o", alpha=0.5, label=f"Reconstructed Detector #{i}")
                plt.plot(self.rec_pix_centers, self.convolved_sky_pix, ".-", lw=2, color="r", alpha=0.8, label="Sky convolved")
                plt.plot(self.input_sky[0], self.input_sky[1], alpha=0.8, label="Input sky")
            plt.legend()
            plt.axhline(y=0, ls="--", color="k")
            plt.xlabel("Angle")
            plt.ylabel("Sky Data")
            plt.title(f"Instrument {self.instrument_type}")

            plt.subplot(2, 2, 3)
            plt.plot(self.rec_pix_centers, res_all, "k-", label=f"Residual w.r.t. Sky Convolved: {ss_all:.3g}")
            plt.axhline(y=0, ls="--", color="k")
            plt.xlabel("Angle")
            plt.ylabel("Residuals")
            plt.legend()

            plt.subplot(2, 2, 4)
            plt.hist(res_all, bins=21, range=[-5 * ss_all, 5 * ss_all], alpha=0.5, label=f"Residual: {ss_all:.3g}")
            plt.xlabel("Residuals")
            plt.ylabel("Counts")
            plt.legend()
            plt.tight_layout()

        return solution, solution_all

    def simulate(self):
        # Generate the input sky
        self.create_sky()

        # Compute theoretical predictions for the reconstructed sky
        self.compute_expected_sky()

        # Generate random pointing directions
        self.generate_pointings()

        # Generate Time-Ordered Data (TOD)
        self.build_TOD()

        return self.TOD

    def reconstruct(self):
        # Compute H operator
        self.build_H_operator(self.kmax_rec)

        # Compute solution
        solution, solution_all = self.mapmaking_solution(self.H, self.TOD)

        # Compute residuals
        res_all = solution_all - self.convolved_sky_pix
        ss_all = np.std(res_all)

        return {"solution": solution, "solution_all": solution_all, "RMS_resall": ss_all}

    def simulate_and_reconstruct(self):
        """
        Runs a 1D sky reconstruction simulation using a given instrument configuration.

        This function simulates the observation of a 1D sky map with an instrument
        characterized by a synthesized beam. It computes the expected signal, generates
        pointing directions, constructs the Time-Ordered Data (TOD), applies the H operator,
        and reconstructs the sky map. The function also calculates residuals and provides
        visualization options.

        Parameters:
        -----------
        None : taken from the class attributes

        Returns:
        --------
        Only returns a few information, the rest is updated in the class attributes
            - result: a dictionary conatining `solution` (for all detectors) and `solution_all` (the detector averaged solution)
        """
        # simulate data
        _ = self.simulate()

        # Reconstruct
        result = self.reconstruct()

        return result


########################################################################################
def rnd_sky_1d(xpix, args):
    """Generate a 1D random sky realization in Fourier space.

    This function generates a 1D random realization of a sky signal in Fourier space.
    The process involves creating a random noise in Fourier space, applying a spectral
    power-law filter, and transforming it back to real space using an inverse Fourier transform.

    Parameters
    ----------
    xpix : array_like
        Input array representing pixel positions (e.g., angular or spatial positions).
        The length of this array determines the number of Fourier modes.
    args : list
        A list of parameters used for the power spectrum. `args[0]` is expected to
        be the exponent used in the spectral filter (e.g., the power-law exponent).

    Returns
    -------
    y : ndarray
        A 1D array representing the random sky realization in real space. The output
        has the same length as `xpix`.

    Notes
    -----
    - The function uses a random Gaussian field in Fourier space and then applies
      a spectral power-law filter to modify its power spectrum.
    - The input `xpix` is used to compute the Fourier frequencies, and the filtering
      is done using the `spec_1_f` function, which is expected to apply a power-law spectrum.
    - The output is in real space, and the function uses the inverse Fourier transform
      (`np.fft.ifft`) to return the real-space realization.

    """

    # Compute the Fourier frequencies based on the pixel positions
    ff = np.fft.fftfreq(len(xpix), xpix[1] - xpix[0])

    # Generate random noise in Fourier space (Gaussian random field)
    rndy = np.random.randn(len(xpix))

    # Perform a forward Fourier transform to get the frequency domain representation
    ftyin = np.fft.fft(rndy)

    # Apply the spectral filter in Fourier space using the spec_1_f function
    fty = ftyin * spec_1_f(ff, args[0])

    # Perform the inverse Fourier transform to return to real space
    y = np.real(np.fft.ifft(fty))

    # Return the generated random sky realization in real space
    return y


########################################################################################
def spec_1_f(ff, pow):
    """Returns a power law.

    This function takes frequencies and a power as an input and returns a power law

    Parameters
    ----------
    ff : array_like
        Input array of frequencies
    pow : float
        The exponent used in the power-law transformation. For each non-zero element `x`,
        the transformation is `|x|^(-pow)`.

    Returns
    -------
    s : ndarray
        An array of the same shape as `ff`, where non-zero values are transformed
        using the power law, and zero values are preserved as zeros.

    Notes
    -----
    - If any element of `ff` is zero, it is preserved as zero in the output array `s`.
    - The `np.nan_to_num()` function is used to handle any potential NaN values that may
      result from dividing by zero (though the zero-check ensures this doesn't happen).
    """

    zz = ff != 0
    s = np.zeros_like(ff)
    s[zz] = np.nan_to_num(np.abs(ff[zz]) ** (-pow))
    s[~zz] = 0
    return s


########################################################################################
def sinsky(th_deg, args):
    """Generate a sinusoidal sky pattern.

    This function generates a sinusoidal pattern based on the input angle `th_deg`
    and a scaling factor from `args[0]`. The sinusoidal pattern is defined by
    `sin(args[0] * theta)`, where `theta` is the input angle in degrees.

    Parameters
    ----------
    th_deg : array_like
        Input array of angles in degrees. The function computes the sine of each angle
        scaled by the factor in `args[0]`.
    args : list
        A list of parameters, where `args[0]` is the scaling factor for the angle.

    Returns
    -------
    result : ndarray
        An array of the same shape as `th_deg`, representing the sinusoidal values
        of the scaled angles.

    Notes
    -----
    - The input angles `th_deg` are assumed to be in degrees, and they are first
      converted to radians before applying the sine function.
    - The scaling factor in `args[0]` is used to modify the frequency of the sine wave.
    """

    # Compute the sinusoidal values using the input angle and scaling factor
    return np.sin(np.radians(th_deg) * args[0])


########################################################################################
def create_1d_sky(npix, funct, args):
    """Create a 1D sky model with a specified function.

    This function generates a 1D sky realization by evaluating a given function
    over a range of pixel values. The result is normalized by its standard deviation.

    Parameters
    ----------
    npix : int
        The number of pixels (or samples) in the 1D sky model.
    funct : function
        A function that takes an array of pixel positions and additional arguments
        to generate the sky model.
    args : list
        A list of arguments passed to the function `funct`.

    Returns
    -------
    xpix : ndarray
        An array of pixel positions (e.g., angles or spatial coordinates).
    truey : ndarray
        The generated sky model, normalized to have unit standard deviation.

    """
    xpix = np.linspace(-180, 180, npix)
    truey = funct(xpix, args)
    truey = truey / np.std(truey)  # Normalize sky model
    return xpix, truey


########################################################################################
def get_primbeam(th, lam, fwhmprimbeam_150=14.0):
    """Compute the primary beam of the instrument.

    This function calculates the primary beam pattern based on a Gaussian approximation
    with a full-width at half-maximum (FWHM) that scales with wavelength.

    Parameters
    ----------
    th : array_like
        Array of angular positions (in degrees) where the primary beam is evaluated.
    lam : float
        Wavelength of the signal (in meters).
    fwhmprimbeam_150 : float, optional
        The FWHM of the primary beam at 150 GHz in degrees. Default is 14.

    Returns
    -------
    primbeam : ndarray
        The primary beam values at the specified angles `th`.

    """
    fwhmprim = fwhmprimbeam_150 * lam / (3e8 / 150e9)  # Scale FWHM for given wavelength
    primbeam = np.exp(-0.5 * th**2 / (fwhmprim / np.sqrt(8 * np.log(2))) ** 2)
    return primbeam


########################################################################################
def give_sbcut(th, dx, lam, sqnh, Df=1.0, detpos=0.0, fwhmprimbeam_150=14.0, kmax=2, type="BI"):
    """Calculate the synthesized beam and related quantities.

    This function computes the synthesized beam of an instrument based on various parameters,
    including the detector positions and the primary beam. It also calculates the ratio
    of the beam to a Gaussian peak approximation.

    Parameters
    ----------
    th : array_like
        Array of angular positions (in degrees).
    dx : float
        Pixel size or spacing (in arcminutes).
    lam : float
        Wavelength of the signal (in meters).
    sqnh : float
        The scaling factor related to the pixel size and wavelength.
    Df : float, optional
        The frequency width, default is 1.
    detpos : float, optional
        Position of the detector (in meters), default is 0.
    fwhmprimbeam_150 : float, optional
        FWHM of the primary beam at 150 GHz, default is 14.
    kmax : int, optional
        Parameter controlling the number of peaks to consider, default is 2.
    type : {'BI', 'BI-GPA', 'Imager'}, optional
        The type of instrument, default is 'BI'.
        BI: stands for a bolometric interferometer
        BI-GPA: stands for a bolometric interferometer with Gaussian peak approximation
        Imager: stands for an imager

    Returns
    -------
    sb : ndarray
        The synthesized beam values at the specified angles `th`.
    fwhmpeak : float
        The full width at half maximum (FWHM) of the synthesized beam.
    thetapeaks : ndarray
        The angular positions of the synthesized beam peaks.
    amppeaks : ndarray
        The amplitudes of the synthesized beam peaks.
    ratio : float
        The ratio of the synthesized beam to the Gaussian peak approximation.

    """
    # Compute the primary beam pattern
    primbeam = get_primbeam(th, lam, fwhmprimbeam_150=fwhmprimbeam_150)

    # Compute the synthesized beam taking care of divisions by zero
    theth = th - np.degrees(detpos / Df)
    arg_to_sin = np.pi * dx / lam * np.radians(theth)
    sb = np.zeros(len(arg_to_sin))
    ok = np.sin(arg_to_sin) != 0
    sb[ok] = np.sin(sqnh * arg_to_sin[ok]) ** 2 / np.sin(arg_to_sin[ok]) ** 2
    sb[~ok] = 1
    # Apply the primary beam
    sb = sb / sqnh**2 * primbeam  # Normalize by scaling factor

    # Find peaks
    fwhmpeak = np.degrees(lam / sqnh / dx)  # FWHM of the synthesized beam
    thetapeaks = np.degrees(lam / dx)
    allthetapeaks = thetapeaks * (np.arange(2 * kmax + 1) - kmax) + np.degrees(detpos)
    allamppeaks = np.interp(allthetapeaks, th, primbeam)

    # Gaussian peak approximation
    gauss = np.zeros_like(th)
    for i in range(len(allthetapeaks)):
        gauss += allamppeaks[i] * np.exp(-0.5 * (th - allthetapeaks[i]) ** 2 / (fwhmpeak / np.sqrt(8 * np.log(2))) ** 2)

    # Compute the ratio of the synthesized beam to the Gaussian peak approximation
    ratio = np.sum(sb * gauss) / np.sum(gauss * gauss)

    if type == "BI-GPA":
        sb = gauss.copy()
        ratio = 1
    elif type == "Imager":
        allthetapeaks = np.degrees(detpos)
        allamppeaks = np.interp(allthetapeaks, th, primbeam)
        ratio = 1.0
        gauss = allamppeaks * np.exp(-0.5 * (th - allthetapeaks) ** 2 / (fwhmpeak / np.sqrt(8 * np.log(2))) ** 2)
        sb = gauss.copy()

    return sb, fwhmpeak, np.array(allthetapeaks), np.array(allamppeaks), ratio


########################################################################################
def conv_circ(signal, ker):
    """Convolve a signal with a kernel using circular convolution in Fourier space.

    This function computes the convolution of a 1D signal with a kernel using the
    Fourier domain, then performs a circular shift to match the input signal's alignment.

    Parameters
    ----------
    signal : ndarray
        The 1D input signal to be convolved.
    ker : ndarray
        The 1D kernel used for the convolution. It must have the same shape as `signal`.

    Returns
    -------
    result : ndarray
        The convolved signal after applying the kernel.

    Notes
    -----
    - The convolution is performed in Fourier space and then shifted back to real space.
    - The kernel and signal must have the same length for the convolution to be valid.

    """
    return np.roll(np.real(np.fft.ifft(np.fft.fft(signal) * np.fft.fft(ker))), len(signal) // 2)


########################################################################################
def norm_filt(newx, xfilt, filt):
    """Normalize a filter by its sum.

    This function interpolates a filter to a new set of `x` values and normalizes
    the filter by dividing by the sum of its elements.

    Parameters
    ----------
    newx : array_like
        The new set of x values where the filter will be interpolated.
    xfilt : array_like
        The x values corresponding to the original filter.
    filt : array_like
        The original filter values to be normalized.

    Returns
    -------
    norm_filt : ndarray
        The normalized filter values at the new `x` positions.

    """
    interp_filter = np.interp(newx, xfilt, filt)
    norm = np.sum(interp_filter)
    return interp_filter / norm


########################################################################################
def convolve_fourier_with_rectangular(time_stream, rect_width, sampling_interval=1):
    """Convolve a 1D time-stream with a rectangular function using Fourier space.

    This function performs a convolution of the input time-stream with a rectangular
    function by transforming both the time-stream and the kernel to Fourier space,
    multiplying them, and then transforming the result back to real space.

    Parameters
    ----------
    time_stream : np.array
        The 1D input time-stream to be convolved.
    rect_width : float
        The width of the rectangular function in time units.
    sampling_interval : float, optional
        The time interval between samples in the time-stream, default is 1.

    Returns
    -------
    np.array
        The convolved time-stream after applying the rectangular filter.

    """
    n = len(time_stream)
    freqs = np.fft.fftfreq(n, d=sampling_interval)  # Frequency array
    time_stream_fft = np.fft.fft(time_stream)  # Fourier transform of the time-stream
    sinc_kernel = rect_width * np.sinc(freqs * rect_width)  # Kernel in Fourier space
    convolved_fft = time_stream_fft * sinc_kernel  # Apply kernel in Fourier space
    convolved_signal = np.fft.ifft(convolved_fft).real  # Inverse Fourier transform
    convolved_signal *= 1 / rect_width  # Normalize by kernel width
    return convolved_signal

    ############ TO REMOVE
    ########################################################################################
    # def create_sky(params):
    """Create a synthetic sky realization based on input parameters.

    This function generates a sky realization using a specified sky model and
    optional plotting parameters. The generated sky is normalized to have unit
    standard deviation.

    Parameters
    ----------
    params : dict
        A dictionary of parameters including `sky_seed`, `npix_sky`, `sky_model`,
        and `plot_sky` for plotting options.

    Returns
    -------
    xpix : ndarray
        Array of pixel positions used to generate the sky.
    truey : ndarray
        The generated 1D sky model.

    """
    np.random.seed(params["sky_seed"])  # Set the random seed for reproducibility
    xpix, truey = create_1d_sky(params["npix_sky"], params["sky_model"][0], params["sky_model"][1:])
    if params["plot_sky"]:
        plt.figure()
        plt.plot(xpix, truey, label="True Sky")
        plt.xlabel("angle")
        plt.ylabel("sky")
        plt.legend()
    return xpix, truey

    ########### TO REMOVE
    ########################################################################################
    # def prepare_synthesized_beam(th, params):
    """Prepare the synthesized beam and related quantities for an instrument.

    This function computes the synthesized beam of an instrument based on the detector
    positions and primary beam properties, and optionally plots the results.

    Parameters
    ----------
    th : array_like
        Array of angular positions where the synthesized beam is evaluated.
    params : dict
        A dictionary of parameters including `detpos`, `nu`, `instrument_type`,
        `sqnh`, and `plot_sb` for plotting options.

    Returns
    -------
    sb : ndarray
        The synthesized beam values at the specified angular positions `th`.
    fwhmpeak : float
        The FWHM of the synthesized beam.
    thetapeaks : ndarray
        The angular positions of the peaks in the synthesized beam.
    amppeaks : ndarray
        The amplitudes of the peaks in the synthesized beam.
    ratio : float
        The ratio of the synthesized beam to a Gaussian peak approximation.

    """
    ndet = len(params["detpos"])
    if params["instrument_type"] not in ["Imager", "BI", "BI-GPA"]:
        print("Wrong instrument type")
        stop
    if params["plot_sb"]:
        plt.figure()
        plt.plot(
            th,
            get_primbeam(th, 3e8 / params["nu"], fwhmprimbeam_150=params["fwhmprim_150"]),
            "k--",
            label="Th. Prim. Beam at {0:3.0f} GHz".format(params["nu"] / 1e9),
        )

    for d in params["detpos"]:
        b, fwhmpeak, thetapeaks, amppeaks, ratio = give_sbcut(
            th,
            params["dist_horns"],
            3e8 / params["nu"],
            params["sqnh"],
            Df=params["Df"],
            detpos=d / 1000,
            fwhmprimbeam_150=params["fwhmprim_150"],
            kmax=params["kmax_build"],
            type=params["instrument_type"],
        )
        if params["plot_sb"]:
            p = plt.plot(
                th,
                b,
                lw=2,
                label="Th. Synth. Beam at {0:3.0f} GHz \n detpos={1:3.1f} mm - Ratio G.P.A. = {2:5.2f} \n FWHM = {3:5.2f} deg".format(
                    params["nu"] / 1e9, d, ratio, fwhmpeak
                ),
            )
            plt.plot(thetapeaks, amppeaks, "o", color=p[0].get_color())
    if params["plot_sb"]:
        plt.xlabel(r"$\theta$ [deg.]")
        plt.ylabel("Synthesized beam")
        plt.legend(loc="upper left", fontsize=8)
        plt.xlim(np.min(th), np.max(th))
        plt.ylim(-0.01, 1.4)
        plt.title("Synthesized Beam: " + params["instrument_type"])

    return b, fwhmpeak, np.array(thetapeaks), np.array(amppeaks), ratio

    ######### TO REMOVE
    ########################################################################################
    # def build_TOD(params, input_sky, ptg_deg, rec_pix_centers, algo="exact"):
    """
    Build Time-Ordered Data (TOD) from a sky model, instrument configuration, and pointing information.

    This function generates TOD by either exactly convolving a given sky model with the detector's synthesized beam
    and interpolating the result at the pointing locations, or by approximating the process using the "H operator" approach
    with additional steps for angular resolution and pixel window functions.

    Parameters
    ----------
    params : dict
        A dictionary of configuration parameters, including `instrument_type`, `detpos`, `nu`, `sqnh`, `noise_seed`,
        and other related settings for the instrument and sky processing.

    input_sky : list
        A list with two elements:
            - `input_sky[0]`: Array of x pixel positions (sky model sampling).
            - `input_sky[1]`: Array of corresponding sky values (true sky model).

    ptg_deg : ndarray
        Array of pointing angles in degrees for which the TOD needs to be calculated.

    rec_pix_centers : ndarray
        Array of pixel center positions for the reconstructed map.

    algo : str, optional
        The algorithm used for TOD generation:
            - `"exact"`: Exact method using full synthesized beam convolution and interpolation.
            - `"approx"`: Approximate method using the "H operator" approach.
            - `"approx_Hd"`: An alternative approximate method that is not yet implemented (currently raises an error).

    Returns
    -------
    TOD : ndarray
        The time-ordered data (TOD) array for all detectors. Each row corresponds to a detector, and each column corresponds
        to a different pointing location.

    convolved : ndarray
        The convolved sky after applying the detector's synthesized beam and any necessary window functions.
    """

    # Reform input sky and corresponding angles
    xpix = input_sky[0]  # Sky pixel positions
    truey = input_sky[1]  # True sky model values

    # Calculate the number of peaks in the synthesized beam based on instrument type
    if params["instrument_type"] == "Imager":
        npeaks = 1
    else:
        npeaks = 2 * params["kmax_build"] + 1

    # Initialize the TOD array with zeros (ndet x npointings)
    ndet = len(params["detpos"])
    TOD = np.zeros((ndet, params["npointings"]))

    # Define theta values for the synthesized beam (SB) in degrees
    th = np.linspace(-params["minmaxtheta"], params["minmaxtheta"], params["ntheta"])

    # Calculate the sampling interval of the input sky
    sampling = xpix[1] - xpix[0]

    # Calculate the width of the reconstructed pixels
    output_pixels_width = rec_pix_centers[1] - rec_pix_centers[0]

    # Loop over each detector to build its TOD
    for i in range(len(params["detpos"])):
        # Initialize arrays for the synthesized beam peaks (location and amplitude)
        thetapeaks = np.zeros(npeaks)
        amppeaks = np.zeros(npeaks)

        # Get the synthesized beam for the current detector
        b, fwhmpeak, thetapeaks[:], amppeaks[:], ratio = give_sbcut(
            th,
            params["dist_horns"],
            3e8 / params["nu"],
            params["sqnh"],
            Df=params["Df"],
            detpos=params["detpos"][i] / 1000,
            kmax=params["kmax_build"],
            type=params["instrument_type"],
        )

        # Normalize the peak amplitudes of the synthesized beam
        amppeaks = amppeaks / np.sum(amppeaks)

        # Choose the algorithm for TOD generation
        if algo == "exact":
            # Step 1: Convolve the true sky with the full synthesized beam
            convolved = conv_circ(truey, norm_filt(xpix, th, b))
            convolved_init = convolved.copy()  # Keep the original convolved sky

            # Step 2: If requested, reconvolve the sky with the reconstructed pixels' window function
            if params["sky_input"] == "BWconv":
                convolved = convolve_fourier_with_rectangular(convolved, output_pixels_width, sampling)

            # Step 3: Interpolate the convolved sky at the pointing locations
            convolved_pix = np.interp(rec_pix_centers, xpix, convolved)
            TOD[i, :] = np.interp(ptg_deg, xpix, convolved)
        else:
            # Step 1: Convolve the true sky with the detector's angular resolution
            bgauss = np.exp(-0.5 * (th / (fwhmpeak / np.sqrt(8 * np.log(2)))) ** 2)
            convolved = conv_circ(truey, norm_filt(xpix, th, bgauss))
            # Step 2: If needed, reconvolve the resolution-convolved sky with the reconstructed pixels' window function
            if params["sky_input"] == "BWconv":
                convolved = convolve_fourier_with_rectangular(convolved, output_pixels_width, sampling)
            # Step 3: Interpolate the convolved sky at the reconstructed pixel center locations
            convolved_pix = np.interp(rec_pix_centers, xpix, convolved)

            if algo == "approx":
                # Step 4: Sum the contributions from each peak in the synthesized beam
                for k in range(npeaks):
                    indices = np.floor(((ptg_deg - thetapeaks[k] + 180) * len(rec_pix_centers) / 360)).astype(int) % len(rec_pix_centers)
                    TOD[i, :] += convolved_pix[indices] * amppeaks[k]
            elif algo == "approx_Hd":
                H = get_H_operator(params, th, ptg_deg)
                for i in range(ndet):
                    TOD[i, :] = np.dot(H[i, :, :], convolved_pix)
            else:
                print("Wrong algorithm (algo parameter to build_TOD())")
                stop

        # Step 5: Add noise to the TOD for the current detector
        np.random.seed(params["noise_seed"])  # Initialize the random seed for reproducibility
        TOD[i, :] += np.random.randn(params["npointings"]) * params["noise_rms"]

        if params["plot_TOD"]:
            plt.figure()
            plt.plot(xpix, truey, label="True Sky")
            plt.plot(xpix, convolved, label="Convolved Sky")
            plt.errorbar(
                rec_pix_centers, convolved_pix, xerr=(rec_pix_centers[1] - rec_pix_centers[0]), label="Pixellized convolved sky", fmt="o", capsize=3
            )
            plt.plot(ptg_deg, TOD[i, :], ".", label="TOD")
            plt.xlabel(r"$\theta$ [deg.]")
            plt.ylabel("TOD")
            plt.title("TOD for Detector {0}".format(i))
            plt.legend()

    # Return the TOD array and the convolved sky
    return (
        TOD,
        convolved,
        convolved_pix,
        rec_pix_centers,
    )

    ######### TO REMOVE
    ########################################################################################
    # def get_H_operator(params, th, ptg_deg):
    nptg = params["npointings"]
    npix = params["npix"]
    ndet = len(params["detpos"])  # Number of detectors
    # Calculate the number of peaks in the synthesized beam based on instrument type
    if params["instrument_type"] == "Imager":
        npeaks = 1
    else:
        npeaks = 2 * params["kmax_build"] + 1

    allthetapeaks = np.zeros((ndet, npeaks))
    allamppeaks = np.zeros((ndet, npeaks))
    for i in range(ndet):
        _, _, allthetapeaks[i, :], allamppeaks[i, :], ratio = give_sbcut(
            th,
            params["dist_horns"],
            3e8 / params["nu"],
            params["sqnh"],
            Df=params["Df"],
            detpos=params["detpos"][i] / 1000,
            kmax=params["kmax_rec"],
            type=params["instrument_type"],
        )
        allamppeaks[i, :] = allamppeaks[i, :] / np.sum(allamppeaks[i, :])

    H = np.zeros((ndet, nptg, npix))
    for i in range(ndet):
        for j in range(npeaks):
            peaks_ptg = (ptg_deg - allthetapeaks[i, j] + 180 + 360) % 360 - 180
            peaks_indices = np.floor(((peaks_ptg - (-180)) * npix / 360)).astype(int)
            for k in range(nptg):
                H[i, k, peaks_indices[k]] = allamppeaks[i, j]
    return H

    ######### TO REMOVE
    ########################################################################################
    # def run_1d_simulation(params):
    """
    Runs a 1D sky reconstruction simulation using a given instrument configuration.

    This function simulates the observation of a 1D sky map with an instrument
    characterized by a synthesized beam. It computes the expected signal, generates
    pointing directions, constructs the Time-Ordered Data (TOD), applies the H operator,
    and reconstructs the sky map. The function also calculates residuals and provides
    visualization options.

    Parameters:
    -----------
    params : dict
        Dictionary containing simulation parameters, including:
        - `npointings`: Number of pointings (time steps)
        - `npix`: Number of pixels in the reconstructed sky map
        - `detpos`: Positions of detectors
        - `minmaxtheta`, `ntheta`: Parameters for synthesized beam computation
        - `instrument_type`: Type of instrument (e.g., "Imager")
        - `sky_input`: Convolution mode ("Bconv" or "BWconv")
        - `TOD_method`: Method used to generate the TOD ("exact" or "approx")
        - `pointing_seed`: Seed for random pointing generation
        - `plot_*`: Boolean flags for different plots

    Returns:
    --------
    tuple
        (pix_center, solution, solution_all, Ctruth_pix, xpix, truey, ss_all, beam_params)
        where:
        - `pix_center`: Centers of reconstructed sky pixels
        - `solution`: Reconstructed sky maps for each detector
        - `solution_all`: Averaged reconstructed sky map across detectors
        - `Ctruth_pix`: Theoretical convolved sky map at reconstructed pixel locations
        - `xpix`: Input sky pixel positions
        - `truey`: True sky signal
        - `ss_all`: Standard deviation of residuals
        - `beam_params`: Tuple containing (fwhmpeak, amppeaks, thetapeaks)
    """

    # Number of pointings, pixels, and detectors
    nptg = params["npointings"]  # Time steps
    npix = params["npix"]  # Pixels in the sky map
    ndet = len(params["detpos"])  # Number of detectors

    # Generate the input sky
    xpix, truey = create_sky(params)

    # Compute the synthesized beam properties
    th = np.linspace(-params["minmaxtheta"], params["minmaxtheta"], params["ntheta"])
    b, fwhmpeak, thetapeaks, amppeaks, ratio = prepare_synthesized_beam(th, params)

    # Define reconstructed sky pixel grid
    pixels = np.linspace(-180, 180, params["npix"] + 1)
    pix_center = 0.5 * (pixels[:-1] + pixels[1:])  # Compute pixel centers
    dpix = pix_center[1] - pix_center[0]  # Pixel width

    # Compute theoretical predictions for the reconstructed sky
    bgauss = np.exp(-0.5 * (th / (fwhmpeak / np.sqrt(8 * np.log(2)))) ** 2)
    Bconvy = conv_circ(truey, norm_filt(xpix, th, bgauss))
    Bconvy_pix = np.interp(pix_center, xpix, Bconvy)  # Interpolate at pixel centers

    if params["sky_input"] == "Bconv":
        Ctruth, Ctruth_pix = Bconvy, Bconvy_pix
    elif params["sky_input"] == "BWconv":
        BWconvy = convolve_fourier_with_rectangular(Bconvy, dpix, xpix[1] - xpix[0])
        BWconvy_pix = np.interp(pix_center, xpix, BWconvy)
        Ctruth, Ctruth_pix = BWconvy, BWconvy_pix
    else:
        raise ValueError("Incorrect <sky_input> in parameters")

    # Plot expected signal if required
    if params["plot_expected"]:
        plt.figure()
        plt.title("Expected signal: " + params["instrument_type"])
        plt.plot(xpix, truey, label="True Sky")
        plt.plot(xpix, Ctruth, label="Beam-convolved sky")
        plt.errorbar(pix_center, Ctruth_pix, xerr=dpix / 2, fmt=".", label="Pixellized convolved sky")
        plt.legend()

    # Generate random pointing directions
    np.random.seed(params["pointing_seed"])
    ptg_deg = np.random.random(nptg) * 360 - 180

    # Plot histogram of pointings if required
    if params["plot_pointings"]:
        plt.figure()
        plt.hist(ptg_deg, range=[-180, 180], bins=90, label="Pointings")
        plt.xlim(-180, 180)
        plt.legend()
        plt.xlabel(r"$\theta$ [deg.]")
        plt.title("Pointings")

    # Generate Time-Ordered Data (TOD)
    TOD, sky_convolved, sky_convolved_pix, _ = build_TOD(params, [xpix, truey], ptg_deg, pix_center, algo=params["TOD_method"])

    # Compute H operator
    H = get_H_operator(params, th, ptg_deg)

    # Compute inverse covariance matrix and reconstruct sky map
    HtH = np.zeros((ndet, npix, npix))
    HtHinv = np.zeros((ndet, npix, npix))
    solution = np.zeros((ndet, npix))

    for i in range(ndet):
        HtH[i] = np.dot(H[i].T, H[i])
        HtHinv[i] = np.linalg.inv(HtH[i])
        solution[i] = np.dot(HtHinv[i], np.dot(H[i].T, TOD[i]))

    # Compute average solution across detectors
    solution_all = np.mean(solution, axis=0)

    # Compute residuals
    res_all = solution_all - Ctruth_pix
    ss_all = np.std(res_all)

    # Plot reconstructed sky map and residuals if required
    if params["plot_reconstructed"]:
        plt.figure()
        plt.subplot(2, 1, 1)
        for i in range(ndet):
            plt.plot(pix_center, solution[i], "o", alpha=0.5, label=f"Reconstructed Detector #{i}")
        plt.plot(pix_center, Ctruth_pix, lw=2, color="r", alpha=0.8, label="Sky convolved")
        plt.legend()
        plt.axhline(y=0, ls="--", color="k")
        plt.xlabel("Angle")
        plt.ylabel("Sky Data")
        plt.title(f"Instrument {params['instrument_type']}")

        plt.subplot(2, 2, 3)
        plt.plot(pix_center, res_all, "k-", label=f"Residual w.r.t. Sky Convolved: {ss_all:.3g}")
        plt.axhline(y=0, ls="--", color="k")
        plt.xlabel("Angle")
        plt.ylabel("Residuals")
        plt.legend()

        plt.subplot(2, 2, 4)
        plt.hist(res_all, bins=21, range=[-5 * ss_all, 5 * ss_all], alpha=0.5, label=f"Residual: {ss_all:.3g}")
        plt.xlabel("Residuals")
        plt.ylabel("Counts")
        plt.legend()
        plt.tight_layout()

    return {
        "pix_center": pix_center,
        "solution": solution,
        "solution_all": solution_all,
        "Ctruth_pix": Ctruth_pix,
        "xpix": xpix,
        "truey": truey,
        "RMS_resall": ss_all,
        "beam_params": (fwhmpeak, amppeaks, thetapeaks),
        "ptg_deg": ptg_deg,
        "TOD": TOD,
    }
