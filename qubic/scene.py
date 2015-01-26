from __future__ import absolute_import, division, print_function
import numpy as np
from pyoperators import asoperator, IdentityOperator, ReciprocalOperator
from pysimulators.interfaces.healpy import SceneHealpix
from scipy.constants import c, h, k

__all__ = ['QubicScene']


class QubicScene(SceneHealpix):
    def __init__(self, band, nside=256, kind='IQU', absolute=False,
                 temperature=True):
        """
        Parameters
        ----------
        band : int
            The operating frequency, in GHz.
        nside : int, optional
            The Healpix scene's nside.
        kind : 'I', 'QU' or 'IQU', optional
            The sky kind: 'I' for intensity-only, 'QU' for Q and U maps,
            and 'IQU' for intensity plus QU maps.
        absolute : boolean, optional
            If true, the scene pixel values include the CMB background and the
            fluctuations in units of Kelvin, otherwise it only represents the
            fluctuations, in microKelvin.
        temperature : boolean, optional
            If true, the scene represents a temperature (absolute in Kelvin or
            differential in microKelvin according the the absolute keyword) and
            a power in W/m^2/Hz otherwise.

        """
        self.nu = band * 1e9
        self.monochromatic = True
        self.absolute = bool(absolute)
        self.temperature = bool(temperature)
        SceneHealpix.__init__(self, nside, kind=kind)

    def get_unit_conversion_operator(self):
        """
        Convert sky temperature into W / m^2 / Hz / pixel.

        If the scene has been initialised with the 'absolute' keyword, the
        scene is assumed to include the CMB background and the fluctuations
        (in Kelvin) and the operator follows the non-linear Planck law.
        Otherwise, the scene only includes the fluctuations (in microKelvin)
        and the operator is linear (i.e. the output also corresponds to power
        fluctuations).

        """
        if not self.temperature:
            return IdentityOperator()

        # solid angle of a sky pixel
        omega = 4 * np.pi / self.shape[0]
        a = 2 * omega * h * self.nu**3 / c**2
        if self.absolute:
            hnu_k = h * self.nu / k
            return a / asoperator(np.expm1)(hnu_k * ReciprocalOperator())
        T = 2.7255  # Fixsen et al. 2009
        hnu_kT = h * self.nu / (k * T)
        val = 1e-6 * a * hnu_kT * np.exp(hnu_kT) / (np.expm1(hnu_kT)**2 * T)
        return asoperator(val)
