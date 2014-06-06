from __future__ import absolute_import, division, print_function
from pysimulators.interfaces.healpy import SceneHealpix

__all__ = ['QubicScene']


class QubicScene(SceneHealpix):
    def __init__(self, band, nside=256, kind='IQU'):
        """
        Parameters
        ----------
        band : int
            The operating frequency, in GHz.
        nside : int
            The Healpix scene's nside.
        kind : 'I', 'QU', 'IQU'
            The sky kind: 'I' for intensity-only, 'QU' for Q and U maps,
            and 'IQU' for intensity plus QU maps.

        """
        self.nu = band * 1e9
        self.monochromatic = True
        SceneHealpix.__init__(self, nside, kind=kind)
