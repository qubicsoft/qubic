try:
    import healpy as hp
    import healpy._healpy_pixel_lib as pixlib
except ImportError:
    hp = None
    pixlib = None
import numpy as np
from pyoperators import (
    Cartesian2SphericalOperator
)
from pyoperators import CompositionOperator, IdentityOperator, Operator
from pyoperators.flags import inplace, real, square, symmetric
from pyoperators.utils import pi, strenum

from pysimulators.interfaces.healpy import (
    Healpix2CartesianOperator
)

# from ...sparse import FSRMatrix, SparseOperator

@real
class _HealPixCartesian(Operator):
    def __init__(self, nside, nest=False, dtype=float, **keywords):
        if hp is None:
            raise ImportError('The package healpy is not installed.')
        self.nside = int(nside)
        self.nest = bool(nest)
        Operator.__init__(self, dtype=dtype, **keywords)

    @staticmethod
    def _reshapehealpix(shape):
        return shape + (3,)

    @staticmethod
    def _reshapecartesian(shape):
        return shape[:-1]

    @staticmethod
    def _validatecartesian(shape):
        if len(shape) == 0 or shape[-1] != 3:
            raise ValueError('Invalid cartesian shape.')

    @staticmethod
    def _rule_identity(o1, o2):
        if o1.nside == o2.nside and o1.nest == o2.nest:
            return IdentityOperator()
        
class Cartesian2HealpixOperator_bricolage(_HealPixCartesian):
    """
    Convert cartesian coordinates into Healpix pixels.

    """

    def __init__(self, nside, nest=False, **keywords):
        """
        nside : int
            Value of the map resolution parameter.
        nest : boolean, optional
            For the nested numbering scheme, set it to True. Default is
            ring scheme.

        """
        super().__init__(
            nside,
            nest=nest,
            reshapein=self._reshapecartesian,
            reshapeout=self._reshapehealpix,
            validatein=self._validatecartesian,
            **keywords,
        )
        self.set_rule('I', lambda s: Healpix2CartesianOperator(s.nside, nest=s.nest))
        self.set_rule(
            ('.', Healpix2CartesianOperator), self._rule_identity, CompositionOperator
        )

    def get_theta_phi(self, input):
        """
        input : (ntimes, ncolmax, 3) array
            Cartesian vectors pointing on a position on the sphere.

        theta : (ntimes, ncolmax) array
            Colatitude in radians.
        phi   : (ntimes, ncolmax) array
            Longitude in radians.
        """
        shape_input = np.shape(input)
        output = hp.vec2ang(input) # radians
        theta, phi = output[0].reshape((shape_input[0], shape_input[1])), output[1].reshape((shape_input[0], shape_input[1]))
        return theta, phi

    def direct(self, input, output):
        raise ValueError("This method is not implemented. Use the method get_interpol instead.")

    def get_interpol(self, input): #get_interp_weights(nside, theta, phi=None, nest=False, lonlat=False)
        """
        input : (ntimes, ncolmax, 3) array
            Cartesian vectors pointing on a position on the sphere.

        pix : (4, ntimes, ncolmax) array
            The four HEALPix pixels closest to the direction.
        wei : (4, ntimes, ncolmax) array
            Their associated weights to compute a bilinear interpolation.
        """
        theta, phi = self.get_theta_phi(input) # radians
        func = pixlib._get_interpol_nest if self.nest else pixlib._get_interpol_ring
        res = func(self.nside, theta, phi)
        pix = np.array(res[0:4])#.astype(int)
        wei = np.array(res[4:8])
        return pix, wei



