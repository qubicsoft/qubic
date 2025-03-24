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
        # convention = "zenith,azimuth"
        # output = Cartesian2SphericalOperator(convention)(input)
        # theta, phi = output[:, :, 0], output[:, :, 1]
        # print(np.shape(input))
        # sys.exit()
        # input = np.moveaxis(input, [0], [2])
        shape_input = np.shape(input)
        output = hp.vec2ang(input) # radians
        theta, phi = output[0].reshape((shape_input[0], shape_input[1])), output[1].reshape((shape_input[0], shape_input[1]))
        # print(np.shape(output))
        
        # print("\n\n\n\ntheta", theta)
        return theta, phi

    def direct(self, input, output): #get_interp_weights(nside, theta, phi=None, nest=False, lonlat=False)
        theta, phi = self.get_theta_phi(input)
        # print("theta, phi")
        func = pixlib._get_interpol_nest if self.nest else pixlib._get_interpol_ring
        # print(np.shape(func(self.nside, theta, phi)))
        # print(func(self.nside, theta, phi)[4][0, 0])
        func(self.nside, theta, phi, output)

    def get_interpol(self, input): #get_interp_weights(nside, theta, phi=None, nest=False, lonlat=False)
        theta, phi = self.get_theta_phi(input) # radians
        # print(np.shape(theta), np.shape(phi))
        # print(theta[0, 0], phi[0, 0])
        # sys.exit()
        # print("theta, phi")
        func = pixlib._get_interpol_nest if self.nest else pixlib._get_interpol_ring
        # print(np.shape(func(self.nside, theta, phi)))
        # print(func(self.nside, theta, phi)[4][0, 0])
        res = func(self.nside, theta, phi)
        # print(np.shape(res))
        pix = np.array(res[0:4])#.astype(int)
        wei = np.array(res[4:8])

        func_2 = pixlib._ang2pix_nest if self.nest else pixlib._ang2pix_ring
        ipix = func_2(self.nside, theta, phi)

        # print(ipix[0, 0])

        # # print(wei[:, 0, 0])
        # argmax_wei = np.argmax(wei, axis=0)
        # for i in range(len(argmax_wei)):
        #     for j in range(len(argmax_wei[0])):
        #         wei[:, i, j] = 0
        #         wei[argmax_wei[i, j], i, j] = 1
        # # print(wei[:, 0, 0])
        # # sys.exit()

        # shape pix : (4, ntimes, npeaks)
        # shape wei : (4, ntimes, npeaks)
        # print(pix[:, 0, 0], wei[:, 0, 0])
        # sys.exit()
        # print(wei)
        # sys.exit()
        return pix, wei, ipix, theta, phi



