import healpy as hp

from pyoperators import Operator
from pyoperators.decorators import inplace, real, symmetric

__all__ = ['HealpixConvolutionGaussianOperator']

@inplace
@real
@symmetric
class HealpixConvolutionGaussianOperator(Operator):
    def __init__(self, nside, fwhm=None, sigma=None, iter=3, lmax=None,
                 mmax=None, use_weights=False, regression=True, datapath=None,
                 **keywords):
        """
        Keywords are passed to the Healpy function smoothing.
        """
        Operator.__init__(self, shapein=12 * nside**2, **keywords)
        self.fwhm = fwhm
        self.sigma = sigma
        self.iter = iter
        self.lmax = lmax
        self.mmax = mmax
        self.use_weights = use_weights
        self.regression = regression
        self.datapath = datapath

    def direct(self, input, output):
        output[...] = hp.smoothing(input, fwhm=self.fwhm, sigma=self.sigma,
                                   iter=self.iter, lmax=self.lmax,
                                   mmax=self.mmax, use_weights=self.use_weights,
                                   regression=self.regression,
                                   datapath=self.datapath)
