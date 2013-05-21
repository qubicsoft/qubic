from __future__ import division

import numpy as np
from numpy import cos, sin
from progressbar import ProgressBar, Bar, ETA, Percentage

def progress_bar(n, info=''):
    """
    Return a default progress bar.

    """
    return ProgressBar(widgets=[info, Percentage(), Bar('=', '[', ']'),
                                ETA()], maxval=n).start()

def _rotateuv(uvin, th, ph, xi, inverse=False):
    """
    Rotate unit vector.

    Parameters
    ----------
    uvin : (n,3) array
        The unit vector to be rotated.

    """
    matrix = np.mat([[cos(xi)*cos(th)*cos(ph)-sin(xi)*sin(ph),
                      cos(xi)*cos(th)*sin(ph)+sin(xi)*cos(ph),
                      -cos(xi)*sin(th)],
                     [-sin(xi)*cos(th)*cos(ph)-cos(xi)*sin(ph),
                      -sin(xi)*cos(th)*sin(ph)+cos(xi)*cos(ph),
                      sin(xi)*sin(th)],
                     [sin(th)*cos(ph),
                      sin(th)*sin(ph),
                      cos(th)]])

    if inverse:
        matrix = matrix.I

    return np.einsum('ij,nj->ni', matrix, uvin)
