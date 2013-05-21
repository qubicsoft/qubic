# coding: utf-8
from __future__ import division

import numpy as np

from numpy.random import random_sample as randomu

__all__ = ['create_random_pointings',
          ]


def create_random_pointings(npointings, dtheta):
    """
    Return the Euler angles (φ,θ,ψ) of the ZY'Z'' intrinsic rotation
    as (θ,φ,ψ) triplets.

    """
    dtheta=np.radians(dtheta)
    theta = np.degrees(np.arccos(np.cos(dtheta) + (1 - np.cos(dtheta)) *
                       randomu(npointings)))
    phi = randomu(npointings) * 360
    pitch = randomu(npointings) * 360
    pointings = np.array([theta, phi, pitch]).T
    return pointings
