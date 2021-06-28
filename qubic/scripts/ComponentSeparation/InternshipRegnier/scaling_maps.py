import numpy as np
import fgbuster
import os
from qubicpack.utilities import Qubic_DataDir
import qubic
from fgbuster import (separation_recipes, xForecast, CMB, Dust, Synchrotron, FreeFree, PowerLaw,  # sky-fitting model
                      basic_comp_sep)

npix = 12*256**2

def SED_eval(comp, nu0 = None, temp = None, beta_d = None, nu = None) :

    if comp == 'Dust' :
        SED = fgbuster.component_model.Dust(nu0 = nu0, temp = temp, beta_d = beta_d).eval(nu = nu)
    elif comp == 'CMB' :
        SED = fgbuster.component_model.CMB().eval(nu = nu)
    else :
        raise TypeError('Give the good component please !')

    return SED

def scaling(maps, okpix, nu) :

    maps_scale = np.zeros(((nu.shape[0], 3, npix)))

    sed = SED_eval('Dust', nu0 = 150., temp = 20., beta_d = 1.54, nu = nu)
    print(sed[0])
    for i in range(nu.shape[0]) :
        maps_scale[i, : okpix] = sed[i] * maps


    return maps_scale
