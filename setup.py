#!/usr/bin/env python
import numpy as np
import sys
import hooks
from numpy.distutils.core import setup
from numpy.distutils.extension import Extension

hooks.F90_COMPILE_ARGS_GFORTRAN += ['-fpack-derived','-g']
hooks.F90_COMPILE_ARGS_IFORT += ['-align norecords','-g']
if sys.platform == 'darwin':
    hooks.F77_COMPILE_OPT_GFORTRAN = ['-O2']
    hooks.F90_COMPILE_OPT_GFORTRAN = ['-O2']


ext_modules = [Extension('qubic._flib',
                         sources=['flib/polarization.f90.src',
                                  'flib/xpol.f90'],
                         include_dirs=['.', np.get_include()],
                         libraries=['gomp',
                                    ('fmod', {'sources': ['flib/wig3j.f']})])]

setup(cmdclass=hooks.cmdclass,
      ext_modules=ext_modules)
