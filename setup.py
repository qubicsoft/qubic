#!/usr/bin/env python
import numpy as np
import os
import sys
import hooks
from numpy.distutils.core import setup
from numpy.distutils.extension import Extension

VERSION = '4.5'

name = 'qubic'
long_description = open('README.md').read()
keywords = 'scientific computing'
platforms = 'MacOS X,Linux,Solaris,Unix,Windows'
delattr(os, 'link')  # force sdist to copy files
hooks.F90_COMPILE_ARGS_GFORTRAN += ['-fpack-derived','-g']
hooks.F90_COMPILE_ARGS_IFORT += ['-align norecords','-g']
if sys.platform == 'darwin':
    hooks.F77_COMPILE_OPT_GFORTRAN = ['-O2']
    hooks.F90_COMPILE_OPT_GFORTRAN = ['-O2']


ext_modules = [Extension('qubic._flib',
                         sources=['src/polarization.f90.src',
                                  'src/xpol.f90'],
                         include_dirs=['.', np.get_include()],
                         libraries=['gomp',
                                    ('fmod', {'sources': ['src/wig3j.f']})])]

setup(name=name,
      version=hooks.get_version(name, VERSION),
      description='Simulation and map-making tools for the QUBIC experiment.',
      long_description=long_description,
      url='',
      author='Pierre Chanial',
      author_email='pierre.chanial@apc.univ-paris7.fr',
      install_requires=[
        'astropy',
        'corner',
        'emcee',
        'GetDist',
        'healpy>=0.6.1',
        'iminuit',
        'numpy',
        'pandas',
        'progressbar',
        'pyfftw',
        'pyoperators>=0.13.18',
        'pysimulators>=1.0.8',
        'pysm3',
        'qubicpack @ git+https://github.com/satorchi/qubicpack@master',
        'scipy',
        'satorchipy @ git+https://github.com/satorchi/mypy@master',
      ],
      packages=['qubic', 'qubic/calfiles', 'qubic/data', 'qubic/io', 'qubic/dicts', 'qubic/data/FastSimulator_version01','qubic/TES'],
      package_data={'qubic': ['calfiles/*', 'data/*', 'scripts/*py', 'dicts/*.dict', 'data/FastSimulator_version01/*.pkl','TES/*']},
      platforms=platforms.split(','),
      keywords=keywords.split(','),
      cmdclass=hooks.get_cmdclass(),
      ext_modules=ext_modules,
      license='CeCILL-B',
      classifiers=[
          'Programming Language :: Python',
          'Programming Language :: Python :: 2 :: Only',
          'Development Status :: 4 - Beta',
          'Intended Audience :: Science/Research',
          'Operating System :: OS Independent',
          'Topic :: Scientific/Engineering',
          'Topic :: Scientific/Engineering :: Astronomy'])
