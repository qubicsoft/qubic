#!/usr/bin/env python
import numpy as np
import os
import sys
from numpy.distutils.core import setup
from numpy.distutils.extension import Extension
from hooks import get_cmdclass, get_version

# force sdist to copy files
delattr(os, 'link')

VERSION = '4.3'

name = 'qubic'
long_description = open('README.rst').read()
keywords = 'scientific computing'
platforms = 'MacOS X,Linux,Solaris,Unix,Windows'
extra_f90_compile_args = ['-g -cpp -fopenmp -fpack-derived']
if '--debug' in sys.argv:
    extra_f90_compile_args += ['-fbounds-check']

if any(c in sys.argv for c in ('build', 'build_ext', 'install')):
    # write f2py's type mapping file
    root = os.path.dirname(__file__)
    with open(os.path.join(root, '.f2py_f2cmap'), 'w') as f:
        f.write("{'real': {'sp': 'float', 'dp': 'double', 'p': 'double'}, 'com"
                "plex': {'sp': 'complex', 'dp': 'complex_double', 'p': 'comple"
                "x_double'}}\n")

ext_modules = [Extension('qubic._flib',
                         sources=['src/polarization.f90.src',
                                  'src/xpol.f90'],
                         extra_f90_compile_args=extra_f90_compile_args,
                         f2py_options=['--quiet'],
                         include_dirs=['.', np.get_include()],
                         libraries=['gomp',
                                    ('fmod', {'sources': ['src/wig3j.f']})])]

setup(name=name,
      version=get_version(name, VERSION),
      description='Simulation and map-making tools for the QUBIC experiment.',
      long_description=long_description,
      url='',
      author='Pierre Chanial',
      author_email='pierre.chanial@apc.univ-paris7.fr',
      install_requires=['progressbar',
                        'pyoperators>=0.12.12',
                        'pysimulators>=1.0.8',
                        'healpy>=0.6.1',
                        'pyYAML'],
      packages=['qubic', 'qubic/io'],
      package_data={'qubic': ['calfiles/*fits', 'data/*', 'scripts/*py']},
      platforms=platforms.split(','),
      keywords=keywords.split(','),
      cmdclass=get_cmdclass(),
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
