#!/usr/bin/env python
import os
import re
import sys
from distutils.core import setup
from subprocess import Popen, PIPE

# force sdist to copy files
delattr(os, 'link')

VERSION = '3.0.1'

if any(c in sys.argv[1:] for c in ('install', 'sdist')):
    init = open('qubic/__init__.py.in').readlines()
    init += ['\n', '__version__ = ' + repr(VERSION) + '\n']
    open('qubic/__init__.py', 'w').writelines(init)

long_description = open('README.rst').read()
keywords = 'scientific computing'
platforms = 'MacOS X,Linux,Solaris,Unix,Windows'

setup(name='qubic',
      version=VERSION,
      description='Simulation and map-making tools for the QUBIC experiment.',
      long_description=long_description,
      url='',
      author='Pierre Chanial',
      author_email='pierre.chanial@apc.univ-paris7.fr',
      install_requires=['progressbar',
                        'pysimulators>=0.7',
                        'healpy>=0.6.1',
                        'pyYAML'],
      packages=['qubic'],
      package_data={'qubic': ['calfiles/*fits', 'data/*']},
      platforms=platforms.split(','),
      keywords=keywords.split(','),
      license='CeCILL-B',
      classifiers=[
          'Programming Language :: Python',
          'Programming Language :: Python :: 2 :: Only',
          'Development Status :: 4 - Beta',
          'Intended Audience :: Science/Research',
          'Operating System :: OS Independent',
          'Topic :: Scientific/Engineering',
          'Topic :: Scientific/Engineering :: Astronomy'])
