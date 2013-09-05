#!/usr/bin/env python
import os
import re
from distutils.core import setup

def version():
    f = open(os.path.join('qubic', 'version.py')).read()
    m = re.search(r"VERSION = '(.*)'", f)
    return m.groups()[0]

# force sdist to copy files
delattr(os, 'link')

version = version()
long_description = open('README.rst').read()
keywords = 'scientific computing'
platforms = 'MacOS X,Linux,Solaris,Unix,Windows'

setup(name='qubic',
      version=version,
      description='Simulation and map-making tools for the QUBIC experiment.',
      long_description=long_description,
      url='',
      author='Pierre Chanial',
      author_email='pierre.chanial@apc.univ-paris7.fr',
      install_requires=['progressbar',
                        'pyoperators>=0.7.1',
                        'pysimulators>=0.4.1',
                        'healpy>=0.6.1',
                        'pyYAML'],
      packages=['qubic'],
      package_data={'qubic':['calfiles/*fits']},
      platforms=platforms.split(','),
      keywords=keywords.split(','),
      license='CeCILL-B',
      classifiers = [
          'Programming Language :: Python',
          'Programming Language :: Python :: 2 :: Only',
          'Development Status :: 4 - Beta',
          'Intended Audience :: Science/Research',
          'Operating System :: OS Independent',
          'Topic :: Scientific/Engineering',
          'Topic :: Scientific/Engineering :: Astronomy',
          ])

