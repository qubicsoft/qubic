#!/usr/bin/env python
import os
import re
import sys
from distutils.core import setup
from subprocess import Popen, PIPE

# force sdist to copy files
delattr(os, 'link')

VERSION = '2.5.0'


def version_sdist():
    stdout, stderr = Popen(['git', 'rev-parse', '--abbrev-ref', 'HEAD'],
                           stdout=PIPE, stderr=PIPE).communicate()
    if stderr:
        return VERSION
    branch = stdout[:-1]
    if re.search('^v[0-9]', branch) is not None:
        branch = branch[1:]
    if branch != 'master':
        return VERSION
    stdout, stderr = Popen(['git', 'rev-parse', '--verify', '--short', 'HEAD'],
                           stdout=PIPE, stderr=PIPE).communicate()
    if stderr:
        return VERSION
    return VERSION + '-' + stdout[:-1]

version = version_sdist()
if 'install' in sys.argv[1:]:
    if '-' in version:
        version = VERSION + '-dev'

if any(c in sys.argv[1:] for c in ('install', 'sdist')):
    init = open('qubic/__init__.py.in').readlines()
    init += ['\n', '__version__ = ' + repr(version) + '\n']
    open('qubic/__init__.py', 'w').writelines(init)

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
                        'pysimulators>=0.6.3',
                        'healpy>=0.6.1',
                        'pyYAML'],
      packages=['qubic'],
      package_data={'qubic': ['calfiles/*fits']},
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
