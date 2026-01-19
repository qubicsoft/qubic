# Fortran sources for the Python extension
flib_sources = [
  'polarization.f90',
  'wig3j.f',
  'xpol.f90',
]

# f2py include directory (for fortranobject.c and fortranobject.h)
incdir_f2py = run_command(py,
  ['-c', 'import numpy.f2py; print(numpy.f2py.get_include())'],
  check: true
).stdout().strip()

# Create .f2py_f2cmap file to fix type mappings
# f2py maps real64 to float by default, we need double
# This file must be in the build directory where f2py runs
f2cmap_gen = custom_target(
  'f2py_f2cmap',
  output: '.f2py_f2cmap',
  command: [
    py, '-c',
    '''import sys; open(sys.argv[1], "w").write("""{
    'integer': {'int8': 'char', 'int16': 'short', 'int32': 'int', 'int64': 'long_long'},
    'real': {'real32': 'float', 'real64': 'double', 'p': 'double'},
    'complex': {'real32': 'complex_float', 'real64': 'complex_double'},
}
""")''',
    '@OUTPUT@',
  ],
)

# Generate f2py wrappers
# Note: f2py generates _flibmodule.c and _flib-f2pywrappers.f in the current directory
# We use a helper script to run f2py in the correct output directory
generate_f2py_wrappers = find_program(meson.project_source_root() / 'tools' / 'generate_f2py_wrappers.py')

# f2py generates different wrapper filenames depending on source format:
# - .f (fixed form): _flib-f2pywrappers.f
# - .f90 (free form): _flib-f2pywrappers2.f90
_flib_wrappers = custom_target(
  '_flib_wrappers',
  input: flib_sources,
  output: ['_flibmodule.c', '_flib-f2pywrappers2.f90'],
  command: [
    generate_f2py_wrappers,
    '--output-dir', '@OUTDIR@',
    '--module-name', '_flib',
    '@INPUT@',
  ],
  depends: [f2cmap_gen],
)

# Build the extension module using Meson's py.extension_module
# This properly handles OpenMP and links with libfmod.a
# Note: On macOS, use GCC (not Apple Clang) so that -fopenmp works
py.extension_module(
  '_flib',
  [flib_sources, _flib_wrappers, incdir_f2py / 'fortranobject.c'],
  include_directories: [incdir_numpy, incdir_f2py],
  dependencies: [py_dep, openmp_dep],
  fortran_args: fortran_args,
  install: true,
  subdir: 'qubic/_flib',
)

