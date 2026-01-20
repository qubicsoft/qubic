# Fortran library and f2py extension for qubic
# ==============================================

# Generate Fortran source files from Jinja2 templates
# ----------------------------------------------------

generate_fortran = find_program(meson.project_source_root() / 'tools' / 'generate_fortran.py')

# Template files that need to be processed
template_files = [
  'polarization.f90.j2',
]

# Generate .f90 files from templates
generated_fortran_sources = []
foreach template : template_files
  output_file = template.replace('.j2', '')
  gen_src = custom_target(
    output_file,
    input: template,
    output: output_file,
    command: [
      generate_fortran,
      '@INPUT@',
      '-o', meson.current_build_dir(),
      '--precision-real', '8',
    ],
    install: false,
  )
  generated_fortran_sources += gen_src
endforeach

# f2py extension module (_flib)
# ------------------------------
#
# We use Meson's native f2py integration:
# 1. custom_target generates the f2py wrappers (_flibmodule.c, _flib-f2pywrappers.f)
# 2. py.extension_module compiles everything with proper dependencies (OpenMP, etc.)

# Fortran sources for the Python extension (non-templated files)
flib_sources = [
  'xpol.f90',
  'wig3j.f',
]

# Generated sources for f2py
flib_generated = [
  'polarization.f90',
]

# Get the generated source files
flib_gen_files = []
foreach gen_file : flib_generated
  foreach gen_src : generated_fortran_sources
    if gen_src.full_path().endswith('/' + gen_file)
      flib_gen_files += gen_src
      break
    endif
  endforeach
endforeach

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
  input: flib_sources + flib_gen_files,
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

# Create the fortranobject.c file reference
fortranobject_c = files(incdir_f2py / 'fortranobject.c')

py.extension_module(
  '_flib',
  [flib_sources, flib_gen_files, _flib_wrappers, fortranobject_c],
  c_args: ['-I' + incdir_numpy, '-I' + incdir_f2py],
  fortran_args: fortran_args + ['-I' + incdir_numpy],
  dependencies: [py_dep, openmp_dep],
  install: true,
  subdir: 'qubic',  # Install in qubic/ directory
)
