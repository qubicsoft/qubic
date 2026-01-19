project('qubic',
  'c', 'fortran',
  version: run_command('python', '-c',
    'from setuptools_scm import get_version; print(get_version())',
    check: false).stdout().strip(),
  meson_version: '>= 1.1.0',
  default_options: [
    'warning_level=1',
    'buildtype=release',
  ],
)

python_prog = find_program('python3', required: true)
py = import('python').find_installation(pure: false)

# Check Python version
if py.language_version().version_compare('< 3.10')
  error('Python 3.10 or newer is required')
endif

py_dep = py.dependency()

# NumPy include directory
incdir_numpy = run_command(python_prog,
  ['-c', 'import numpy; print(numpy.get_include())'],
  check: true
).stdout().strip()

# Use declare_dependency to avoid absolute path issues
numpy_inc = declare_dependency(
  compile_args: ['-I' + incdir_numpy]
)

# NumPy f2py executable
incdir_f2py = run_command(python_prog,
  ['-c', 'import numpy.f2py; print(numpy.f2py.get_include())'],
  check: true
).stdout().strip()

# Compilers
cc = meson.get_compiler('c')
fc = meson.get_compiler('fortran')

# Fortran compiler flags
fortran_args = []

if fc.get_id() == 'gcc'
  # gfortran
  fortran_args += ['-cpp', '-fpack-derived']
  add_project_arguments('-DGFORTRAN', language: 'fortran')
elif fc.get_id() == 'intel' or fc.get_id() == 'intel-cl'
  # ifort
  fortran_args += ['-fpp', '-ftz','-fp-model precise','-ftrapuv','-warn all', '-align', 'norecords']
endif

# OpenMP support
openmp_dep = dependency('openmp', language: 'fortran', required: true)

# Install qubic package
subdir('qubic')
