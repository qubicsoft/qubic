# Find all Python files and data, dicts & calfiles files in qubic package
# We use run_command to get all .py, .dict, .fits, and .pkl files
python_files = run_command(
  'find', 'calfiles', 'data', 'dicts', 'lib',
  '(',
  '-name', '*.py',
  '-o', '-name', '*.dict',
  '-o', '-name', '*.fits',
  '-o', '-name', '*.pkl',
  '-o', '-name', '*.txt',
  ')',
  '-not', '-path', '*/.*',
  '-not', '-path', '*/__pycache__/*',
  check: true,
  capture: true,
).stdout().strip().split('\n')

# Install main __init__.py
py.install_sources(['__init__.py'], subdir: 'qubic')

# Install all found Python files with preserve_path
if python_files != ['']
  py.install_sources(
    python_files,
    subdir: 'qubic',
    preserve_path: true,
  )
endif
