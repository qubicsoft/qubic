# Find all Python files recursively in qubic package
# We use run_command to get all .py files
python_files = run_command(
  'find', 'calfiles', 'data', 'dicts', 'lib',
  '-name', '*.py',
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
