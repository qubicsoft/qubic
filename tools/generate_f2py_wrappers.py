#!/usr/bin/env python3
"""
Generate f2py wrappers for the _flib extension.

This script is called by Meson to generate the f2py wrapper files
(_flibmodule.c and _flib-f2pywrappers.f) in the correct output directory.
"""

import argparse
import os
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description="Generate f2py wrappers")
    parser.add_argument("--output-dir", required=True, help="Output directory for generated files")
    parser.add_argument("--module-name", default="_flib", help="Module name")
    parser.add_argument("sources", nargs="+", help="Fortran source files")
    args = parser.parse_args()

    # Convert all paths to absolute paths before changing directory
    original_dir = os.getcwd()
    output_dir = os.path.abspath(args.output_dir)
    sources = [os.path.abspath(s) for s in args.sources]

    os.makedirs(output_dir, exist_ok=True)

    # Change to output directory so f2py generates files there
    os.chdir(output_dir)

    try:
        cmd = [
            sys.executable,
            "-m",
            "numpy.f2py",
            *sources,
            "-m",
            args.module_name,
            "--lower",
        ]
        print(f"Running f2py in {output_dir}:")
        print(" ".join(cmd))

        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.stdout:
            print(result.stdout)
        if result.stderr:
            print(result.stderr, file=sys.stderr)

        if result.returncode != 0:
            sys.exit(result.returncode)

        # Verify outputs exist
        # f2py generates different wrapper filenames depending on source format:
        # - .f (fixed form): _flib-f2pywrappers.f
        # - .f90 (free form): _flib-f2pywrappers2.f90
        module_c = f"{args.module_name}module.c"
        wrapper_f = f"{args.module_name}-f2pywrappers.f"
        wrapper_f90 = f"{args.module_name}-f2pywrappers2.f90"

        if not os.path.exists(module_c):
            print(f"Error: Expected output file '{module_c}' not found in {output_dir}")
            print(f"Directory contents: {os.listdir('.')}")
            sys.exit(1)

        # Check for either wrapper format
        if not os.path.exists(wrapper_f) and not os.path.exists(wrapper_f90):
            print(f"Error: Expected wrapper file not found in {output_dir}")
            print(f"  Looked for: {wrapper_f} or {wrapper_f90}")
            print(f"Directory contents: {os.listdir('.')}")
            sys.exit(1)

        print(f"Successfully generated f2py wrappers in {output_dir}")

    finally:
        os.chdir(original_dir)


if __name__ == "__main__":
    main()
