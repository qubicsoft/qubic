#!/usr/bin/env python3
"""
Script to generate Fortran source files from Jinja2 templates.

This script replaces the preprocessing system previously provided by
pchanial-legacy-install-hooks for the migration to numpy 2.0 and Meson.

The old .f90.src files used a custom syntax:
    ! <variable=value1,value2,value3>

This script processes Jinja2 templates (.f90.j2) that use standard
Jinja2 syntax to generate multiple variations of Fortran code.
"""

import argparse
import re
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, StrictUndefined


def parse_template_variables(template_content: str) -> tuple[dict[str, list[str]], str]:
    """
    Parse template variable definitions from the template content.

    Looks for lines like:
        {# variables: var1=val1,val2,val3; var2=val1,val2 #}

    Returns:
        A tuple of (variable_dict, cleaned_content)
        where variable_dict maps variable names to lists of values
    """
    variables = {}

    # Look for the special variables comment
    pattern = r"\{#\s*variables:\s*(.*?)\s*#\}"
    match = re.search(pattern, template_content, re.MULTILINE)

    if match:
        vars_str = match.group(1)
        # Parse each variable definition
        for var_def in vars_str.split(";"):
            var_def = var_def.strip()
            if "=" in var_def:
                name, values_str = var_def.split("=", 1)
                values = [v.strip() for v in values_str.split(",")]
                variables[name.strip()] = values

    return variables, template_content


def generate_combinations(variables: dict[str, list[str]]) -> list[dict[str, str]]:
    """
    Generate all combinations of variable values.

    All variable lists must have the same length, and combinations are
    formed by taking the i-th element from each list.

    Example:
        variables = {'ikind': ['int32', 'int64'], 'isize': ['4', '8']}
        returns: [{'ikind': 'int32', 'isize': '4'}, {'ikind': 'int64', 'isize': '8'}]
    """
    if not variables:
        return [{}]

    # Check that all lists have the same length
    lengths = [len(values) for values in variables.values()]
    if len(set(lengths)) > 1:
        raise ValueError(f"All variable lists must have the same length. Got: {lengths}")

    if not lengths:
        return [{}]

    n = lengths[0]
    combinations = []

    for i in range(n):
        combo = {name: values[i] for name, values in variables.items()}
        combinations.append(combo)

    return combinations


def generate_fortran_file(template_path: Path, output_path: Path, precision_real: int = 8) -> None:
    """
    Generate a Fortran source file from a Jinja2 template.

    Args:
        template_path: Path to the .f90.j2 template file
        output_path: Path where the generated .f90 file should be written
        precision_real: Value for PRECISION_REAL preprocessor macro (4, 8, or 16)
    """
    # Read template
    with open(template_path) as f:
        template_content = f.read()

    # Parse variable definitions
    variables, template_content = parse_template_variables(template_content)

    # Setup Jinja2 environment
    env = Environment(
        loader=FileSystemLoader(template_path.parent),
        undefined=StrictUndefined,
        trim_blocks=True,
        lstrip_blocks=True,
    )

    # Add custom filters if needed
    env.filters["fortran_type_size"] = lambda kind: {
        "int8": "1",
        "int16": "2",
        "int32": "4",
        "int64": "8",
        "real32": "4",
        "real64": "8",
        "real128": "16",
    }.get(kind, "?")

    template = env.from_string(template_content)

    # Generate combinations
    combinations = generate_combinations(variables)

    # Render template with all combinations
    # The template should use {% for combo in combinations %}...{% endfor %}
    # to generate multiple versions of functions
    output_content = template.render(
        combinations=combinations,
        PRECISION_REAL=precision_real,
        GFORTRAN=True,  # Default to gfortran
    )

    # Write output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write(output_content)

    print(f"Generated {output_path} from {template_path}")
    if combinations:
        print(f"  Generated {len(combinations)} function variations")


def main():
    parser = argparse.ArgumentParser(description="Generate Fortran source files from Jinja2 templates")
    parser.add_argument("templates", nargs="+", type=Path, help="Template files (.f90.j2) to process")
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        help="Output directory (default: same as template directory)",
    )
    parser.add_argument(
        "--precision-real",
        type=int,
        choices=[4, 8, 16],
        default=8,
        help="PRECISION_REAL value (default: 8)",
    )

    args = parser.parse_args()

    for template_path in args.templates:
        if not template_path.exists():
            print(f"Warning: Template {template_path} does not exist, skipping")
            continue

        if args.output_dir:
            output_name = template_path.name.replace(".j2", "")
            output_path = args.output_dir / output_name
        else:
            output_path = template_path.parent / template_path.name.replace(".j2", "")

        try:
            generate_fortran_file(template_path, output_path, args.precision_real)
        except Exception as e:
            print(f"Error processing {template_path}: {e}")
            raise


if __name__ == "__main__":
    main()
