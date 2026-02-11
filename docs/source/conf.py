# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys

sys.path.insert(0, os.path.abspath("../../qubic/"))


# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "qubicsoft"
copyright = "2024, QUBIC Collaboration"
author = "QUBIC Collaboration"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ["sphinx.ext.autodoc", "sphinx.ext.intersphinx", "sphinx.ext.napoleon", "sphinx.ext.githubpages", "sphinx_autodoc_typehints", "numpydoc", "sphinx.ext.mathjax", "sphinx.ext.autosummary"]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

autosummary_generate = True

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_rtd_theme"
html_theme_options = {
    "show_nav_level": 2,
    "collapse_navigation": False,
    "navigation_depth": 4,
    "use_edit_page_button": True,
    "navbar_end": ["search-field.html"],
}
html_sidebars = {
    "**": ["sidebar-logo.html", "sidebar-nav.html", "sidebar-ethical-ads.html"],
}
html_static_path = ["_static"]
