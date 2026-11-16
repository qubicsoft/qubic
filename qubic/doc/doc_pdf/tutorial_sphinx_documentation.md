The tutorial to build documentation using Sphinx can be found following: https://github.com/sfarrens/ecole-euclid-2023?tab=readme-ov-file

The commands used are the following.

# Build Documentation
In the main branch, generate your documentation using Sphinx : 

```bash
sphinx-quickstart docs/
sphinx-apidoc -Mfeo docs/source module_path
```

Then, you have to modify the two files created by *sphinx-apidoc* : *index.rst* and *conf.py*.
The file *index.rst* is used to build the "home page" of you documentation. It can be highly customizable but I didn't dig into the details. A low cost modification is to add :
```bash
Software name
=======================

.. toctree::
   :caption: Summary
   :maxdepth: 1
   :glob:

   modules

Index & tables :
===================

* :ref:`genindex`     — General index
* :ref:`modindex`     — General module index
* :ref:`search`       — Full-text search
```
The *conf.py* file is used to customize your documentation, by adding sphinx extensions, themes, parameters for HTMLM file, etc... Also, do not forget to add the path of your repository at the beginning of the file, so that sphinx can read and import the correct Python module. For example : 
```bash
import os
import sys

sys.path.insert(0, os.path.abspath("../module_path"))
```
For your general configuration, you have plenty of options, I am using this one : 
```bash
extensions = ["sphinx.ext.autodoc", "sphinx.ext.intersphinx", "sphinx.ext.napoleon", "sphinx.ext.githubpages", "sphinx_autodoc_typehints", "numpydoc", "sphinx.ext.mathjax", "sphinx.ext.autosummary"]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

autosummary_generate = True
```
And, I am using this HTML options : 
```bash
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
```

If you want to verify your documentation, you can build and see it with :

```bash
sphinx-build docs/source docs/build
firefox docs/build/index.html
```

# Configure gh-pages

The next step is to build and configure a gh-pages. 

## Build gh-pages

You can build this specific branch following :

```bash
git checkout --orphan gh-pages
git reset --hard
echo "gh-pages branch for GitHub Pages" > README.md
git add README.md
git commit -m "Initialize gh-pages branch"
git push origin gh-pages
```

## Activate GitHub pages

To activate this functionality in your GitHub repository, you have to go into settings and click on Pages. In the "Build and Deployment" section, select "Deploy from a branch" in Source, and select the gh-pages in the Branch section.
Validate by clicking on save.

# Add GitHub Actions Workflow

Come back into your main branch and create the yaml file to configure the Continuous Deployment in the following path :

```bash
.github/workflows/cd.yml
```

I put an example of this file on my repository, you can use the same lines but do not forget to modify the different paths written.

# Deployment

After committing and pushing any change in the main branch, you will see in the Actions section the execution of the different commands we wrote. Please feel free to check if everything works as expected.
If everything runs well, you can find your documentation following the https address : 

```bash
<username>.github.io/<repository_name>/index.html
```








