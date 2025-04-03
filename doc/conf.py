# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import datetime
import os
import sys

sys.path.insert(0, os.path.abspath(".."))
sys.path.insert(0, os.path.abspath("../lenapy"))

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

curr_year = datetime.datetime.now().year

project = "lenapy"
copyright = f"2023 - {curr_year}, Sebastien Fourest"
author = "Sebastien Fourest"
# The short X.Y version.
version = "0.7"
# The full version, including alpha/beta/rc tags.
release = "First beta version on github"

language = "en"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.viewcode",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx.ext.napoleon",
    "sphinx_mdinclude",
    "nbsphinx",
]

# autosummaries from source-files : use .. autosummary::
autosummary_generate = True
# Inherit docstrings from parent classes
autodoc_inherit_docstrings = True
# autodoc don't show __init__ docstring
autoclass_content = "both"
# sort class members by their order in the source
autodoc_member_order = "bysource"

# show all members of a class in the Methods and Attributes sections automatically
numpydoc_show_class_members = True
# create a Sphinx table of contents for the lists of class methods and attributes
numpydoc_class_members_toctree = True
# show all inherited members of a class in the Methods and Attributes sections
numpydoc_show_inherited_class_members = True

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# The suffix(es) of source filenames.
source_suffix = [".rst", ".md"]

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_rtd_theme"
html_theme_options = {
    "logo_only": False,
    "display_version": True,
    "prev_next_buttons_location": "both",
    "collapse_navigation": False,
    "sticky_navigation": True,
    "navigation_depth": 4,
    "includehidden": True,
    "titles_only": False,
    "style_nav_header_background": "#2980B9",  # header color en haut à gauche
    # github_url for open access version
}
html_static_path = ["_static"]

# Example configuration for intersphinx: refer to the Python standard library.
intersphinx_mapping = {
    "python": ("https://docs.python.org/", None),
    "numpy": ("http://docs.scipy.org/doc/numpy/", None),
    "scipy": ("http://docs.scipy.org/doc/scipy/reference", None),
    "matplotlib": ("http://matplotlib.org", None),
    "xarray": ("https://docs.xarray.dev/en/stable/", None),
}

# We recommend adding the following config value.
# Sphinx defaults to automatically resolve *unresolved* labels using all your Intersphinx mappings.
# This behavior has unintended side-effects, namely that documentations local references can
# suddenly resolve to an external location.
# See also:
# https://www.sphinx-doc.org/en/master/usage/extensions/intersphinx.html#confval-intersphinx_disabled_reftypes
intersphinx_disabled_reftypes = ["*"]
