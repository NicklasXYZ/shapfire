# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# http://www.sphinx-doc.org/en/master/config

# -- Path setup --------------------------------------------------------------

# add docs path to python sys.path to allow autodoc-ing a test_py_module
import os
import sys

sys.path.insert(0, os.path.abspath("../"))

# -- Project information -----------------------------------------------------

project = "ShapFire"
copyright = "2022, NicklasXYZ"  # noqa
author = "NicklasXYZ"

# The full version, including alpha/beta/rc tags
release = "0.1.0"

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",
    # "sphinx.ext.doctest",
    # "sphinx.ext.extlinks",
    "sphinx.ext.intersphinx",
    # "sphinx.ext.todo",
    # "sphinx.ext.mathjax",
    # "sphinx.ext.autosummary",
    "sphinx.ext.viewcode",
    # "sphinxcontrib.details.directive",
    "sphinx.ext.napoleon",  # Support for google docstrings
    # "sphinx_autodoc_typehints",
    "nbsphinx",
    "IPython.sphinxext.ipython_console_highlighting",
    "IPython.sphinxext.ipython_directive",
]

autodoc_preserve_defaults = True

# sphinx_autodoc_typehints
# autodoc_typehints = 'description'

# autodoc_typehints = 'none'
# typehints_fully_qualified = False

# autodoc_typehints_format = "fully-qualified"
# autodoc_typehints_format = "short"
# autodoc_typehints = "signature"

# typehints_fully_qualified = True

# autodoc_typehints = "both"

# Napoleon settings
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = False
napoleon_use_admonition_for_notes = False
napoleon_use_admonition_for_references = False
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_type_aliases = None
napoleon_attr_annotations = True

# autodoc_type_aliases = False

# napoleon_google_docstring = True
# napoleon_numpy_docstring = True
# napoleon_include_init_with_doc = False
# napoleon_include_private_with_doc = False
# napoleon_include_special_with_doc = False
# napoleon_use_admonition_for_examples = False
# napoleon_use_admonition_for_notes = False
# napoleon_use_admonition_for_references = False
# napoleon_use_ivar = True
# napoleon_use_param = True
# napoleon_use_rtype = False
# napoleon_type_aliases = False
# napoleon_attr_annotations = True


intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "pandas": ("https://pandas.pydata.org/pandas-docs/dev", None),
    "matplotlib": ("https://matplotlib.org/stable/", None),
    "scikit-learn": ("https://scikit-learn.org/stable/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "lightgbm": ("https://lightgbm.readthedocs.io/en/latest/", None),
}

# Python 3 [latest]: ('https://docs.python.org/3/', None)
# Python 3 [3.x]: ('https://docs.python.org/3.9/', None)
# attrs [stable]: ('https://www.attrs.org/en/stable/', None)
# Flask [1.1.x]: ('https://flask.palletsprojects.com/en/1.1.x/', None)
# h5py [latest]: ('https://docs.h5py.org/en/latest/', None)
# matplotlib [stable]: ('https://matplotlib.org/stable/', None)
# numpy [stable]: ('https://numpy.org/doc/stable/', None)
# pandas [latest?]: ('https://pandas.pydata.org/docs/', None)
# scikit-learn [stable]: ('https://scikit-learn.org/stable/', None)


# The reST default role (used for this markup: `text`) to use for all
# documents.
# default_role = "any"

# autosummary_generate = True
# autoclass_content = "class"
autoclass_content = "init"

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named 'default.css' will overwrite the builtin 'default.css'.
html_static_path = ["_static"]
# html_css_files = [
#     "extra_css.css",
#     "https://cdnjs.cloudflare.com/ajax/"
#     + "libs/font-awesome/5.15.4/css/all.min.css",
# ]

# Custom css file to adjust font size
html_css_files = [
    "extra_cssv2.css",
]

# -- HTML theme settings ------------------------------------------------

extensions.append("sphinx_immaterial")
html_title = "ShapFire"
html_theme = "sphinx_immaterial"
html_favicon = "_static/images/favicon.ico"
html_logo = "_static/images/fire.svg"  # from https://gifer.com/en/Ybin

# The master toctree document.
# master_doc = "index"

# material theme options (see theme.conf for more information)
html_theme_options = {
    "icon": {
        "repo": "fontawesome/brands/github",
    },
    "site_url": "https://nicklasxyz.github.io/shapfire/",
    "repo_url": "https://github.com/nicklasxyz/shapfire/",
    "repo_name": "ShapFire",
    "repo_type": "github",
    "edit_uri": "blob/main/docs",
    # "google_analytics": ["UA-XXXXX", "auto"],
    "globaltoc_collapse": True,
    "globaltoc_includehidden": True,
    "features": [
        # "navigation.expand",
        # "navigation.tabs",
        "toc.integrate",
        "navigation.sections",
        # "navigation.instant",
        # "header.autohide",
        # "navigation.top",
        # "navigation.tracking",
        # "search.highlight",
        "search.share",
    ],
    "palette": [
        {
            # "media": "(prefers-color-scheme: light)",
            "media": "(prefers-color-scheme: light)",
            # "scheme": "slate",
            "scheme": "default",
            "primary": "cyan",
            "accent": "cyan",
            # "toggle": {
            #     "icon": "material/lightbulb-outline",
            #     "name": "Switch to dark mode",
            # },
        },
        # {
        #     "media": "(prefers-color-scheme: dark)",
        #     "scheme": "default",
        #     "primary": "#15252f",
        #     # "accent": "teal",
        #     "toggle": {
        #         "icon": "material/lightbulb",
        #         "name": "Switch to light mode",
        #     },
        # },
    ],
    "version_dropdown": True,
    # "version_info": [
    #     {
    #         "version": "https://shapfire.rtfd.io",
    #         "title": "ReadTheDocs",
    #         "aliases": []
    #     },
    #     {
    #         "version": "https://nicklasxyz.github.io/shapfire",
    #         "title": "Github Pages",
    #         "aliases": []
    #     },
    # ],
    "toc_title_is_page_title": True,
}  # end html_theme_options

html_last_updated_fmt = ""
html_use_index = False
html_domain_indices = False

# ---- Other documentation options -------------------------

# todo_include_todos = True


def setup(app):
    app.add_object_type(
        "confval",
        "confval",
        objname="configuration value",
        indextemplate="pair: %s; configuration value",
    )


# Fav icons:
# <link rel="apple-touch-icon" sizes="180x180" href="/apple-touch-icon.png">
# <link rel="icon" type="image/png" sizes="32x32" href="/favicon-32x32.png">
# <link rel="icon" type="image/png" sizes="16x16" href="/favicon-16x16.png">
# <link rel="manifest" href="/site.webmanifest">
# <link rel="mask-icon" href="/safari-pinned-tab.svg" color="#5bbad5">
# <meta name="msapplication-TileColor" content="#25c5da">
# <meta name="theme-color" content="#ffffff">


# NBSphinx options
nbsphinx_execute = "never"


extlinks = {
    "pr": ("https://github.com/statsmodels/statsmodels/pull/%s", "PR #"),
    "issue": (
        "https://github.com/statsmodels/statsmodels/issues/%s",
        "Issue #",
    ),
}

autosectionlabel_prefix_document = True
