# Configuration file for the Sphinx documentation builder.
#
# For a full list of options:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

from __future__ import annotations

import os
import sys
from pathlib import Path

try:
    from importlib.metadata import PackageNotFoundError, version as pkg_version
except ImportError:  # pragma: no cover (py<3.8)
    from importlib_metadata import PackageNotFoundError, version as pkg_version  # type: ignore


# -- Path setup --------------------------------------------------------------

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
sys.path.insert(0, str(PROJECT_ROOT))


# -- Project information -----------------------------------------------------

project = "matchms"
author = "matchms development team"
copyright = "2023, Düsseldorf University of Applied Sciences & Netherlands eScience Center"

try:
    release = pkg_version("matchms")
except PackageNotFoundError:
    # Docs can still build even if the package isn't installed (e.g. local preview),
    # but you may want to fail hard instead, depending on your policy.
    release = "0+unknown"

# Optional: keep major.minor as "version"
version = ".".join(release.split(".")[:2]) if release and release[0].isdigit() else release


# -- General configuration ---------------------------------------------------

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.coverage",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
    "sphinx.ext.todo",
    "sphinx.ext.doctest",
    "sphinx.ext.napoleon",
    "sphinxcontrib.apidoc",
]

templates_path = ["_templates"]
language = "en"

exclude_patterns = [
    "_build",
    "Thumbs.db",
    ".DS_Store",
    "readthedocs/conf.rst",
]

# Hide undocumented member by excluding default undoc-members option
os.environ["SPHINX_APIDOC_OPTIONS"] = "members,show-inheritance"


# -- sphinxcontrib-apidoc ----------------------------------------------------

apidoc_module_dir = str(PROJECT_ROOT / "matchms")
apidoc_output_dir = "api"
apidoc_excluded_paths = [
    "tests",
    "readthedocs",
]
apidoc_separate_modules = True
apidoc_module_first = True


# -- autodoc -----------------------------------------------------------------

autodoc_default_options = {
    "special-members": "__init__,__call__",
    "inherited-members": True,
}

# Only mock truly problematic/heavy optional deps at import time.
# NOTE: we no longer import matchms in conf.py, so this is mostly for autodoc.
autodoc_mock_imports = [
    "rdkit",
]

napoleon_google_docstring = False


# -- HTML output -------------------------------------------------------------

html_theme = "alabaster"

html_theme_options = {
    "logo": "matchms.png",
    "github_user": "matchms",
    "github_repo": "matchms",
    "page_width": "1080px",
}

html_static_path = ["_static"]


# -- todo extension ----------------------------------------------------------

todo_include_todos = True


# -- intersphinx -------------------------------------------------------------

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "gensim": ("https://radimrehurek.com/gensim", None),
    "matplotlib": ("https://matplotlib.org", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "pyteomics": ("https://pyteomics.readthedocs.io/en/latest/", None),
    "rdkit": ("http://rdkit.org/docs", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
}