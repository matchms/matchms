"""Sphinx configuration for the matchms 1.0 documentation."""

from __future__ import annotations

import sys
import tomllib
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version as package_version
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
sys.path.insert(0, str(PROJECT_ROOT))

project = "matchms"
author = "matchms development team"
copyright = f"2020-{datetime.now(timezone.utc).year}, matchms contributors"

# The CI docs build installs matchms. For local previews, fall back to the
# version declared by pyproject.toml instead of importing matchms during setup.
try:
    release = package_version("matchms")
except PackageNotFoundError:
    project_config = PROJECT_ROOT / "pyproject.toml"
    if project_config.exists():
        with project_config.open("rb") as stream:
            metadata = tomllib.load(stream)
        release = (metadata.get("project", {}).get("version")
                   or metadata.get("tool", {}).get("poetry", {}).get("version")
                   or "0+unknown")
    else:
        release = "0+unknown"
version = ".".join(release.split(".")[:2]) if release[0].isdigit() else release

extensions = [
    "sphinx.ext.apidoc",  # Included in Sphinx >= 8.2
    "sphinx.ext.autodoc",
    "sphinx.ext.coverage",
    "sphinx.ext.doctest",
    "sphinx.ext.intersphinx",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx_rtd_theme",  # Registers theme assets/jQuery consistently
]

root_doc = "index"
language = "en"
source_suffix = {".rst": "restructuredtext"}
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# Sphinx's own apidoc replaces the unmaintained sphinxcontrib-apidoc extension.
# Do not generate undocumented private modules or overwrite hand-written guides.
apidoc_modules = [
    {
        "path": str(PROJECT_ROOT / "matchms"),
        "destination": "api",
        "exclude_patterns": ["**/tests/**", "**/test_*.py"],
        "separate_modules": True,
        "module_first": True,
        "automodule_options": {"members", "show-inheritance"},
    }
]

# Document the *public* API from docstrings. Automatically publishing every
# undocumented member masks missing documentation and inflates the site.
autodoc_default_options = {
    "members": True,
    "show-inheritance": True,
}
autoclass_content = "both"
napoleon_google_docstring = False
napoleon_numpy_docstring = True

doctest_test_doctest_blocks = ""

html_theme = "sphinx_rtd_theme"
html_theme_options = {
    "logo_only": False,
    "style_external_links": True,
    "navigation_depth": 4,
    "collapse_navigation": False,
}
html_static_path = ["_static"] if (HERE / "_static").is_dir() else []
if (HERE / "_static" / "matchms.png").is_file():
    html_logo = "_static/matchms.png"

# These inventories are useful for NumPy and Python type references. Avoid
# optional/down dependencies with unreliable inventories in CI.
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
}
