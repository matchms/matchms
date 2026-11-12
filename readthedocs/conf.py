"""Sphinx configuration for the matchms 1.0 documentation."""

from __future__ import annotations

import re
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
# The generated pages already contain :members:. Keep this OFF globally so
# package overview pages can show their docstring without re-documenting
# classes re-exported from matchms.__init__ and subpackage __init__ modules.
autodoc_default_options = {
    "show-inheritance": True,
}
autoclass_content = "both"
# Preserve useful type annotations in signatures. The few short aliases that
# lack cross-project inventory targets are allowlisted below.
autodoc_typehints = "signature"
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
    # External objects in docstrings: pandas.DataFrame, matplotlib axes, etc.
    "pandas": ("https://pandas.pydata.org/docs/", None),
    "matplotlib": ("https://matplotlib.org/stable/", None),
}


# Canonical API objects are defined in module pages, e.g.
# matchms.spectrum.Spectrum rather than the re-export matchms.Spectrum.
# Prevent generating a second object target on each package overview page.
def _package_overviews_without_duplicate_members(app, docname, source):
    if not docname.startswith("api/"):
        return

    module = docname.removeprefix("api/")
    if module != "matchms" and not module.startswith("matchms."):
        return

    module_path = PROJECT_ROOT.joinpath(*module.split("."))
    if not (module_path / "__init__.py").is_file():
        return  # Real implementation module: retain all its members.

    pattern = re.compile(
        r"(^\.\. automodule::\s+" + re.escape(module) +
        r"[ \t]*\n)(?P<options>(?:[ \t]+:[\w-]+:.*\n)*)",
        re.MULTILINE,
    )

    def keep_module_docstring_only(match):
        option_lines = match.group("options").splitlines(keepends=True)
        option_lines = [
            line for line in option_lines
            if not re.match(r"^[ \t]+:(?:members|imported-members):", line)
        ]
        return match.group(1) + "".join(option_lines)

    source[0] = pattern.sub(keep_module_docstring_only, source[0], count=1)


# Some annotation aliases are short spellings, or refer to private implementation
# details which do not have public Sphinx targets. Keep this list intentional:
# never ignore the whole matchms namespace, and never silence all ref warnings.
nitpick_ignore = {
    ("py:class", "np.ndarray"),
    ("py:class", "pd.Series"),
    ("py:class", "plt.Axes"),
    ("py:class", "Path"),
    ("py:class", "csr_array"),
    ("py:class", "np.dtype"),
    ("py:class", "numpy.array"),
    ("py:class", "numpy.float32"),
    ("py:class", "numpy.float64"),
}

nitpick_ignore_regex = [
    (
        r"py:class",
        r"matchms\.similarity\.(?:"
        r"entropy_search\._(?:PreparedEntropySpectra|SearchView)|"
        r"flash_similarity\._BaseFlashSimilarity|"
        r"flash_index\._(?:CosineKernelData|EntropyKernelData))",
    ),
]


def setup(app):
    app.connect("source-read", _package_overviews_without_duplicate_members)
