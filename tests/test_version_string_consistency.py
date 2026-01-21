import os
import re
from matchms import __version__ as expected_version


def test_version_string_consistency_pyproject_toml():
    """Check whether version in pyproject.toml is consistent with that in matchms.__version__"""

    repository_root = os.path.join(os.path.dirname(__file__), "..")
    fixture = os.path.join(repository_root, "pyproject.toml")

    with open(fixture, "r", encoding="utf-8") as f:
        contents = f.read()

    match = re.search(r"^version = (?P<semver>.*)$", contents, re.MULTILINE)
    actual_version = match["semver"].strip('"')

    assert expected_version == actual_version, (
        "Expected version string used in pyproject.toml to be consistent with that in matchms.__version__"
    )
