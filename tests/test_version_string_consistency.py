import os
import re
from matchms import __version__ as expected_version


def test_version_string_consistency():
    """Check whether version in meta.yaml is consistent with that in matchms.__version__"""

    repository_root = os.path.join(os.path.dirname(__file__), '..')
    fixture = os.path.join(repository_root, "meta.yaml")

    with open(fixture, "r") as f:
        metayaml_contents = f.read()

    match = re.search(r"^{% set version = \"(?P<semver>.*)\" %}$", metayaml_contents, re.MULTILINE)
    actual_version = match["semver"]

    assert expected_version == actual_version, "Expected version string used in meta.yaml to be consistent with" \
                                               " that in matchms.__version__"
