import os
import re
from matchms import __version__ as expected_version


def test_version_string_consistency_meta_yml():
    """Check whether version in meta.yaml is consistent with that in matchms.__version__"""

    repository_root = os.path.join(os.path.dirname(__file__), '..')
    fixture = os.path.join(repository_root, "conda/meta.yaml")

    with open(fixture, "r") as f:
        contents = f.read()

    match = re.search(r"^{% set version = \"(?P<semver>.*)\" %}$", contents, re.MULTILINE)
    actual_version = match["semver"]

    assert expected_version == actual_version, "Expected version string used in conda/meta.yaml to be consistent with" \
                                               " that in matchms.__version__"


def test_version_string_consistency_setup_cfg():
    """Check whether version in setup.cfg is consistent with that in matchms.__version__"""

    repository_root = os.path.join(os.path.dirname(__file__), '..')
    fixture = os.path.join(repository_root, "setup.cfg")

    with open(fixture, "r") as f:
        contents = f.read()

    match = re.search(r"^current_version = (?P<semver>.*)$", contents, re.MULTILINE)
    actual_version = match["semver"]

    assert expected_version == actual_version, "Expected version string used in setup.cfg to be consistent with" \
                                               " that in matchms.__version__"
