"""Tests for Metadata class."""
import pytest
from matchms import Metadata


@pytest.mark.parametrize("input_dict, expected", [
    [None, {}],
    [{"precursor_mz": 101.01}, {"precursor_mz": 101.01}],
    [{"precursormz": 101.01}, {"precursor_mz": 101.01}]])
def test_metadata_init(input_dict, expected):
    metadata = Metadata(input_dict)
    assert metadata.metadata == expected, \
        "Expected different _metadata dictionary."
