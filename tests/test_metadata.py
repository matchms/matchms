"""Tests for Metadata class."""
import pytest
from matchms import Metadata


@pytest.mark.parametrize("input_dict, harmonize, expected", [
    [None, True, {}],
    [{"precursor_mz": 101.01}, True, {"precursor_mz": 101.01}],
    [{"precursormz": 101.01}, True, {"precursor_mz": 101.01}],
    [{"precursormz": 101.01}, False, {"precursormz": 101.01}],
    [{"ionmode": "Negative"}, True, {"ionmode": "negative"}]])
def test_metadata_init(input_dict, harmonize, expected):
    metadata = Metadata(input_dict, harmonize_defaults=harmonize)
    assert metadata.metadata == expected, \
        "Expected different _metadata dictionary."


@pytest.mark.parametrize("harmonize, set_key, set_value, expected", [
    [True, "precursor_mz", 101.01, {"precursor_mz": 101.01}],
    [True, "precursormz", 101.01, {"precursor_mz": 101.01}],
    [False, "precursormz", 101.01, {"precursormz": 101.01}],
    [True, "ionmode", "NEGATIVE", {"ionmode": "negative"}]])
def test_metadata_setter(harmonize, set_key, set_value, expected):
    metadata = Metadata(harmonize_defaults=harmonize)
    metadata.set(set_key, set_value)
    assert metadata.metadata == expected, \
        "Expected different _metadata dictionary."
