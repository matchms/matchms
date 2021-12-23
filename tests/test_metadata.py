"""Tests for Metadata class."""
import numpy as np
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
    assert metadata.data == expected, \
        "Expected different _metadata dictionary."


@pytest.mark.parametrize("harmonize, set_key, set_value, expected", [
    [True, "precursor_mz", 101.01, {"precursor_mz": 101.01}],
    [True, "precursormz", 101.01, {"precursor_mz": 101.01}],
    [False, "precursormz", 101.01, {"precursormz": 101.01}],
    [True, "ionmode", "NEGATIVE", {"ionmode": "negative"}]])
def test_metadata_setter(harmonize, set_key, set_value, expected):
    metadata = Metadata(harmonize_defaults=harmonize)
    metadata.set(set_key, set_value)
    assert metadata.data == expected, \
        "Expected different _metadata dictionary."


@pytest.mark.parametrize("harmonize, set_key, set_value, get_key, get_value", [
    [True, "precursor_mz", 101.01, "precursor_mz", 101.01],
    [True, "precursormz", 101.01, "precursor_mz", 101.01],
    [False, "precursormz", 101.01, "precursormz", 101.01],
    [True, "ionmode", "NEGATIVE", "ionmode", "negative"]])
def test_metadata_setter_getter(harmonize, set_key, set_value, get_key, get_value):
    metadata = Metadata(harmonize_defaults=harmonize)
    metadata.set(set_key, set_value)
    assert metadata.get(get_key) == get_value, \
        "Expected different _metadata dictionary."


@pytest.mark.parametrize("harmonize, set_key, set_value, get_key, get_value", [
    [True, "precursor_mz", 101.01, "precursor_mz", 101.01],
    [True, "precursormz", 101.01, "precursor_mz", 101.01],
    [False, "precursormz", 101.01, "precursormz", 101.01],
    [True, "ionmode", "NEGATIVE", "ionmode", "negative"]])
def test_metadata_setitem_getitem(harmonize, set_key, set_value, get_key, get_value):
    metadata = Metadata(harmonize_defaults=harmonize)
    metadata[set_key] = set_value
    assert metadata[get_key] == metadata.get(get_key) == get_value, \
        "Expected different _metadata dictionary."


@pytest.mark.parametrize("dict1, dict2, expected", [
    [{"precursor_mz": 101.01}, {"precursor_mz": 101.01}, True],
    [{"abc": ["a", 5, 7.01]}, {"abc": ["a", 7.01, 5]}, False],
    [{"abc": 3.7, "def": 4.2},
     {"def": 4.2, "abc": 3.7}, True],
    [{"A": np.array([1.2, 1.3, 1.4, 1.5001, 1.6])},
     {"A": np.array([1.2, 1.3, 1.4, 1.5, 1.6])}, False],
    [{"A": np.array([1.2, 1.3, 1.4, 1.5001, 1.6])},
     {"A": np.array([1.2, 1.3, 1.4, 1.5001, 1.6])}, True],
    [{"abc": ["a", 5, [7, 1]]}, {"abc":  ["a", 5, [1, 7]]}, False]])
def test_metadata_equal(dict1, dict2, expected):
    metadata1 = Metadata(dict1)
    metadata2 = Metadata(dict2)
    if expected is True:
        assert metadata1 == metadata2, "Expected metadata to be equal."
    else:
        assert metadata1 != metadata2, "Expected metadata NOT to be equal."
