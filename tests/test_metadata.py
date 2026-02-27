"""Tests for Metadata class."""

import numpy as np
import pytest
from pickydict import PickyDict
from matchms import Metadata
from matchms.utils import load_known_key_conversions


@pytest.mark.parametrize(
    "input_dict, harmonize, expected",
    [
        [None, True, {}],
        [{"precursor_mz": 101.01}, True, {"precursor_mz": 101.01}],
        [{"precursormz": 101.01}, True, {"precursor_mz": 101.01}],
        [{"precursormz": 101.01}, False, {"precursormz": 101.01}],
        [{"charge": "2+"}, True, {"charge": "2+"}],
        [{"charge": -1}, True, {"charge": -1}],
        [{"ionmode": "Negative"}, True, {"ionmode": "Negative"}],
    ],
)
def test_metadata_init(input_dict, harmonize, expected):
    metadata = Metadata(input_dict, matchms_key_style=harmonize)
    assert metadata.data == expected, "Expected different _metadata dictionary."


@pytest.mark.parametrize(
    "harmonize, set_key, set_value, expected",
    [
        [True, "precursor_mz", 101.01, {"precursor_mz": 101.01}],
        [True, "precursormz", 101.01, {"precursor_mz": 101.01}],
        [False, "precursormz", 101.01, {"precursormz": 101.01}],
        [True, "ionmode", "NEGATIVE", {"ionmode": "NEGATIVE"}],
    ],
)
def test_metadata_setter(harmonize, set_key, set_value, expected):
    metadata = Metadata(matchms_key_style=harmonize)
    metadata.set(set_key, set_value)
    assert metadata.data == expected, "Expected different _metadata dictionary."


@pytest.mark.parametrize(
    "harmonize, set_key, set_value, get_key, get_value",
    [
        [True, "precursor_mz", 101.01, "precursor_mz", 101.01],
        [True, "precursormz", 101.01, "precursor_mz", 101.01],
        [False, "precursormz", 101.01, "precursormz", 101.01],
        [True, "ionmode", "NEGATIVE", "ionmode", "NEGATIVE"],
    ],
)
def test_metadata_setter_getter(harmonize, set_key, set_value, get_key, get_value):
    metadata = Metadata(matchms_key_style=harmonize)
    metadata.set(set_key, set_value)
    assert metadata.get(get_key) == get_value, "Expected different _metadata dictionary."


@pytest.mark.parametrize(
    "harmonize, set_key, set_value, get_key, get_value",
    [
        [True, "precursor_mz", 101.01, "precursor_mz", 101.01],
        [True, "precursormz", 101.01, "precursor_mz", 101.01],
        [False, "precursormz", 101.01, "precursormz", 101.01],
        [True, "ionmode", "NEGATIVE", "ionmode", "NEGATIVE"],
    ],
)
def test_metadata_setitem_getitem(harmonize, set_key, set_value, get_key, get_value):
    metadata = Metadata(matchms_key_style=harmonize)
    metadata[set_key] = set_value
    assert metadata[get_key] == metadata.get(get_key) == get_value, "Expected different _metadata dictionary."


@pytest.mark.parametrize(
    "input_dict, export_style, expected",
    [
        [None, "matchms", {}],
        [{"precursor_mz": 101.01}, "matchms", {"precursor_mz": 101.01}],
        [{"peptide_modifications": 1}, "massbank", {"COMMENT:PEPTIDE_MODIFICATIONS": 1}],
        [{"ionmode": "Negative"}, "massbank", {"AC$MASS_SPECTROMETRY:ION_MODE": "Negative"}],
        [{"compound_name": "Dummy"}, "nist", {"Name": "Dummy"}],
        [{"compound_name": "Dummy"}, "riken", {"NAME": "Dummy"}],
        [{"compound_name": "Dummy"}, "gnps", {"NAME": "Dummy"}],
    ],
)
def test_metadata_to_dict(input_dict, export_style, expected):
    metadata = Metadata(input_dict)
    assert metadata.to_dict(export_style) == expected, "Expected different metadata dictionary."


@pytest.mark.parametrize(
    "dict1, dict2, expected",
    [
        [{"precursor_mz": 101.01}, {"precursor_mz": 101.01}, True],
        [{"abc": ["a", 5, 7.01]}, {"abc": ["a", 7.01, 5]}, False],
        [{"abc": 3.7, "def": 4.2}, {"def": 4.2, "abc": 3.7}, True],
        [{"A": np.array([1.2, 1.3, 1.4, 1.5001, 1.6])}, {"A": np.array([1.2, 1.3, 1.4, 1.5, 1.6])}, False],
        [{"A": np.array([1.2, 1.3, 1.4, 1.5001, 1.6])}, {"A": np.array([1.2, 1.3, 1.4, 1.5001, 1.6])}, True],
        [{"abc": ["a", 5, [7, 1]]}, {"abc": ["a", 5, [1, 7]]}, False],
    ],
)
def test_metadata_equal(dict1, dict2, expected):
    metadata1 = Metadata(dict1)
    metadata2 = Metadata(dict2)
    if expected is True:
        assert metadata1 == metadata2, "Expected metadata to be equal."
    else:
        assert metadata1 != metadata2, "Expected metadata NOT to be equal."


def test_metadata_full_setter():
    metadata = Metadata()
    metadata.data = {"Precursor Mz": 101.01}
    assert isinstance(metadata.data, PickyDict), "Expected PickyDict"
    assert metadata["precursor_mz"] == 101.01, "Expected differnt entry"


@pytest.mark.parametrize(
    "input_dict, expected",
    [
        [
            {
                "scannumber": "1516",
                "spectrumtype": "N/A",
                "formula": "C8H16NO5P",
                "ionization": "NA",
                "compound_name": "Dicrotophos",
                "comment": "",
                "precursor_mz": 238.0844,
            },
            {"scannumber": "1516", "formula": "C8H16NO5P", "compound_name": "Dicrotophos", "precursor_mz": 238.0844},
        ]
    ],
)
def test_remove_invalid_metadata(input_dict, expected):
    metadata = Metadata(input_dict)
    metadata.harmonize_values()

    assert metadata == expected, "Expected metadata to be equal."


@pytest.mark.parametrize(
    "mapping, metadata",
    [
        [{"name": "compound_name"}, {"name": "test"}],
        [{}, {"name": "test"}],
    ],
)
def test_metadata_key_mapping(mapping: dict, metadata: dict):
    Metadata.set_key_replacements(mapping)

    sut = Metadata(metadata)
    if len(mapping) > 0:
        assert sut[next(iter(mapping.values()))] == next(iter(metadata.values()))
        assert sut[next(iter(mapping))] is None
    else:
        assert sut[next(iter(metadata))] == next(iter(metadata.values()))

    Metadata.set_key_replacements(load_known_key_conversions())
