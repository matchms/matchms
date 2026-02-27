import json
import os
import numpy as np
import pytest
from matchms.exporting import save_as_json
from matchms.importing import load_from_json
from tests.builder_Spectrum import SpectrumBuilder


@pytest.fixture
def builder() -> SpectrumBuilder:
    mz = np.array([100, 200, 300], dtype="float")
    intensities = np.array([10, 10, 500], dtype="float")
    builder = SpectrumBuilder().with_mz(mz).with_intensities(intensities)
    return builder


def load_test_spectra_file(test_filename):
    module_root = os.path.join(os.path.dirname(__file__), "..")
    spectra_file = os.path.join(module_root, "testdata", test_filename)
    spectra = list(load_from_json(spectra_file))
    return spectra


def test_save_and_load_json_single_spectrum(tmp_path, builder):
    """Test saving spectrum to .json file"""
    spectrum = builder.with_metadata(
        {"charge": -1, "inchi": '"InChI=1S/C6H12"', "precursor_mz": 222.2, "test_field": "test"}
    ).build()
    # Write to test file
    filename = os.path.join(tmp_path, "test.json")
    save_as_json(spectrum, filename)

    # test if file exists
    assert os.path.isfile(filename)

    # Test if content of json file is correct
    spectrum_import = load_from_json(filename, metadata_harmonization=False)[0]
    assert spectrum_import == spectrum, "Original and saved+loaded spectrum not identical"


@pytest.mark.parametrize("metadata_harmonization", [True, False])
def test_save_and_load_json_spectrum_list(metadata_harmonization, tmp_path, builder):
    """Test saving spectrum list to .json file"""
    spectrum1 = builder.with_metadata({"test_field": "test1"}, metadata_harmonization=metadata_harmonization).build()
    spectrum2 = builder.with_metadata({"test_field": "test2"}, metadata_harmonization=metadata_harmonization).build()
    spectrum3 = None

    # Write to test file
    filename = os.path.join(tmp_path, "test.json")
    save_as_json([spectrum1, spectrum2, spectrum3], filename)

    # test if file exists
    assert os.path.isfile(filename)

    # Test if content of json file is correct
    spectrum_imports = load_from_json(filename, metadata_harmonization=metadata_harmonization)
    assert spectrum_imports[0] == spectrum1, "Original and saved+loaded spectrum not identical"
    assert spectrum_imports[1] == spectrum2, "Original and saved+loaded spectrum not identical"
    assert len(spectrum_imports) == 2


@pytest.mark.parametrize("style, expected", [("matchms", "precursor_mz"), ("nist", "PrecursorMZ")])
def test_save_as_json_different_export_styles(tmp_path, builder, style, expected):
    spectrum = builder.with_metadata({"precursor_mz": 123}).build()
    filename = tmp_path / "test_matchms.json"
    save_as_json([spectrum], str(filename), export_style=style)

    with open(filename, "r", encoding="utf-8") as f:
        data = json.load(f)
        assert expected in data[0]


def test_load_from_json_zero_peaks(tmp_path):
    spectrum1 = SpectrumBuilder().with_metadata({"test_field": "test1"}).build()

    filename = tmp_path / "test.json"

    save_as_json([spectrum1], filename)

    # test if file exists
    assert os.path.isfile(filename)

    # Test if content of json file is correct
    spectrum_imports = load_from_json(filename)
    assert len(spectrum_imports) == 0, "Spectrum without peaks should be skipped"


def test_load_from_json_with_minimal_json(tmp_path, builder):
    filename = tmp_path / "test.json"
    body = '[{"test_field": "test1", "peaks_json": [[100.0, 10.0], [200.0, 10.0], [300.0, 500.0]]}]'

    with open(filename, "w", encoding="utf-8") as f:
        f.write(body)

    spectrum_imports = load_from_json(filename, metadata_harmonization=False)

    expected = builder.with_metadata({"test_field": "test1"}, metadata_harmonization=False).build()

    assert spectrum_imports == [expected], "Loaded JSON document not identical to expected Spectrum"


def test_save_as_json_with_minimal_json(tmp_path, builder):
    filename = tmp_path / "test.json"

    spectrum1 = builder.with_metadata({"test_field": "test1"}, metadata_harmonization=False).build()

    save_as_json([spectrum1], filename)

    with open(filename, encoding="utf-8") as f:
        spectrum_imports = json.load(f)

    expected = [{"test_field": "test1", "peaks_json": [[100.0, 10.0], [200.0, 10.0], [300.0, 500.0]]}]
    assert spectrum_imports == expected, "Saved Spectrum not identical to expected JSON Document"


@pytest.mark.parametrize("filename, expected_length", [["gnps_spectra.json", 5]])
def test_read_gnps_spectra(filename, expected_length):
    actual = load_test_spectra_file(filename)

    assert len(actual) == expected_length
