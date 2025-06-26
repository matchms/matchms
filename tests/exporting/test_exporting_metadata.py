import difflib
import json
import os
import numpy as np
import pytest
from matchms.exporting.metadata_export import (
    _get_metadata_dict,
    _get_metadata_keys,
    _subset_metadata,
    export_metadata_as_csv,
    export_metadata_as_json,
    get_metadata_as_array,
)
from matchms.importing import load_from_msp
from tests.builder_Spectrum import SpectrumBuilder


def assert_files_equal_ignoring_line_endings(file1, file2):
    with open(file1, "r", encoding="utf-8") as f1, open(file2, "r", encoding="utf-8") as f2:
        a = f1.read().replace("\r\n", "\n").replace("\r", "\n").splitlines()
        b = f2.read().replace("\r\n", "\n").replace("\r", "\n").splitlines()
        assert a == b, "\n" + "\n".join(difflib.unified_diff(b, a, fromfile=str(file2), tofile=str(file1)))
    

@pytest.fixture
def spectra():
    module_root = os.path.join(os.path.dirname(__file__), "..")
    inpath = os.path.join(module_root, "testdata", "MoNA-export-GC-MS-first10.msp")
    spectra = list(load_from_msp(inpath))
    return spectra


def test_get_metadata_dict():
    spectrum = SpectrumBuilder().with_metadata({"test": "peter", "inchi": "16583"}).build()
    assert _get_metadata_dict(spectrum, ["all"]) == {"test": "peter", "inchi": "16583"}
    assert _get_metadata_dict(spectrum) == {"test": "peter", "inchi": "16583"}
    assert _get_metadata_dict(spectrum, ["inchi"]) == {"inchi": "16583"}
    assert _get_metadata_dict(spectrum, ["smiles"]) == {}


def test_get_metadata_as_array(spectra):
    actual, colnames = get_metadata_as_array(spectra)
    assert len(actual) == 10
    assert len(colnames) == 23


@pytest.mark.parametrize("delimiter", [",", "\t"])
def test_export_as_csv(tmp_path, spectra, delimiter):
    extension = {",": "csv", "\t": "tsv"}
    module_root = os.path.join(os.path.dirname(__file__), "..")
    outpath = tmp_path / f"metadata.{extension[delimiter]}"

    export_metadata_as_csv(spectra, outpath, delimiter=delimiter)
    expected = os.path.join(module_root, "testdata", f"expected_metadata.{extension[delimiter]}")

    assert_files_equal_ignoring_line_endings(outpath, expected)


def test_subset_metadata(spectra):
    actual, colnames = get_metadata_as_array(spectra)

    subset = ["inchi", "smiles"]
    newdata, newcolnames = _subset_metadata(subset, actual, colnames)

    np.testing.assert_equal(newdata, actual[subset])
    assert newcolnames == subset


def test_export_metadata_as_json(tmp_path, spectra):
    outpath = tmp_path / "metadata.json"
    module_root = os.path.join(os.path.dirname(__file__), "..")

    export_metadata_as_json(spectra, outpath)

    expected = os.path.join(module_root, "testdata", "expected_metadata.json")

    assert_files_equal_ignoring_line_endings(outpath, expected)


@pytest.mark.parametrize("file_name", ["metadata.csv", "metadata.json"])
def test_export_metadata_none_spectra(tmp_path, spectra, file_name, caplog):
    outpath = os.path.join(tmp_path, file_name)

    spectra.append(None)
    export_metadata_as_json(spectra, outpath)

    with open(outpath, "r", encoding="utf-8") as f:
        actual = json.load(f)
        assert len(actual) == (len(spectra) - 1)

    # Test invalid field export config
    export_metadata_as_json(spectra, outpath, "invalid")
    assert "'Include_fields' must be 'all' or list of keys." in caplog.text
    with open(outpath, "r", encoding="utf-8") as f:
        actual = json.load(f)
        assert len(actual) == 0


def test__get_metadata_keys(spectra):
    """Test that _get_metadata_keys returns the correct keys."""
    expected_keys = ['author', 'compound_name', 'formula', 'inchi', 'inchikey', 'instrument', 'instrument_type',
                     'ion_type', 'ionization_energy', 'ionization_mode', 'last_auto-curation', 'license',
                     'molecular_formula', 'ms_level', 'nominal_mass', 'num_peaks', 'parent_mass',
                     'precursor_mz', 'smiles', 'smiles_2', 'spectrum_id', 'synonym', 'total_exact_mass']
    actual_keys = _get_metadata_keys(spectra)
    assert actual_keys == expected_keys