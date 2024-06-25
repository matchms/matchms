import filecmp
import json
import os
import numpy as np
import pytest
from matchms.exporting.metadata_export import (_get_metadata_dict,
                                               _subset_metadata,
                                               export_metadata_as_csv,
                                               export_metadata_as_json,
                                               get_metadata_as_array)
from matchms.importing import load_from_msp
from tests.builder_Spectrum import SpectrumBuilder


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


def test_export_as_csv(tmp_path, spectra):
    module_root = os.path.join(os.path.dirname(__file__), "..")
    expected = os.path.join(module_root, "testdata", "expected_metadata.csv")
    outpath = tmp_path / "metadata.csv"
    export_metadata_as_csv(spectra, outpath)

    filecmp.cmp(outpath, expected)


def test_subset_metadata(spectra):
    actual, colnames = get_metadata_as_array(spectra)

    subset = ["inchi", "smiles"]
    newdata, newcolnames = _subset_metadata(subset, actual, colnames)

    np.testing.assert_equal(newdata, actual[subset])
    assert newcolnames == set(subset)


def test_export_metadata_as_json(tmp_path, spectra):
    outpath = tmp_path / "metadata.json"
    module_root = os.path.join(os.path.dirname(__file__), "..")
    expected = os.path.join(module_root, "testdata", "expected_metadata.json")

    export_metadata_as_json(spectra, outpath)
    filecmp.cmp(outpath, expected)


@pytest.mark.parametrize("file_name", ["metadata.csv", "metadata.json"])
def test_export_metadata_none_spectra(tmp_path, spectra, file_name):
    outpath = os.path.join(tmp_path, file_name)

    spectra.append(None)
    export_metadata_as_json(spectra, outpath)

    with open(outpath, "r") as f:
        actual = json.load(f)
        assert len(actual) == (len(spectra) - 1)
