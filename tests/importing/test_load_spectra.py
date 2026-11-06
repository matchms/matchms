import os
import pytest
from matchms import SpectraCollection, Spectrum
from matchms.importing.load_spectra import load_ms2_dataset, load_spectra


def test_load_spectra_unknown_file(tmp_path):
    """Tests if unknown file raises an AssertionError."""
    with pytest.raises(AssertionError):
        load_spectra(os.path.join(tmp_path, "file_that_does_not_exist.json"))


@pytest.mark.parametrize(
    "filename, ftype, metadata_harmonization, expected_num_spectra",
    [
        ["pesticides.mgf", None, True, 76],
        ["pesticides.mgf", "auto", True, 76],
        ["testdata.mgf", "mgf", True, 30],
        ["testdata.mgf", "auto", True, 30],
        ["testdata.mgf", "mgf", False, 30],
        ["testdata.mzml", None, True, 10],
        ["testdata.mzml", "auto", True, 10],
        ["testdata.mzXML", None, True, 1],
        ["testdata.mzXML", "auto", True, 1],
        ["massbank_five_spectra.msp", None, True, 5],
        ["massbank_five_spectra.msp", "auto", True, 5],
    ],
)
def test_load_spectra(filename, ftype, metadata_harmonization, expected_num_spectra):
    """Test if spectrum files are loaded correctly."""
    tests_root = os.path.join(os.path.dirname(__file__), "../testdata")
    file = os.path.join(tests_root, filename)

    actual = list(
        load_spectra(
            file,
            metadata_harmonization=metadata_harmonization,
            ftype=ftype,
        )
    )

    assert isinstance(actual, list), "expected list of spectra"
    assert len(actual) == expected_num_spectra
    for spectrum in actual:
        assert isinstance(spectrum, Spectrum)


@pytest.mark.parametrize(
    "filename, ftype, metadata_harmonization, expected_num_spectra",
    [
        ["pesticides.mgf", "auto", True, 76],
        ["testdata.mgf", "mgf", True, 30],
        ["testdata.mgf", "auto", False, 30],
        ["testdata.mzml", "auto", True, 10],
        ["testdata.mzXML", "auto", True, 1],
        ["massbank_five_spectra.msp", "auto", True, 5],
    ],
)
def test_load_ms2_dataset(filename, ftype, metadata_harmonization, expected_num_spectra):
    """Test if spectrum files are loaded as SpectraCollection."""
    tests_root = os.path.join(os.path.dirname(__file__), "../testdata")
    file = os.path.join(tests_root, filename)

    actual = load_ms2_dataset(
        file,
        metadata_harmonization=metadata_harmonization,
        ftype=ftype,
    )

    assert isinstance(actual, SpectraCollection)
    assert len(actual) == expected_num_spectra