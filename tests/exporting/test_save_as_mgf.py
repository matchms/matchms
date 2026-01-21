import os
import tempfile
import numpy as np
import pytest
from matchms.exporting import save_as_mgf
from matchms.importing import load_from_mgf
from ..builder_Spectrum import SpectrumBuilder


def load_test_spectra_file(test_filename):
    module_root = os.path.join(os.path.dirname(__file__), "..")
    spectra_file = os.path.join(module_root, "testdata", test_filename)
    spectra = list(load_from_mgf(spectra_file))
    return spectra


def test_save_as_mgf_single_spectrum():
    """Test saving spectrum to .mgf file"""
    spectrum = (
        SpectrumBuilder()
        .with_mz(np.array([100, 200, 300], dtype="float"))
        .with_intensities(np.array([10, 10, 500], dtype="float"))
        .with_metadata(
            {"charge": -1, "inchi": '"InChI=1S/C6H12"', "pepmass": (100, 10.0), "test_field": "test"},
            metadata_harmonization=False,
        )
        .build()
    )

    # Write to test file
    with tempfile.TemporaryDirectory() as d:
        filename = os.path.join(d, "test.mgf")
        filename2 = os.path.join(d, "test2.mgf")
        save_as_mgf(spectrum, filename)
        save_as_mgf(spectrum, filename2, file_mode="w")

        # test if file exists
        assert os.path.isfile(filename)
        assert os.path.isfile(filename2)

        # Test if content of mgf file is correct
        with open(filename, "r", encoding="utf-8") as f:
            mgf_content = f.readlines()
        assert mgf_content[0] == "BEGIN IONS\n"
        assert mgf_content[2] == "CHARGE=1-\n"
        assert mgf_content[4] == "TEST_FIELD=test\n"
        assert mgf_content[7].split(" ")[0] == "300.0"


def test_save_as_mgf_spectrum_list():
    """Test saving spectrum list to .mgf file"""
    mz = np.array([100, 200, 300], dtype="float")
    intensities = np.array([10, 10, 500], dtype="float")
    builder = SpectrumBuilder().with_mz(mz).with_intensities(intensities)
    spectrum1 = builder.with_metadata({"test_field": "test1"}, metadata_harmonization=False).build()
    spectrum2 = builder.with_metadata({"test_field": "test2"}, metadata_harmonization=False).build()
    spectrum3 = None

    # Write to test file
    with tempfile.TemporaryDirectory() as d:
        filename = os.path.join(d, "test.mgf")
        save_as_mgf([spectrum1, spectrum2, spectrum3], filename)

        # test if file exists
        assert os.path.isfile(filename)

        # Test if content of mgf file is correct
        with open(filename, "r", encoding="utf-8") as f:
            mgf_content = f.readlines()
        assert mgf_content[5] == mgf_content[12] == "END IONS\n"
        assert mgf_content[1].split("=")[1] == "test1\n"
        assert mgf_content[8].split("=")[1] == "test2\n"

        end_ion_count_actual = len([x for x in mgf_content if x == "END IONS\n"])
        assert end_ion_count_actual == 2


@pytest.mark.parametrize(
    "style, expected",
    [
        ("matchms", ["PRECURSOR_MZ=100.1\n", "PRECURSOR_MZ=200.2\n"]),
        ("nist", ["PRECURSORMZ=100.1\n", "PRECURSORMZ=200.2\n"]),
        ("gnps", ["PEPMASS=100.1\n", "PEPMASS=200.2\n"]),
    ],
)
def test_save_as_mgf_export_style(style, expected):
    """Test saving spectrum list to .mgf file using differnt export styles."""
    mz = np.array([100, 200], dtype="float")
    intensities = np.array([10, 500], dtype="float")
    builder = SpectrumBuilder().with_mz(mz).with_intensities(intensities)
    spectrum1 = builder.with_metadata({"precursor_mz": 100.1}, metadata_harmonization=False).build()
    spectrum2 = builder.with_metadata({"precursor_mz": 200.2}, metadata_harmonization=False).build()

    # Write to test file
    with tempfile.TemporaryDirectory() as d:
        filename = os.path.join(d, "test.mgf")
        save_as_mgf([spectrum1, spectrum2], filename, export_style=style)

        # Test if content of mgf file is correct
        with open(filename, "r", encoding="utf-8") as f:
            mgf_content = f.readlines()
        assert mgf_content[1] == expected[0]
        assert mgf_content[7] == expected[1]


@pytest.mark.parametrize(
    "charge, ionmode, parent_mass", [(-1, "negative", 218.5), (2, "positive", "n/a"), (None, "n/a", 250)]
)
def test_save_load_mgf_consistency(tmpdir, charge, ionmode, parent_mass):
    """Test saving and loading spectrum to .mgf file"""
    mz = np.array([100.1, 200.02, 300.003], dtype="float")
    intensities = np.array([0.01, 0.02, 1.0], dtype="float")
    metadata = {"precursor_mz": 200.5, "charge": charge, "ionmode": ionmode, "parent_mass": parent_mass}
    builder = SpectrumBuilder().with_mz(mz).with_intensities(intensities)
    spectrum1 = builder.with_metadata(metadata, metadata_harmonization=True).build()
    spectrum2 = builder.with_metadata(metadata, metadata_harmonization=True).build()

    # Write to test file
    filename = os.path.join(tmpdir, "test.mgf")
    save_as_mgf([spectrum1, spectrum2], filename)

    # Test if file exists
    assert os.path.isfile(filename)

    # Test importing spectra again
    spectrum_imports = list(load_from_mgf(filename))
    assert spectrum_imports[0].get("precursor_mz") == 200.5
    assert spectrum_imports[0].get("charge") == charge
    assert spectrum_imports[0].get("ionmode") == ionmode
    assert spectrum_imports[0].get("parent_mass") == str(parent_mass)


@pytest.mark.parametrize("test_file", ["testdata.mgf", "pesticides.mgf"])
def test_write_append(test_file, tmpdir):
    expected = load_test_spectra_file(test_file)
    tmp_file = os.path.join(tmpdir, "testfile.mgf")
    save_as_mgf(expected[:2], tmp_file)
    save_as_mgf(expected[2:], tmp_file)

    actual = list(load_from_mgf(tmp_file))

    assert expected == actual
