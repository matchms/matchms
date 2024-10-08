import os
import tempfile
import numpy as np
from matchms.exporting import save_as_mzml
from matchms.importing import load_from_mzml
from ..builder_Spectrum import SpectrumBuilder


def test_save_as_mgf_single_spectrum():
    """Test saving spectrum to .mgf file"""
    spectrum = SpectrumBuilder().with_mz(
        np.array([100, 200, 300], dtype="float")).with_intensities(
            np.array([10, 10, 500], dtype="float")).with_metadata(
                {"charge": -1,
                 "inchi": '"InChI=1S/C6H12"',
                 "pepmass": (100, 10.0),
                 "test_field": "test"},
                metadata_harmonization=True).build()

    # Write to test file
    with tempfile.TemporaryDirectory() as d:
        filename = os.path.join(d, "test.mgf")
        save_as_mzml(spectrum, filename)

        # test if file exists
        assert os.path.isfile(filename)

        output_spectra = list(load_from_mzml(filename))
        assert len(output_spectra) == 1
        assert isinstance(output_spectra, list)
        assert np.all(output_spectra[0].mz == spectrum.mz)
        assert np.all(output_spectra[0].intensities == spectrum.intensities)
        assert output_spectra[0].metadata["charge"] == -1.0
        assert output_spectra[0].metadata["inchi"] == spectrum.get("inchi")
        assert output_spectra[0].metadata["precursor_mz"] == spectrum.get("precursor_mz")
        assert output_spectra[0].metadata["test_field"] == spectrum.get("test_field")
        assert output_spectra[0].metadata["precursor_intensity"] == spectrum.get("precursor_intensity")


def test_save_as_mgf_spectrum_list():
    """Test saving spectrum list to .mgf file"""
    mz = np.array([100, 200, 300], dtype="float")
    intensities = np.array([10, 10, 500], dtype="float")
    builder = SpectrumBuilder().with_mz(mz).with_intensities(intensities)
    spectrum1 = builder.with_metadata({"test_field": "test1"},
                                      metadata_harmonization=False).build()
    spectrum2 = builder.with_metadata({"test_field": "test2"},
                                      metadata_harmonization=False).build()
    spectrum3 = None

    # Write to test file
    with tempfile.TemporaryDirectory() as d:
        filename = os.path.join(d, "test.mgf")
        save_as_mzml([spectrum1, spectrum2, spectrum3], filename)

        # test if file exists
        assert os.path.isfile(filename)

        output_spectra = list(load_from_mzml(filename))
        assert len(output_spectra) == 2
        assert isinstance(output_spectra, list)
        assert np.all(output_spectra[0].mz == spectrum1.mz)
        assert np.all(output_spectra[0].intensities == spectrum1.intensities)
        assert output_spectra[0].get("test_field") == spectrum1.get('test_field')
        assert output_spectra[1].get("test_field") == spectrum2.get('test_field')


def test_load_and_save_existing_file():
    module_root = os.path.join(os.path.dirname(__file__), "..")
    spectra_file = os.path.join(module_root, "testdata", "testdata.mzml")
    spectra = list(load_from_mzml(spectra_file))
    # Write to test file
    with tempfile.TemporaryDirectory() as d:
        filename = os.path.join(d, "test.mgf")
        save_as_mzml(spectra, filename)
        # test if file exists
        assert os.path.isfile(filename)

        output_spectra = list(load_from_mzml(filename))
        print(output_spectra)
        assert len(output_spectra) == len(spectra)
