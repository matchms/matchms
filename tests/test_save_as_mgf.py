import os
import tempfile
import numpy
from matchms.exporting import save_as_mgf
from .builder_Spectrum import SpectrumBuilder


def test_save_as_mgf_single_spectrum():
    """Test saving spectrum to .mgf file"""
    spectrum = SpectrumBuilder().with_mz(
        numpy.array([100, 200, 300], dtype="float")).with_intensities(
            numpy.array([10, 10, 500], dtype="float")).with_metadata(
                {"charge": -1,
                 "inchi": '"InChI=1S/C6H12"',
                 "pepmass": (100, 10.0),
                 "test_field": "test"}).build()

    # Write to test file
    with tempfile.TemporaryDirectory() as d:
        filename = os.path.join(d, "test.mgf")
        save_as_mgf(spectrum, filename)

        # test if file exists
        assert os.path.isfile(filename)

        # Test if content of mgf file is correct
        with open(filename, "r", encoding="utf-8") as f:
            mgf_content = f.readlines()
        assert mgf_content[0] == "BEGIN IONS\n"
        assert mgf_content[2] == "CHARGE=1-\n"
        assert mgf_content[4] == "TEST_FIELD=test\n"
        assert mgf_content[7].split(" ")[0] == "300.0"


def test_save_as_mgf_spectrum_list():
    """Test saving spectrum list to .mgf file"""
    mz = numpy.array([100, 200, 300], dtype="float")
    intensities = numpy.array([10, 10, 500], dtype="float")
    builder = SpectrumBuilder().with_mz(mz).with_intensities(intensities)
    spectrum1 = builder.with_metadata({"test_field": "test1"}).build()
    spectrum2 = builder.with_metadata({"test_field": "test2"}).build()

    # Write to test file
    with tempfile.TemporaryDirectory() as d:
        filename = os.path.join(d, "test.mgf")
        save_as_mgf([spectrum1, spectrum2], filename)

        # test if file exists
        assert os.path.isfile(filename)

        # Test if content of mgf file is correct
        with open(filename, "r", encoding="utf-8") as f:
            mgf_content = f.readlines()
        assert mgf_content[5] == mgf_content[12] == "END IONS\n"
        assert mgf_content[1].split("=")[1] == "test1\n"
        assert mgf_content[8].split("=")[1] == "test2\n"
