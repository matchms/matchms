import os
import tempfile
import numpy
from matchms import Spectrum
from matchms.exporting import save_as_mgf


def test_save_as_mgf_single_spectrum():
    """Test saving spectrum to .mgf file"""
    spectrum = Spectrum(mz=numpy.array([100, 200, 300], dtype="float"),
                        intensities=numpy.array([10, 10, 500], dtype="float"),
                        metadata={"charge": -1,
                                  "inchi": '"InChI=1S/C6H12"',
                                  "pepmass": (100, 10.0),
                                  "test_field": "test"})
    # Write to test file
    with tempfile.TemporaryDirectory() as d:
        filename = os.path.join(d, "test.mgf")
        save_as_mgf(spectrum, filename)

        # test if file exists
        assert os.path.isfile(filename)

        # Test if content of mgf file is correct
        with open(filename, "r") as f:
            mgf_content = f.readlines()
        assert mgf_content[0] == "BEGIN IONS\n"
        assert mgf_content[2] == "CHARGE=1-\n"
        assert mgf_content[4] == "TEST_FIELD=test\n"
        assert mgf_content[7].split(" ")[0] == "300.0"


def test_save_as_mgf_spectrum_list():
    """Test saving spectrum list to .mgf file"""
    spectrum1 = Spectrum(mz=numpy.array([100, 200, 300], dtype="float"),
                         intensities=numpy.array([10, 10, 500], dtype="float"),
                         metadata={"test_field": "test1"})

    spectrum2 = Spectrum(mz=numpy.array([100, 200, 300], dtype="float"),
                         intensities=numpy.array([10, 10, 500], dtype="float"),
                         metadata={"test_field": "test2"})
    # Write to test file
    with tempfile.TemporaryDirectory() as d:
        filename = os.path.join(d, "test.mgf")
        save_as_mgf([spectrum1, spectrum2], filename)

        # test if file exists
        assert os.path.isfile(filename)

        # Test if content of mgf file is correct
        with open(filename, "r") as f:
            mgf_content = f.readlines()
        assert mgf_content[5] == mgf_content[12] == "END IONS\n"
        assert mgf_content[1].split("=")[1] == "test1\n"
        assert mgf_content[8].split("=")[1] == "test2\n"
