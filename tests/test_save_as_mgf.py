import os
import numpy as np
from matchms import Spectrum
from matchms.exporting import save_as_mgf
import tempfile


def test_save_as_mgf_single_spectrum():
    """Test saving spectrum to .mgf file"""
    spectrum = Spectrum(mz=np.array([100, 200, 300], dtype="float"),
                        intensities=np.array([10, 10, 500], dtype="float"),
                        metadata={"charge": -1,
                                  "inchi": '"InChI=1S/C6H12"',
                                  "pepmass": (100, 10.0),
                                  "test_field": 'test'})
    # Write to test file
    with tempfile.NamedTemporaryFile() as fp:
        save_as_mgf(spectrum, fp.name)

        # test if file exists
        assert os.path.isfile(fp.name)

        # Test if content of mgf file is correct
        with open(fp.name, 'r') as f:
            mgf_content = f.readlines()
        assert mgf_content[0] == 'BEGIN IONS\n'
        assert mgf_content[2] == 'CHARGE=1-\n'
        assert mgf_content[4] == 'TEST_FIELD=test\n'
        assert mgf_content[7].split(" ")[0] == '300.0'


def test_save_as_mgf_spectrum_list():
    """Test saving spectrum list to .mgf file"""
    spectrum1 = Spectrum(mz=np.array([100, 200, 300], dtype="float"),
                         intensities=np.array([10, 10, 500], dtype="float"),
                         metadata={"test_field": 'test1'})

    spectrum2 = Spectrum(mz=np.array([100, 200, 300], dtype="float"),
                         intensities=np.array([10, 10, 500], dtype="float"),
                         metadata={"test_field": 'test2'})
    # Write to test file
    with tempfile.NamedTemporaryFile() as fp:
        save_as_mgf([spectrum1, spectrum2], fp.name)

        # test if file exists
        assert os.path.isfile(fp.name)

        # Test if content of mgf file is correct
        with open(fp.name, 'r') as f:
            mgf_content = f.readlines()
        assert mgf_content[5] == mgf_content[12] == 'END IONS\n'
        assert mgf_content[1].split("=")[1] == 'test1\n'
        assert mgf_content[8].split("=")[1] == 'test2\n'


if __name__ == "__main__":
    test_save_as_mgf_single_spectrum()
    test_save_as_mgf_spectrum_list()
