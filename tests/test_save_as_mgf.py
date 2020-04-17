import os
import numpy as np
from matchms import Spectrum
from matchms.exporting import save_as_mgf


def test_save_as_mgf_single_spectrum(tmpdir):
    """Test saving spectrum to .mgf file"""
    spectrum = Spectrum(mz=np.array([100, 200, 300], dtype="float"),
                        intensities=np.array([10, 10, 500], dtype="float"),
                        metadata={"charge": -1,
                                  "inchi": '"InChI=1S/C6H12"',
                                  "pepmass": (100, 10.0),
                                  "test_field": 'test'})
    # Write to test file
    file = tmpdir.join('test.mgf')
    save_as_mgf(spectrum, file)
    assert os.path.isfile(file)

    # Test if content of mgf file is correct
    with open(file, 'r') as f:
        mgf_content = f.readlines()
    assert mgf_content[0] == 'BEGIN IONS\n'
    assert mgf_content[2] == 'CHARGE=1-\n'
    assert mgf_content[4] == 'TEST_FIELD=test\n'
    assert mgf_content[7].split(" ")[0] == '300.0'
    # Remove testfile again
    os.remove('test.mgf')


def test_save_as_mgf_spectrum_list():
    """Test saving spectrum list to .mgf file"""
    spectrum1 = Spectrum(mz=np.array([100, 200, 300], dtype="float"),
                         intensities=np.array([10, 10, 500], dtype="float"),
                         metadata={"test_field": 'test1'})

    spectrum2 = Spectrum(mz=np.array([100, 200, 300], dtype="float"),
                         intensities=np.array([10, 10, 500], dtype="float"),
                         metadata={"test_field": 'test2'})
    # Write to test file
    save_as_mgf([spectrum1, spectrum2], 'test.mgf')
    assert os.path.isfile('test.mgf')

    # Test if content of mgf file is correct
    with open('test.mgf', 'r') as f:
        mgf_content = f.readlines()
    assert mgf_content[5] == mgf_content[12] == 'END IONS\n'
    assert mgf_content[1].split("=")[1] == 'test1\n'
    assert mgf_content[8].split("=")[1] == 'test2\n'
    # Remove testfile again
    os.remove('test.mgf')


if __name__ == "__main__":
    test_save_as_mgf_single_spectrum(tmpdir)
    test_save_as_mgf_spectrum_list()
