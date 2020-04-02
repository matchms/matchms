# test functions

import os
import unittest

from matchms.importing import load_from_mgf

# Use test data from following folder
PATH_TEST = os.path.dirname(os.path.abspath(__file__))
PATH_TESTDATA = os.path.join(PATH_TEST, 'testdata')

class ModelGenerationSuite(unittest.TestCase):
    """Basic test cases."""

    def test_load_mgf_data(self):
        test_mgf_file = os.path.join(PATH_TESTDATA,
                                     'GNPS-COLLECTIONS-PESTICIDES-NEGATIVE.mgf')
        spectra = load_from_mgf(test_mgf_file)
        # Test size and types
        assert len(spectra) == len(spec_dict) == len(ms_docs) == 76, 'number of load spectra not correct'
        assert type(spectra[0]) == Spectrum.Spectrum, 'expected list of Spectrum.Spectrum objects'
        assert spectra[0].metadata['smiles'] == 'C1=CC=C2C(=C1)NC(=N2)C3=CC=CO3', 'expected other smiles entry'
        assert spectra[-1].precursor_mz == 342.024

if __name__ == '__main__':
    unittest.main()
