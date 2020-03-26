# test functions

import os
import numpy as np
import unittest
from MS_functions import Spectrum, load_MGF_data
from MS_functions import process_peaks, exponential_peak_filter
from MS_functions import likely_inchi_match, likely_inchikey_match

# Use test data from following folder
PATH_TESTDATA = os.path.join(os.path.dirname(__file__), 'testdata')

class ModelGenerationSuite(unittest.TestCase):
    """Basic test cases."""

    def load_MGF_data(self):
        test_mgf_file = os.path.join(PATH_TESTDATA, 'GNPS-COLLECTIONS-PESTICIDES-NEGATIVE.mgf')
        spectra, spec_dict, MS_docs, MS_docs_intensity, metadata = load_MGF_data(test_mgf_file,
                                                        file_json = None,
                                                        num_decimals = 1,
                                                        min_frag = 0.0, max_frag = 1000.0,
                                                        min_loss = 5.0, max_loss = 500.0,
                                                        min_intensity_perc = 0,
                                                        exp_intensity_filter = 0.8,
                                                        min_keep_peaks_0 = 10,
                                                        min_keep_peaks_per_mz = 20/200,
                                                        min_peaks = 5,
                                                        max_peaks = None,
                                                        peak_loss_words = ['peak_', 'loss_'])
        # Test size and types
        assert len(spectra) == len(spec_dict) == len(MS_docs) == 76, 'number of load spectra not correct'
        assert type(spectra[0]) == MS_functions.Spectrum, 'expected list of MS_functions.Spectrum objects'
        assert spectra[0].smiles == 'C1=CC=C2C(=C1)NC(=N2)C3=CC=CO3', 'expected other smiles entry'
        # Test some specific entries:
        assert MS_docs[0][:5] == ['peak_75.0', 'peak_94.9', 'peak_94.9', 'peak_95.0', 'peak_113.0']
        assert MS_docs_intensity[0][:5] == [313, 10964, 462, 322, 2100]


    def test_process_peaks(self):
        """ Basic test of process_peaks function."""
        peak_mass = [100, 150, 200, 300, 500, 510, 1100]
        peak_intensity = [700, 200, 100, 1000, 200, 5, 500]
        peaks = list(zip(peak_mass, peak_intensity))

        # Test peak processing function using spectrum with known outcome
        peaks_processed = process_peaks(peaks, min_frag=0, max_frag=1000,
                          min_intensity_perc=1,
                          exp_intensity_filter=None,
                          min_peaks=0)
        assert peaks_processed == [(200, 100), (150, 200), (500, 200), (100, 700), (300, 1000)], 'expected different peaks or differnt sorting'


    def test_exp_intensity_filter(self):
        """ Basic test of process_peaks function."""
        peak_mass = [100, 150, 200, 300, 500, 510, 1100] + [300] * 100 + [400] * 200
        peak_intensity = [700, 200, 100, 1000, 200, 5, 500] + [4] * 100 + [2] * 200
        peaks = list(zip(peak_mass, peak_intensity))

        # Test peak processing function using spectrum with known outcome
        peaks_processed = exponential_peak_filter(np.array(peaks), 0.5, 5, 10)
        assert peaks_processed.shape[0] == 6, 'expected different number of peaks after filtering'

        peaks_processed = exponential_peak_filter(np.array(peaks), 0.2, 5, 10)
        assert peaks_processed.shape[0] == 107, 'expected different number of peaks after filtering'


    def test_likely_inchi_match(self):
        """ Test with known Inchikeys """

        inchi_1 = 'InChI=1S/C6H8O6/c7-1-2(8)5-3(9)4(10)6(11)12-5/h2,5,7-10H,1H2/t2-,5+/m0/s1'
        inchi_2 = 'InChI=1S/C6H8O6/c7-1-2(8)5-3(9)4(10)6(11)12-5/h2,5,7-10H,1H2/txxx/zzzzz'
        inchi_3 = 'InChI=1S/C6H8O6/aaa/bbb/txxx/zzzzz'

        assert likely_inchi_match(inchi_1, inchi_2, min_agreement = 3), 'expected True match'
        assert likely_inchi_match(inchi_1, inchi_2, min_agreement = 6) == False, 'expected False match'
        assert likely_inchi_match(inchi_1, inchi_3, min_agreement = 3) == False, 'expected False match'


    def test_likely_inchikey_match(self):
        """ Test with known Inchikeys """

        inchikey_1 = 'BQJCRHHNABKAKU-KBQPJGBKSA-N'
        inchikey_2 = 'BQJCRHHNABKAKU-KBQPJXXXSA-N'
        inchikey_3 = 'BQJCRHHNABKXXX-KBQPJXXXSA-N'

        assert likely_inchikey_match(inchikey_1, inchikey_2, min_agreement = 1), 'expected True match'
        assert likely_inchikey_match(inchikey_1, inchikey_2, min_agreement = 2) == False, 'expected False match'
        assert likely_inchikey_match(inchikey_1, inchikey_3, min_agreement = 1) == False, 'expected False match'

if __name__ == '__main__':
    unittest.main()