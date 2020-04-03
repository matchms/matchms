# test functions
import os
import numpy as np
import pytest
import unittest

from matchms.old.ms_functions import Spectrum, load_MGF_data
from matchms.old.ms_similarity_classical import cosine_score_greedy, cosine_score_matrix

# Use test data from following folder
PATH_TEST = os.path.dirname(os.path.abspath(__file__))
PATH_TESTDATA = os.path.join(PATH_TEST, 'testdata')

class ModelGenerationSuite(unittest.TestCase):
    """Basic test cases."""

    def test_cosine_score_greedy(self):
        """ Test of cosine and modified cosine score calculations."""
        peak_mass = [100, 150, 200, 300, 500, 510, 1100]
        peak_intensity = [700, 200, 100, 1000, 200, 5, 500]
        peaks = list(zip(peak_mass, peak_intensity))
        testspectrum_1 = Spectrum()
        testspectrum_1.peaks = peaks
        spec1 = np.array(testspectrum_1.peaks, dtype=float)

        # Build other test spectra
        peak_mass = [100, 140, 190, 300, 490, 510, 1090]
        peak_intensity = [700, 200, 100, 1000, 200, 5, 500]
        peaks = list(zip(peak_mass, peak_intensity))
        testspectrum_2 = Spectrum()
        testspectrum_2.peaks = peaks
        spec2 = np.array(testspectrum_2.peaks, dtype=float)

        peak_mass = [50, 100, 200, 299.5, 489.5, 510.5, 1040]
        peak_intensity = [700, 200, 100, 1000, 200, 5, 500]
        peaks = list(zip(peak_mass, peak_intensity))
        testspectrum_3 = Spectrum()
        testspectrum_3.peaks = peaks
        spec3 = np.array(testspectrum_3.peaks, dtype=float)

        score12, _ = cosine_score_greedy(spec1, spec2, mass_shift=None, tol=0.2)
        score13, _ = cosine_score_greedy(spec1, spec3, mass_shift=None, tol=0.2)
        assert score12 == pytest.approx(0.81421, 0.0001), 'expected different cosine score'
        assert score13 == pytest.approx(0.081966, 0.0001), 'expected different cosine score'

        score13_049, _ = cosine_score_greedy(spec1, spec3, mass_shift=None, tol=0.49)
        score13_050, _ = cosine_score_greedy(spec1, spec3, mass_shift=None, tol=0.5)
        assert score13_049 < score13_050, 'expected different cosine score'
        assert score13_050 == pytest.approx(0.6284203, 0.0001), 'expected different cosine score'

        # Test modified cosine (includes mass shift)
        score12_shift10, _ = cosine_score_greedy(spec1, spec2, mass_shift=10, tol=0.2)
        assert score12_shift10 == 1, 'expected different modified cosine score'

    def test_cosine_score_matrix(self):
        """ Test importing spectra, calculating cosine score matrix and modified
        cosine matrix. """
        # Import spectra
        test_mgf_file = os.path.join(PATH_TESTDATA, 'GNPS-COLLECTIONS-PESTICIDES-NEGATIVE.mgf')
        spectra, _, _, _, _ = load_MGF_data(test_mgf_file,
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

        # Calculate cosine score all-vs-all matrix and save results
        filename = os.path.join(PATH_TESTDATA, 'M_sim_modcos.npy')
        M_sim_modcos, M_matches_modcos = cosine_score_matrix(spectra,
                          tol = 0.005,
                          max_mz = 1000.0,
                          min_intens = 0,
                          mass_shifting = True,
                          method='greedy-numba',
                          num_workers = 4,
                          filename = filename,
                          safety_points = None)

        assert os.path.isfile(filename) == os.path.isfile(filename[:-4] + '_matches.npy') == True, 'Similarity matrix was not saved as expected.'
        assert M_sim_modcos.shape == M_sim_modcos.shape == (76,76), 'Different shape expected for modified cosine similarity matrices.'
        assert np.mean(M_sim_modcos.diagonal()) == 1.0, 'diagonal values of all-vs-all similarity matrix should be 1'
        assert np.max(M_sim_modcos) <= 1.000001, 'similarity matrix cannot contain values > 1 (except minor rounding error)'

        # Test loading already computed results
        filename = os.path.join(PATH_TESTDATA, 'M_sim_modcos.npy')
        M_sim_modcos, M_matches_modcos = cosine_score_matrix([],
                          tol = 0.005,
                          max_mz = 1000.0,
                          min_intens = 0,
                          mass_shifting = True,
                          method='greedy-numba',
                          num_workers = 4,
                          filename = filename,
                          safety_points = None)

        # Remove saved similarity matrix files
        os.remove(filename)
        os.remove(filename[:-4] + '_matches.npy')
        assert os.path.isfile(filename) == False


if __name__ == '__main__':
    unittest.main()
