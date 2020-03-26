# test functions

import numpy as np
from MS_functions import Spectrum
import pytest
import unittest

from MS_similarity_classical import cosine_score_greedy

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



if __name__ == '__main__':
    unittest.main()