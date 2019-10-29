# test functions

import numpy as np
from MS_functions import Spectrum


peak_mass = [100, 150, 200, 300, 500, 510, 1100]
peak_intensity = [700, 200, 100, 1000, 200, 5, 500]
peaks = list(zip(peak_mass, peak_intensity))

from MS_functions import process_peaks
def test_process_peaks():
    # Test peak processing function using spectrum with known outcome
    peaks_processed = process_peaks(peaks, min_frag=0, max_frag=1000, 
                      min_intensity_perc=1,
                      exp_intensity_filter=None,
                      min_peaks=0)
    
    assert peaks_processed == [(100, 700), (150, 200), (200, 100), (300, 1000), (500, 200)]



from MS_functions import fast_cosine_shift
def test_fast_cosine_shift():
    
    testspectrum_1 = Spectrum()
    testspectrum_1.peaks = peaks
    testspectrum_1.parent_mz = 1100
    
    # Build other test spectrum
    peak_mass = [100, 140, 190, 300, 490, 510, 1090]
    peak_intensity = [700, 200, 100, 1000, 200, 5, 500]
    peaks = list(zip(peak_mass, peak_intensity))
    
    testspectrum_2 = Spectrum()
    testspectrum_2.peaks = peaks
    testspectrum_2.parent_mz = 1090
    
    
    peak_mass = [50, 100, 200, 299.5, 489.5, 510.5, 1040]
    peak_intensity = [700, 200, 100, 1000, 200, 5, 500]
    peaks = list(zip(peak_mass, peak_intensity))
    
    testspectrum_3 = Spectrum()
    testspectrum_3.peaks = peaks
    testspectrum_3.parent_mz = 1050

    score12 = fast_cosine_shift(testspectrum_1, testspectrum_2, tol=0.2, min_match=2, min_intens = 0)
    score13 = fast_cosine_shift(testspectrum_1, testspectrum_3, tol=0.2, min_match=2, min_intens = 0)
    
    assert score12 == 1    
    assert score13 < 0.3 
    assert score13 > 0.29 
    
    score13_050 = fast_cosine_shift(testspectrum_1, testspectrum_3, tol=0.5, min_match=2, min_intens = 0)
    score13_051 = fast_cosine_shift(testspectrum_1, testspectrum_3, tol=0.51, min_match=2, min_intens = 0)
    
    assert score13_050 < score13_051
    assert score13_050 > 0.84

  