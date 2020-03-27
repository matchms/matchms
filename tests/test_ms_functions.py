# test functions

from matchms.ms_functions import process_peaks

peak_mass = [100, 150, 200, 300, 500, 510, 1100]
peak_intensity = [700, 200, 100, 1000, 200, 5, 500]
peaks = list(zip(peak_mass, peak_intensity))


def test_process_peaks():
    # Test peak processing function using spectrum with known outcome

    peaks_processed = process_peaks(peaks, min_frag=0, max_frag=1000, min_intensity_perc=1,
                                    exp_intensity_filter=None, min_peaks=0)

    # assert peaks_processed == [(100, 700), (150, 200), (200, 100), (300, 1000), (500, 200)]
