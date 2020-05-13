from matchms import Spectrum
from matchms.filtering import reduce_to_number_of_peaks
import numpy


def test_reduce_to_number_of_peaks_no_params():

    mz = numpy.array([10, 20, 30, 40], dtype="float")
    intensities = numpy.array([0, 1, 10, 100], dtype="float")
    spectrum_in = Spectrum(mz=mz, intensities=intensities)

    spectrum = reduce_to_number_of_peaks(spectrum_in)

    assert spectrum == spectrum, "Expected no changes."


def test_reduce_to_number_of_peaks_n_max_4():

    mz = numpy.array([10, 20, 30, 40, 50], dtype="float")
    intensities = numpy.array([1, 1, 10, 20, 100], dtype="float")
    spectrum_in = Spectrum(mz=mz, intensities=intensities)
    
    spectrum = reduce_to_number_of_peaks(spectrum_in, n_max=4)
    
    assert len(spectrum.peaks) == 4, "Expected that only 4 peaks remain."
    assert spectrum.peaks.mz.tolist() == [20., 30., 40., 50.], 'Expected different peaks to remain.'


def test_reduce_to_number_of_peaks_n_max_4_or_1_no_parent_mass():

    mz = numpy.array([10, 20, 30, 40], dtype="float")
    intensities = numpy.array([0, 1, 10, 100], dtype="float")
    spectrum_in = Spectrum(mz=mz, intensities=intensities)

    spectrum = reduce_to_number_of_peaks(spectrum_in, n_required=4, ratio_required=0.1)

    assert spectrum == spectrum_in, "Expected the spectrum to qualify because the number of peaks (4) is equal to the" \
                                    "required number (4)."


def test_reduce_to_number_of_peaks_required_2_ratio_2():

    mz = numpy.array([10, 20, 30, 40], dtype="float")
    intensities = numpy.array([0, 1, 10, 100], dtype="float")
    spectrum_in = Spectrum(mz=mz, intensities=intensities,
                           metadata={"parent_mass": 20})
    
    spectrum = reduce_to_number_of_peaks(spectrum_in, n_required=2, n_max=4, ratio_required=0.1)
    
    assert len(spectrum.peaks) == 2, "Expected that only 2 peaks remain."
    assert spectrum.peaks.mz.tolist() == [30., 40.], 'Expected different peaks to remain.'


def test_reduce_to_number_of_peaks_required_2_ratio_3():
    mz = numpy.array([10, 20, 30, 40], dtype="float")
    intensities = numpy.array([0, 1, 10, 100], dtype="float")
    spectrum_in = Spectrum(mz=mz, intensities=intensities,
                           metadata={"parent_mass": 20})
    
    spectrum = reduce_to_number_of_peaks(spectrum_in, n_required=3, n_max=4, ratio_required=0.1)
    
    assert len(spectrum.peaks) == 3, "Expected that only 3 peaks remain."
    assert spectrum.peaks.mz.tolist() == [20., 30., 40.], 'Expected different peaks to remain.'
