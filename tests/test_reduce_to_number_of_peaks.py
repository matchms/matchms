import numpy
from matchms import Spectrum
from matchms.filtering import reduce_to_number_of_peaks


def test_reduce_to_number_of_peaks_no_params():

    mz = numpy.array([10, 20, 30, 40], dtype="float")
    intensities = numpy.array([0, 1, 10, 100], dtype="float")
    spectrum_in = Spectrum(mz=mz, intensities=intensities)

    spectrum = reduce_to_number_of_peaks(spectrum_in)

    assert spectrum == spectrum_in, "Expected no changes."


def test_reduce_to_number_of_peaks_n_max_4():

    mz = numpy.array([10, 20, 30, 40, 50], dtype="float")
    intensities = numpy.array([1, 1, 10, 20, 100], dtype="float")
    spectrum_in = Spectrum(mz=mz, intensities=intensities)

    spectrum = reduce_to_number_of_peaks(spectrum_in, n_max=4)

    assert len(spectrum.peaks) == 4, "Expected that only 4 peaks remain."
    assert spectrum.peaks.mz.tolist() == [20., 30., 40., 50.], "Expected different peaks to remain."


def test_reduce_to_number_of_peaks_n_max_4_or_1_no_parent_mass():

    mz = numpy.array([10, 20, 30, 40], dtype="float")
    intensities = numpy.array([0, 1, 10, 100], dtype="float")
    spectrum_in = Spectrum(mz=mz, intensities=intensities)

    spectrum = reduce_to_number_of_peaks(spectrum_in, n_required=4, ratio_desired=0.1)

    assert spectrum == spectrum_in, "Expected the spectrum to qualify because the number of peaks (4) is equal to the" \
                                    "required number (4)."


def test_reduce_to_number_of_peaks_required_2_desired_2():

    mz = numpy.array([10, 20, 30, 40], dtype="float")
    intensities = numpy.array([0, 1, 10, 100], dtype="float")
    spectrum_in = Spectrum(mz=mz, intensities=intensities,
                           metadata={"parent_mass": 20})

    spectrum = reduce_to_number_of_peaks(spectrum_in, n_required=2, n_max=4, ratio_desired=0.1)

    assert len(spectrum.peaks) == 2, "Expected that only 2 peaks remain."
    assert spectrum.peaks.mz.tolist() == [30., 40.], "Expected different peaks to remain."


def test_reduce_to_number_of_peaks_required_2_desired_3():

    mz = numpy.array([10, 20, 30, 40], dtype="float")
    intensities = numpy.array([0, 1, 10, 100], dtype="float")
    spectrum_in = Spectrum(mz=mz, intensities=intensities,
                           metadata={"parent_mass": 20})

    spectrum = reduce_to_number_of_peaks(spectrum_in, n_required=3, n_max=4, ratio_desired=0.1)

    assert len(spectrum.peaks) == 3, "Expected that only 3 peaks remain."
    assert spectrum.peaks.mz.tolist() == [20., 30., 40.], "Expected different peaks to remain."


def test_reduce_to_number_of_peaks_desired_5_check_sorting():
    """Check if mz and intensities order is sorted correctly """
    mz = numpy.array([10, 20, 30, 40, 50, 60], dtype="float")
    intensities = numpy.array([5, 1, 4, 3, 100, 2], dtype="float")
    spectrum_in = Spectrum(mz=mz, intensities=intensities)

    spectrum = reduce_to_number_of_peaks(spectrum_in, n_max=5)

    assert spectrum.peaks.intensities.tolist() == [5., 4., 3., 100., 2.], "Expected different intensities."
    assert spectrum.peaks.mz.tolist() == [10., 30., 40., 50., 60.], "Expected different peaks to remain."
