import numpy
from matplotlib import pyplot as plt
from matchms import Spectrum


def _assert_plots_ok(fig, n_plots):
    assert len(fig.axes) == n_plots
    assert fig is not None
    assert hasattr(fig, "axes")
    assert isinstance(fig.axes, list)
    assert isinstance(fig.axes[0], plt.Axes)
    assert hasattr(fig.axes[0], "lines")
    assert isinstance(fig.axes[0].lines, list)
    assert len(fig.axes[0].lines) == 11
    assert isinstance(fig.axes[0].lines[0], plt.Line2D)
    assert hasattr(fig.axes[0].lines[0], "_x")


def _create_test_spectrum():
    intensities = numpy.array([1, 1, 5, 5, 5, 5, 7, 7, 7, 9, 9], dtype="float")
    return _create_test_spectrum_with_intensities(intensities)


def _create_test_spectrum_with_intensities(intensities):
    mz = numpy.array([10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110], dtype="float")
    return Spectrum(mz=mz, intensities=intensities)


def test_spectrum_getters_return_copies():
    """Test if getters return (deep)copies so that edits won't change the original entries."""
    spectrum = Spectrum(mz=numpy.array([100.0, 101.0], dtype="float"),
                        intensities=numpy.array([0.4, 0.5], dtype="float"),
                        metadata={"testdata": 1})
    # Get entries and modify
    testdata = spectrum.get("testdata")
    testdata += 1
    assert spectrum.get("testdata") == 1, "Expected different entry"
    peaks_mz = spectrum.peaks.mz
    peaks_mz += 100.0
    assert numpy.all(spectrum.peaks.mz == numpy.array([100.0, 101.0])), "Expected different peaks.mz"
    metadata = spectrum.metadata
    metadata["added_info"] = "this"
    assert spectrum.metadata == {'testdata': 1}, "Expected metadata to remain unchanged"


def test_comparing_spectra_with_metadata():
    """Test if spectra can be compared that contain numpy.arrays in the metadata.
    (Failed in an earlier version)"""
    spectrum0 = Spectrum(mz=numpy.array([100.0, 101.0], dtype="float"),
                         intensities=numpy.array([0.4, 0.5], dtype="float"),
                         metadata={"float_example": 400.768,
                                   "str_example": "whatever",
                                   "list_example": [3, 4, "abc"]})

    spectrum1 = Spectrum(mz=numpy.array([100.0, 101.0], dtype="float"),
                         intensities=numpy.array([0.4, 0.5], dtype="float"),
                         metadata={"float_example": 400.768,
                                   "str_example": "whatever",
                                   "list_example": [3, 4, "abc"]})

    spectrum2 = Spectrum(mz=numpy.array([100.0, 101.0], dtype="float"),
                         intensities=numpy.array([0.4, 0.5], dtype="float"),
                         metadata={"float_example": 400.768,
                                   "str_example": "whatever",
                                   "list_example": [3, 4, "abc"],
                                   "more_stuff": 15})

    spectrum3 = Spectrum(mz=numpy.array([100.0, 101.0], dtype="float"),
                         intensities=numpy.array([0.4, 0.5], dtype="float"),
                         metadata={"float_example": 400.768,
                                   "str_example": "whatever",
                                   "list_example": [3, 4, "abc", "extra"]})
    assert spectrum0 == spectrum1, "Expected spectra to be equal"
    assert spectrum0 != spectrum2, "Expected spectra to not be equal"
    assert spectrum0 != spectrum3, "Expected spectra to not be equal"


def test_comparing_spectra_with_arrays():
    """Test if spectra can be compared that contain numpy.arrays in the metadata.
    (Failed in an earlier version)"""
    spectrum0 = Spectrum(mz=numpy.array([], dtype="float"),
                         intensities=numpy.array([], dtype="float"),
                         metadata={})

    fingerprint1 = numpy.array([0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0])
    spectrum1 = Spectrum(mz=numpy.array([], dtype="float"),
                         intensities=numpy.array([], dtype="float"),
                         metadata={"fingerprint": fingerprint1})
    assert spectrum0 != spectrum1, "Expected spectra to not be equal"


def test_spectrum_plot_same_peak_height():
    intensities_with_zero_variance = numpy.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], dtype="float")
    spectrum = _create_test_spectrum_with_intensities(intensities_with_zero_variance)
    fig = spectrum.plot(with_histogram=True, intensity_to=10.0)
    _assert_plots_ok(fig, n_plots=2)


def test_spectrum_plot_with_histogram_false():
    spectrum = _create_test_spectrum()
    fig = spectrum.plot(with_histogram=False)
    _assert_plots_ok(fig, n_plots=1)


def test_spectrum_plot_with_histogram_true():
    spectrum = _create_test_spectrum()
    fig = spectrum.plot(with_histogram=True)
    _assert_plots_ok(fig, n_plots=2)


def test_spectrum_plot_with_histogram_true_and_intensity_limit():
    spectrum = _create_test_spectrum()
    fig = spectrum.plot(with_histogram=True, intensity_to=10.0)
    _assert_plots_ok(fig, n_plots=2)


def test_spectrum_plot_with_histogram_unspecified():
    spectrum = _create_test_spectrum()
    fig = spectrum.plot()
    _assert_plots_ok(fig, n_plots=1)
