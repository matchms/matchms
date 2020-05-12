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


def test_spectrum_plot_same_peak_height():
    intensities_with_zero_variance = numpy.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], dtype="float")
    spectrum = _create_test_spectrum_with_intensities(intensities_with_zero_variance)
    fig = spectrum.plot(with_histogram=True, with_expfit=True, intensity_to=10.0)
    _assert_plots_ok(fig, n_plots=2)


def test_spectrum_plot_with_histogram_false():
    spectrum = _create_test_spectrum()
    fig = spectrum.plot(with_histogram=False)
    _assert_plots_ok(fig, n_plots=1)


def test_spectrum_plot_with_histogram_true():
    spectrum = _create_test_spectrum()
    fig = spectrum.plot(with_histogram=True)
    _assert_plots_ok(fig, n_plots=2)


def test_spectrum_plot_with_histogram_true_and_expfit_true_and_intensity_limit():
    spectrum = _create_test_spectrum()
    fig = spectrum.plot(with_histogram=True, with_expfit=True, intensity_to=10.0)
    _assert_plots_ok(fig, n_plots=2)


def test_spectrum_plot_with_histogram_true_and_intensity_limit():
    spectrum = _create_test_spectrum()
    fig = spectrum.plot(with_histogram=True, intensity_to=10.0)
    _assert_plots_ok(fig, n_plots=2)


def test_spectrum_plot_with_histogram_unspecified():
    spectrum = _create_test_spectrum()
    fig = spectrum.plot()
    _assert_plots_ok(fig, n_plots=1)
