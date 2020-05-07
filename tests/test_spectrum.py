import numpy
from matplotlib import pyplot as plt
from matchms import Spectrum


def test_spectrum_plot_with_histogram_unspecified():

    mz = numpy.array([10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110], dtype="float")
    intensities = numpy.array([1, 1, 5, 5, 5, 5, 7, 7, 7, 9, 9], dtype="float")
    spectrum = Spectrum(mz=mz, intensities=intensities)

    fig = spectrum.plot()

    assert fig is not None
    assert hasattr(fig, "axes")
    assert isinstance(fig.axes, list)
    assert len(fig.axes) == 1
    assert isinstance(fig.axes[0], plt.Axes)
    assert hasattr(fig.axes[0], "lines")
    assert isinstance(fig.axes[0].lines, list)
    assert len(fig.axes[0].lines) == 11
    assert isinstance(fig.axes[0].lines[0], plt.Line2D)
    assert hasattr(fig.axes[0].lines[0], "_x")


def test_spectrum_plot_with_histogram_false():

    mz = numpy.array([10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110], dtype="float")
    intensities = numpy.array([1, 1, 5, 5, 5, 5, 7, 7, 7, 9, 9], dtype="float")
    spectrum = Spectrum(mz=mz, intensities=intensities)

    fig = spectrum.plot(with_histogram=False)

    assert fig is not None
    assert hasattr(fig, "axes")
    assert isinstance(fig.axes, list)
    assert len(fig.axes) == 1
    assert isinstance(fig.axes[0], plt.Axes)
    assert hasattr(fig.axes[0], "lines")
    assert isinstance(fig.axes[0].lines, list)
    assert len(fig.axes[0].lines) == 11
    assert isinstance(fig.axes[0].lines[0], plt.Line2D)
    assert hasattr(fig.axes[0].lines[0], "_x")


def test_spectrum_plot_with_histogram_true():

    mz = numpy.array([10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110], dtype="float")
    intensities = numpy.array([1, 1, 5, 5, 5, 5, 7, 7, 7, 9, 9], dtype="float")
    spectrum = Spectrum(mz=mz, intensities=intensities)

    fig = spectrum.plot(with_histogram=True)

    assert fig is not None
    assert hasattr(fig, "axes")
    assert isinstance(fig.axes, list)
    assert len(fig.axes) == 2
    assert isinstance(fig.axes[0], plt.Axes)
    assert hasattr(fig.axes[0], "lines")
    assert isinstance(fig.axes[0].lines, list)
    assert len(fig.axes[0].lines) == 11
    assert isinstance(fig.axes[0].lines[0], plt.Line2D)
    assert hasattr(fig.axes[0].lines[0], "_x")


def test_spectrum_plot_with_histogram_true_and_intensity_limit():

    mz = numpy.array([10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110], dtype="float")
    intensities = numpy.array([1, 1, 5, 5, 5, 5, 7, 7, 7, 9, 9], dtype="float")
    spectrum = Spectrum(mz=mz, intensities=intensities)

    fig = spectrum.plot(with_histogram=True, intensity_to=10.0)

    assert fig is not None
    assert hasattr(fig, "axes")
    assert isinstance(fig.axes, list)
    assert len(fig.axes) == 2
    assert isinstance(fig.axes[0], plt.Axes)
    assert hasattr(fig.axes[0], "lines")
    assert isinstance(fig.axes[0].lines, list)
    assert len(fig.axes[0].lines) == 11
    assert isinstance(fig.axes[0].lines[0], plt.Line2D)
    assert hasattr(fig.axes[0].lines[0], "_x")


def test_spectrum_plot_with_histogram_true_and_expfit_true_and_intensity_limit():

    mz = numpy.array([10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110], dtype="float")
    intensities = numpy.array([1, 1, 5, 5, 5, 5, 7, 7, 7, 9, 9], dtype="float")
    spectrum = Spectrum(mz=mz, intensities=intensities)

    fig = spectrum.plot(with_histogram=True, with_expfit=True, intensity_to=10.0)

    assert fig is not None
    assert hasattr(fig, "axes")
    assert isinstance(fig.axes, list)
    assert len(fig.axes) == 2
    assert isinstance(fig.axes[0], plt.Axes)
    assert hasattr(fig.axes[0], "lines")
    assert isinstance(fig.axes[0].lines, list)
    assert len(fig.axes[0].lines) == 11
    assert isinstance(fig.axes[0].lines[0], plt.Line2D)
    assert hasattr(fig.axes[0].lines[0], "_x")
