import numpy as np
from matplotlib import pyplot as plt
from matchms import Spectrum
from matchms.plotting import plot_spectra_array
from matchms.plotting import plot_spectrum


def _assert_fig_ok(fig, n_plots, dpi, height):
    assert len(fig.axes) == n_plots
    assert fig.get_dpi() == dpi
    assert fig.get_figheight() == height


def _assert_ax_ok(ax, n_lines, ylim, xlabel, ylabel):
    assert isinstance(ax, plt.Axes)
    assert hasattr(ax, "lines")
    assert isinstance(ax.get_lines(), list)
    assert len(ax.lines) == n_lines
    if n_lines > 0:
        assert isinstance(ax.lines[0], plt.Line2D)
    assert ax.get_ylim() == (0.0, ylim)
    assert ax.get_xlabel() == xlabel
    assert ax.get_ylabel() == ylabel


def test_plot_spectrum_default():
    n_peaks = 100
    mz = np.random.randint(0, 1000, n_peaks).astype("float")
    mz.sort()

    spectrum = Spectrum(mz=mz,
                        intensities=np.random.random(n_peaks),
                        metadata={"compound_name": "SuperSubstance"})

    _, ax = plt.subplots()
    ax = plot_spectrum(spectrum)
    _assert_ax_ok(ax, n_lines=n_peaks, ylim=1.1,
                  xlabel="m/z", ylabel="Intensity")


def test_plot_spectrum_peak_comments():
    n_peaks = 51
    mz = np.linspace(0, 500, n_peaks).astype("float")
    mz.sort()

    spectrum = Spectrum(mz=mz,
                        intensities=np.random.random(n_peaks),
                        metadata={"compound_name": "SuperSubstance",
                                  "peak_comments": {100: "known peak"}})

    _, ax = plt.subplots()
    ax = plot_spectrum(spectrum, annotate_ions=True)
    _assert_ax_ok(ax, n_lines=n_peaks, ylim=1.25,
                  xlabel="m/z", ylabel="Intensity")


def test_plot_spectra_array_default():
    n = 9
    n_peaks = 50

    spectrums = []
    for i in range(n):
        mz = np.random.randint(0, 1000, n_peaks).astype("float")
        mz.sort()

        spectrum = Spectrum(mz=mz,
                            intensities=np.random.random(n_peaks),
                            metadata={"compound_name": f"Spectrum name {i}"})
        spectrums.append(spectrum)

    fig, axes = plot_spectra_array(spectrums)

    assert axes.shape == (5, 2)
    _assert_fig_ok(fig, n_plots=10, dpi=200, height=15)
    _assert_ax_ok(axes[0, 0], n_lines=n_peaks, ylim=1.1,
                  xlabel="m/z", ylabel="Intensity")
    _assert_ax_ok(axes[4, 0], n_lines=n_peaks, ylim=1.1,
                  xlabel="m/z", ylabel="Intensity")
    # Last subplot should be empty:
    _assert_ax_ok(axes[-1, -1], n_lines=0, ylim=1,
                  xlabel="", ylabel="")


def test_plot_spectra_array():
    n = 10
    n_peaks = 50

    spectrums = []
    for i in range(n):
        mz = np.random.randint(0, 1000, n_peaks).astype("float")
        mz.sort()

        spectrum = Spectrum(mz=mz,
                            intensities=np.random.random(n_peaks),
                            metadata={"compound_name": f"Spectrum name {i}"})
        spectrums.append(spectrum)

    fig, axes = plot_spectra_array(spectrums,
                                   n_cols=4,
                                   peak_color="darkblue",
                                   dpi=150)

    assert axes.shape == (3, 4)
    _assert_fig_ok(fig, n_plots=12, dpi=150, height=9)
    _assert_ax_ok(axes[0, 0], n_lines=n_peaks, ylim=1.1,
                  xlabel="m/z", ylabel="Intensity")
    _assert_ax_ok(axes[1, 3], n_lines=n_peaks, ylim=1.1,
                  xlabel="m/z", ylabel="Intensity")
    # Last subplot should be empty:
    _assert_ax_ok(axes[-1, -1], n_lines=0, ylim=1,
                  xlabel="", ylabel="")
