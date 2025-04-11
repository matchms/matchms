import numpy as np
import pytest
from matplotlib import pyplot as plt
from matchms import Spectrum
from matchms.plotting import plot_spectra_array, plot_spectra_mirror, plot_spectrum


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


def test_plot_mirror_plot():
    # Create two random spectra
    spec_a = Spectrum(mz=np.array([100, 200, 300, 400.2]), intensities=np.array([0.5, 0.3, 0.1, 0.05]))
    spec_b = Spectrum(mz=np.array([10.2, 20.2, 30.2, 40.2, 78.2]), intensities=np.array([0.5, 0.3, 0.1, 0.05, 3.1]))
    max_mz = 250
    min_mz = 60
    # Boundaries were not applied in the previous coce
    ax = plot_spectra_mirror(spec_a, spec_b, max_mz=max_mz, min_mz=min_mz)
    assert ax.get_xlim() == (min_mz, max_mz)


def test_plot_mirror_colors():
    # Create two random spectra
    spec_a = Spectrum(mz=np.array([100.0, 200.0]), intensities=np.array([0.5, 0.3]))
    spec_b = Spectrum(mz=np.array([10.2, 20.2, 30.2]), intensities=np.array([0.5, 0.3, 0.1]))

    # Set specific colors to the top and bottom spectra
    ax = plot_spectra_mirror(spec_a, spec_b, color_top="red", color_bottom="blue")
    # Get the color of the lines
    all_colors = [line.get_color() for line in ax.get_lines()]
    assert "red" in all_colors
    assert "blue" in all_colors

    # Test that it gives proper error if the wrong color argument is given
    with pytest.raises(ValueError, match="'peak_color' should not be set for `plot_spectra_mirror`. "):
        plot_spectra_mirror(spec_a, spec_b, peak_color="green")


def test_plot_spectrum_default():
    n_peaks = 100
    mz = np.random.randint(0, 1000, n_peaks).astype("float")
    mz.sort()

    spectrum = Spectrum(mz=mz, intensities=np.random.random(n_peaks), metadata={"compound_name": "SuperSubstance"})

    _, ax = plt.subplots()
    ax = plot_spectrum(spectrum)
    _assert_ax_ok(ax, n_lines=n_peaks, ylim=1.1, xlabel="m/z", ylabel="Intensity")


def test_plot_spectrum_peak_comments():
    n_peaks = 51
    mz = np.linspace(0, 500, n_peaks).astype("float")
    mz.sort()

    spectrum = Spectrum(mz=mz, intensities=np.random.random(n_peaks), metadata={"compound_name": "SuperSubstance", "peak_comments": {100: "known peak"}})

    _, ax = plt.subplots()
    ax = plot_spectrum(spectrum, annotate_ions=True)
    _assert_ax_ok(ax, n_lines=n_peaks, ylim=1.25, xlabel="m/z", ylabel="Intensity")


def test_plot_single_spectrum_plot_spectra_array():
    """Test that inputing a single spectrum doesn't break plot_spectra_array"""
    n_peaks = 50

    mz = np.random.randint(0, 1000, n_peaks).astype("float")
    mz.sort()

    spectrum = Spectrum(mz=mz, intensities=np.random.random(n_peaks), metadata={"compound_name": "Spectrum name"})
    plot_spectra_array([spectrum])


def test_plot_spectra_array_default():
    n = 9
    n_peaks = 50

    spectra = []
    for i in range(n):
        mz = np.random.randint(0, 1000, n_peaks).astype("float")
        mz.sort()

        spectrum = Spectrum(mz=mz, intensities=np.random.random(n_peaks), metadata={"compound_name": f"Spectrum name {i}"})
        spectra.append(spectrum)

    fig, axes = plot_spectra_array(spectra)

    assert axes.shape == (5, 2)
    _assert_fig_ok(fig, n_plots=10, dpi=200, height=15)
    _assert_ax_ok(axes[0, 0], n_lines=n_peaks, ylim=1.1, xlabel="m/z", ylabel="Intensity")
    _assert_ax_ok(axes[4, 0], n_lines=n_peaks, ylim=1.1, xlabel="m/z", ylabel="Intensity")
    # Last subplot should be empty:
    _assert_ax_ok(axes[-1, -1], n_lines=0, ylim=1, xlabel="", ylabel="")


def test_plot_spectra_array():
    n = 10
    n_peaks = 50

    spectra = []
    for i in range(n):
        mz = np.random.randint(0, 1000, n_peaks).astype("float")
        mz.sort()

        spectrum = Spectrum(mz=mz, intensities=np.random.random(n_peaks), metadata={"compound_name": f"Spectrum name {i}"})
        spectra.append(spectrum)

    fig, axes = plot_spectra_array(spectra, n_cols=4, peak_color="darkblue", dpi=150)

    assert axes.shape == (3, 4)
    _assert_fig_ok(fig, n_plots=12, dpi=150, height=9)
    _assert_ax_ok(axes[0, 0], n_lines=n_peaks, ylim=1.1, xlabel="m/z", ylabel="Intensity")
    _assert_ax_ok(axes[1, 3], n_lines=n_peaks, ylim=1.1, xlabel="m/z", ylabel="Intensity")
    # Last subplot should be empty:
    _assert_ax_ok(axes[-1, -1], n_lines=0, ylim=1, xlabel="", ylabel="")
