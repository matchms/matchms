import numpy
import pytest
from matplotlib import pyplot as plt
from matchms import Spectrum
from .builder_Spectrum import SpectrumBuilder


def _assert_plots_ok(fig, n_plots):
    assert len(fig.axes) == n_plots
    assert fig is not None
    assert hasattr(fig, "axes")
    assert isinstance(fig.axes, list)
    assert isinstance(fig.axes[0], plt.Axes)
    assert hasattr(fig.axes[0], "lines")
    assert isinstance(fig.axes[0].get_lines(), list)  # .lines breakes for new matplotlib versions
    assert len(fig.axes[0].lines) == 11
    assert isinstance(fig.axes[0].lines[0], plt.Line2D)
    assert hasattr(fig.axes[0].lines[0], "_x")


def _create_test_spectrum():
    intensities = numpy.array([1, 1, 5, 5, 5, 5, 7, 7, 7, 9, 9], dtype="float")
    return _create_test_spectrum_with_intensities(intensities)


def _create_test_spectrum_with_intensities(intensities):
    mz = numpy.array([10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110], dtype="float")
    return Spectrum(mz=mz, intensities=intensities)


@pytest.fixture
def spectrum() -> Spectrum:
    mz = numpy.array([100.00003, 110.2, 200.581], dtype='float')
    intensities = numpy.array([0.51, 1.0, 0.011], dtype='float')
    metadata = {"pepmass": (444.0, 11), "charge": -1}
    builder = SpectrumBuilder().with_mz(mz).with_intensities(intensities).with_metadata(metadata)
    return builder.build()


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
    """Test if spectra with (slightly) different metadata are correctly compared."""
    metadata: dict = {
        "float_example": 400.768,
        "str_example": "whatever",
        "list_example": [3, 4, "abc"]
    }

    builder = SpectrumBuilder().with_mz(numpy.array([100.0, 101.0], dtype="float")).with_intensities(
        numpy.array([0.4, 0.5], dtype="float")).with_metadata(metadata)

    spectrum0 = builder.build()
    spectrum1 = builder.build()

    metadata2 = metadata.copy()
    metadata2["more_stuff"] = 15
    spectrum2 = builder.with_metadata(metadata2).build()

    metadata3 = metadata.copy()
    metadata3.update({"list_example": [3, 4, "abc", "extra"]})
    spectrum3 = builder.with_metadata(metadata3).build()

    assert spectrum0 == spectrum1, "Expected spectra to be equal"
    assert spectrum0 != spectrum2, "Expected spectra to not be equal"
    assert spectrum0 != spectrum3, "Expected spectra to not be equal"


def test_comparing_spectra_with_arrays():
    """Test if spectra can be compared that contain numpy.arrays in the metadata.
    (Failed in an earlier version)"""
    builder = SpectrumBuilder()
    spectrum0 = builder.build()

    fingerprint1 = numpy.array([0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0])
    spectrum1 = builder.with_metadata({"fingerprint": fingerprint1}).build()

    assert spectrum0 != spectrum1, "Expected spectra to not be equal"


def test_spectrum_hash(spectrum: Spectrum):
    assert hash(spectrum) == 1516465757675504211, "Expected different hash."
    assert spectrum.metadata_hash() == "92c0464af949ae56627f", \
        "Expected different metadata hash."
    assert spectrum.spectrum_hash() == "c79de5a8b333f780c206", \
        "Expected different spectrum hash."


def test_spectrum_hash_mz_sensitivity(spectrum: Spectrum):
    """Test is changes indeed lead to different hashes as expected."""
    mz2 = spectrum.peaks.mz.copy()
    mz2[0] += 0.00001
    spectrum2 = SpectrumBuilder().from_spectrum(spectrum).with_mz(mz2).build()

    assert hash(spectrum) != hash(spectrum2), "Expected hashes to be different."
    assert spectrum.metadata_hash() == spectrum2.metadata_hash(), \
        "Expected metadata hashes to be unchanged."
    assert spectrum.spectrum_hash() != spectrum2.spectrum_hash(), \
        "Expected spectrum hashes to be different."


def test_spectrum_hash_intensity_sensitivity(spectrum: Spectrum):
    """Test is changes indeed lead to different hashes as expected."""
    intensities2 = spectrum.peaks.intensities.copy()
    intensities2[0] += 0.01
    spectrum2 = SpectrumBuilder().from_spectrum(spectrum).with_intensities(intensities2).build()

    assert hash(spectrum) != hash(spectrum2), "Expected hashes to be different."
    assert spectrum.metadata_hash() == spectrum2.metadata_hash(), \
        "Expected metadata hashes to be unchanged."
    assert spectrum.spectrum_hash() != spectrum2.spectrum_hash(), \
        "Expected hashes to be different."


def test_spectrum_hash_metadata_sensitivity(spectrum: Spectrum):
    """Test is changes indeed lead to different hashes as expected."""
    spectrum2 = SpectrumBuilder().from_spectrum(spectrum).with_metadata(
        {"pepmass": (444.1, 11), "charge": -1}).build()

    assert hash(spectrum) != hash(spectrum2), "Expected hashes to be different."
    assert spectrum.metadata_hash() != spectrum2.metadata_hash(), \
        "Expected metadata hashes to be different."
    assert spectrum.spectrum_hash() == spectrum2.spectrum_hash(), \
        "Expected hashes to be unchanged."


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
