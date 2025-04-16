import numpy as np
import pytest
from matplotlib import pyplot as plt
from matchms import Spectrum
from .builder_Spectrum import SpectrumBuilder


def _assert_plots_ok(fig, n_plots, n_lines):
    assert len(fig.axes) == n_plots
    assert fig is not None
    assert hasattr(fig, "axes")
    assert isinstance(fig.axes, list)
    assert isinstance(fig.axes[0], plt.Axes)
    assert hasattr(fig.axes[0], "lines")
    assert isinstance(fig.axes[0].get_lines(), list)  # .lines breakes for new matplotlib versions
    assert len(fig.axes[0].lines) == n_lines
    assert isinstance(fig.axes[0].lines[0], plt.Line2D)
    assert hasattr(fig.axes[0].lines[0], "_x")


def _create_test_spectrum():
    intensities = np.array([1, 1, 5, 5, 5, 5, 7, 7, 7, 9, 9], dtype="float")
    return _create_test_spectrum_with_intensities(intensities)


def _create_test_spectrum_with_intensities(intensities):
    mz = np.array([10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110], dtype="float")
    return Spectrum(mz=mz, intensities=intensities)


@pytest.fixture
def spectrum() -> Spectrum:
    mz = np.array([100.00003, 110.2, 200.581], dtype="float")
    intensities = np.array([0.51, 1.0, 0.011], dtype="float")
    metadata = {"precursor_mz": 444.0, "charge": -1}
    builder = SpectrumBuilder().with_mz(mz).with_intensities(intensities).with_metadata(metadata)
    return builder.build()


def test_spectrum_getters_return_copies():
    """Test if getters return (deep)copies so that edits won't change the original entries."""
    spectrum = Spectrum(
        mz=np.array([100.0, 101.0], dtype="float"),
        intensities=np.array([0.4, 0.5], dtype="float"),
        metadata={"testdata": 1},
        metadata_harmonization=False,
    )
    # Get entries and modify
    testdata = spectrum.get("testdata")
    testdata += 1
    assert spectrum.get("testdata") == 1, "Expected different entry"
    peaks_mz = spectrum.peaks.mz
    peaks_mz += 100.0
    assert np.all(spectrum.peaks.mz == np.array([100.0, 101.0])), "Expected different peaks.mz"
    metadata = spectrum.metadata
    metadata["added_info"] = "this"
    assert spectrum.metadata == {"testdata": 1}, "Expected metadata to remain unchanged"


def test_spectrum_getters(spectrum: Spectrum):
    assert np.all(spectrum.mz == spectrum.peaks.mz)
    assert np.all(spectrum.intensities == spectrum.peaks.intensities)
    # Test if true copy
    mz = spectrum.mz
    mz[0] = 1111
    assert np.allclose(spectrum.peaks.mz[0], 100.00003)


@pytest.mark.parametrize(
    "input_dict, expected_dict",
    [
        [
            {"precursor mass": 400.768, "Some Key": "Whatever.", "NEW\tSTUFF": "XYZ"},
            {"precursor_mz": 400.768, "some_key": "Whatever.", "new_stuff": "XYZ"},
        ],
        [{"Name": "Whatever123", "ION MODE": "XYZ"}, {"compound_name": "Whatever123", "ionmode": "XYZ"}],
        [{"ri": "200"}, {"retention_index": "200"}],
    ],
)
def test_spectrum_metadata_harmonization(input_dict, expected_dict):
    builder = SpectrumBuilder().with_metadata(input_dict, metadata_harmonization=False)
    spectrum = builder.build()
    assert spectrum.metadata == expected_dict, "Expected different metadata dict"


def test_comparing_spectra_with_metadata():
    """Test if spectra with (slightly) different metadata are correctly compared."""
    metadata: dict = {
        "float_example": 400.768,
        "str_example": "whatever",
        "list_example": [3, 4, "abc"],
        "fingerprint": np.array([0.1, 5, 1.1]),
    }

    builder = (
        SpectrumBuilder()
        .with_mz(np.array([100.0, 101.0], dtype="float"))
        .with_intensities(np.array([0.4, 0.5], dtype="float"))
        .with_metadata(metadata)
    )

    spectrum0 = builder.build()
    spectrum1 = builder.build()

    metadata2 = metadata.copy()
    metadata2["more_stuff"] = 15
    spectrum2 = builder.with_metadata(metadata2).build()

    metadata3 = metadata.copy()
    metadata3.update({"list_example": [3, 4, "abc", "extra"]})
    spectrum3 = builder.with_metadata(metadata3).build()

    metadata4 = metadata.copy()
    metadata4["fingerprint"] = np.array([0.1, 5, 1.10001])
    spectrum4 = builder.with_metadata(metadata3).build()

    assert spectrum0 == spectrum1, "Expected spectra to be equal"
    assert spectrum0 != spectrum2, "Expected spectra to not be equal"
    assert spectrum0 != spectrum3, "Expected spectra to not be equal"
    assert spectrum0 != spectrum4, "Expected spectra to not be equal"


def test_comparing_spectra_with_arrays():
    """Test if spectra can be compared that contain numpy arrays in the metadata.
    (Failed in an earlier version)"""
    builder = SpectrumBuilder()
    spectrum0 = builder.build()

    fingerprint1 = np.array([0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0])
    spectrum1 = builder.with_metadata({"fingerprint": fingerprint1}).build()

    assert spectrum0 != spectrum1, "Expected spectra to not be equal"


def test_spectrum_to_dict(spectrum: Spectrum):
    """Test if export to Python dictionary works as intended"""
    spectrum_dict = spectrum.to_dict()
    expected_dict = {
        "charge": -1,
        "peaks_json": [[100.00003, 0.51], [110.2, 1.0], [200.581, 0.011]],
        "precursor_mz": 444.0,
    }
    assert spectrum_dict == expected_dict


def test_spectrum_to_dict_matchms_style(spectrum: Spectrum):
    """Test if export to Python dictionary works as intended"""
    spectrum_dict = spectrum.to_dict(export_style="nist")
    expected_dict = {
        "Charge": -1,
        "peaks_json": [[100.00003, 0.51], [110.2, 1.0], [200.581, 0.011]],
        "PrecursorMZ": 444.0,
    }
    assert spectrum_dict == expected_dict


def test_spectrum_repr_and_str_method():
    # Test the __repr__ and __str__ methods.
    spectrum = Spectrum(mz=np.array([1.0, 2.0, 3.0]), intensities=np.array([0, 0.5, 1.0]))
    assert repr(spectrum) == str(spectrum) == "Spectrum(precursor m/z=0.00, 3 fragments between 1.0 and 3.0)"

    # Test if spectra with empty peak arrays are handled correctly:
    spectrum = Spectrum(mz=np.array([]), intensities=np.array([]))
    assert repr(spectrum) == str(spectrum) == "Spectrum(precursor m/z=0.00, no fragments)"


def test_spectrum_hash(spectrum: Spectrum):
    assert hash(spectrum) == 382278160858921722, "Expected different hash."
    assert spectrum.metadata_hash() == "78c223faa157cc130390", "Expected different metadata hash."
    assert spectrum.spectrum_hash() == "c79de5a8b333f780c206", "Expected different spectrum hash."


def test_spectrum_hash_mz_sensitivity(spectrum: Spectrum):
    """Test is changes indeed lead to different hashes as expected."""
    mz2 = spectrum.peaks.mz.copy()
    mz2[0] += 0.00001
    spectrum2 = SpectrumBuilder().from_spectrum(spectrum).with_mz(mz2).build()

    assert hash(spectrum) != hash(spectrum2), "Expected hashes to be different."
    assert spectrum.metadata_hash() == spectrum2.metadata_hash(), "Expected metadata hashes to be unchanged."
    assert spectrum.spectrum_hash() != spectrum2.spectrum_hash(), "Expected spectrum hashes to be different."


def test_spectrum_hash_intensity_sensitivity(spectrum: Spectrum):
    """Test is changes indeed lead to different hashes as expected."""
    intensities2 = spectrum.peaks.intensities.copy()
    intensities2[0] += 0.01
    spectrum2 = SpectrumBuilder().from_spectrum(spectrum).with_intensities(intensities2).build()

    assert hash(spectrum) != hash(spectrum2), "Expected hashes to be different."
    assert spectrum.metadata_hash() == spectrum2.metadata_hash(), "Expected metadata hashes to be unchanged."
    assert spectrum.spectrum_hash() != spectrum2.spectrum_hash(), "Expected hashes to be different."


def test_spectrum_hash_metadata_sensitivity(spectrum: Spectrum):
    """Test is changes indeed lead to different hashes as expected."""
    spectrum2 = SpectrumBuilder().from_spectrum(spectrum).with_metadata({"precursor_mz": 444.1, "charge": -1}).build()

    assert hash(spectrum) != hash(spectrum2), "Expected hashes to be different."
    assert spectrum.metadata_hash() != spectrum2.metadata_hash(), "Expected metadata hashes to be different."
    assert spectrum.spectrum_hash() == spectrum2.spectrum_hash(), "Expected hashes to be unchanged."


@pytest.mark.parametrize("default_filtering", [True, False])
def test_spectrum_clone(spectrum, default_filtering):
    spectrum = (
        SpectrumBuilder()
        .from_spectrum(spectrum)
        .with_metadata({"precursor_mz": 444.1, "TEST FIELD": "Some Text"}, metadata_harmonization=default_filtering)
        .build()
    )
    spectrum_clone = spectrum.clone()

    assert spectrum_clone == spectrum.clone(), "Spectra should be equal"

    # Check if no shallow copy was made
    spectrum_clone.metadata = {"precursor_mz": 424.1, "TEST FIELD": "Some Text"}
    assert spectrum_clone != spectrum.clone(), "Only cloned spectrum should have changed"


@pytest.mark.parametrize(
    "input_dict, default_filtering, expected",
    [
        [{}, True, {}],
        [{"precursor_mz": 101.01}, True, {"precursor_mz": 101.01}],
        [{"precursormz": 101.01}, True, {"precursor_mz": 101.01}],
        [{"precursormz": 101.01}, False, {"precursor_mz": 101.01}],
        [{"ExactMass": 105.055}, True, {"parent_mass": 105.055}],
        [{"ExactMass": 105.055, "parent_mass": 107.077}, True, {"parent_mass": 107.077}],
        [{"charge": "2+"}, True, {"charge": 2}],
        [{"charge": -1}, True, {"charge": -1}],
        [{"charge": [-1, 0]}, True, {"charge": -1}],
        [{"ionmode": "Negative"}, True, {"ionmode": "negative"}],
        [{"ri": "200"}, True, {"retention_index": 200.0}],
        [{"rt": "200"}, True, {"retention_time": 200.0}],
    ],
)
def test_metadata_default_filtering(spectrum, input_dict, default_filtering, expected):
    spectrum = (
        SpectrumBuilder()
        .from_spectrum(spectrum)
        .with_metadata(input_dict, metadata_harmonization=default_filtering)
        .build()
    )
    assert spectrum.metadata == expected, "Expected different _metadata dictionary."


@pytest.mark.parametrize(
    "mz, loss_mz_to, expected_mz, expected_intensities",
    [
        [
            np.array([100, 150, 200, 300], dtype="float"),
            1000,
            np.array([145, 245, 295, 345], "float"),
            np.array([1000, 100, 200, 700], "float"),
        ],
        [
            np.array([100, 150, 200, 450], dtype="float"),
            1000,
            np.array([245, 295, 345], "float"),
            np.array([100, 200, 700], "float"),
        ],
        [
            np.array([100, 150, 200, 300], dtype="float"),
            250,
            np.array([145, 245], "float"),
            np.array([1000, 100], "float"),
        ],
    ],
)
def test_compute_losses_parameterized(mz, loss_mz_to, expected_mz, expected_intensities):
    intensities = np.array([700, 200, 100, 1000], "float")
    metadata = {"precursor_mz": 445.0}
    spectrum = SpectrumBuilder().with_mz(mz).with_intensities(intensities).with_metadata(metadata).build()

    losses = spectrum.compute_losses(loss_mz_to=loss_mz_to)

    assert np.allclose(losses.mz, expected_mz), "Expected different loss m/z."
    assert np.allclose(losses.intensities, expected_intensities), "Expected different intensities."


def test_losses_property():
    mz = np.array([100, 150, 200, 300.0])
    intensities = np.array([700, 200, 100, 1000], "float")
    metadata = {"precursor_mz": 445.0}
    spectrum = SpectrumBuilder().with_mz(mz).with_intensities(intensities).with_metadata(metadata).build()

    # Check if class property "losses" works
    losses_default = spectrum.compute_losses()
    assert np.allclose(spectrum.losses.mz, losses_default.mz)
    assert np.allclose(spectrum.losses.intensities, losses_default.intensities)


def test_spectrum_plot_same_peak_height():
    intensities_with_zero_variance = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], dtype="float")
    spectrum = _create_test_spectrum_with_intensities(intensities_with_zero_variance)
    fig, _ = spectrum.plot()
    _assert_plots_ok(fig, n_plots=1, n_lines=11)


def test_spectrum_plot():
    spectrum = _create_test_spectrum()
    fig, _ = spectrum.plot()
    _assert_plots_ok(fig, n_plots=1, n_lines=11)


def test_spectrum_mirror_plot():
    spectrum = _create_test_spectrum()
    fig, _ = spectrum.plot_against(spectrum)
    _assert_plots_ok(fig, n_plots=1, n_lines=23)
