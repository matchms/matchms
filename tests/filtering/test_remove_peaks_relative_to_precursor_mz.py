import numpy as np
import pytest
from matchms.filtering import remove_peaks_relative_to_precursor_mz
from matchms.Fragments import Fragments
from tests.builder_Spectrum import SpectrumBuilder
from tests.run_spectrum_and_collection import run_filter_as_spectrum_or_collection


def _build_spectrum(precursor_mz, mzs, intensities, extra_meta=None):
    """Helper to create a Spectrum with metadata and peaks."""
    metadata = {"precursor_mz": precursor_mz}
    if extra_meta:
        metadata.update(extra_meta)

    spectrum = SpectrumBuilder().with_metadata(metadata).build()
    spectrum.peaks = Fragments(
        mz=np.asarray(mzs, dtype=float),
        intensities=np.asarray(intensities, dtype=float),
    )
    return spectrum


def _assert_peaks_equal(spectrum, expected_mz, expected_intensities):
    np.testing.assert_allclose(
        spectrum.peaks.mz,
        np.asarray(expected_mz, dtype=float),
        atol=1e-6,
    )
    np.testing.assert_allclose(
        spectrum.peaks.intensities,
        np.asarray(expected_intensities, dtype=float),
    )


def test_none_input_returns_none():
    assert remove_peaks_relative_to_precursor_mz(None) is None


@pytest.mark.parametrize("as_collection", [False, True], ids=["spectrum", "collection"])
def test_missing_precursor_mz_raises_value_error(as_collection):
    spectrum = SpectrumBuilder().with_metadata({}).build()
    spectrum.peaks = Fragments(mz=np.array([100.0]), intensities=np.array([1.0]))

    with pytest.raises(ValueError, match="Undefined 'precursor_mz'"):
        run_filter_as_spectrum_or_collection(
            remove_peaks_relative_to_precursor_mz,
            spectrum,
            as_collection,
        )


@pytest.mark.parametrize("as_collection", [False, True], ids=["spectrum", "collection"])
def test_non_scalar_precursor_mz_raises_value_error(as_collection):
    spectrum = SpectrumBuilder().with_metadata({"precursor_mz": [500.0]}).build()
    spectrum.peaks = Fragments(mz=np.array([100.0]), intensities=np.array([1.0]))

    with pytest.raises(ValueError, match="Expected 'precursor_mz' to be a scalar"):
        run_filter_as_spectrum_or_collection(
            remove_peaks_relative_to_precursor_mz,
            spectrum,
            as_collection,
        )


@pytest.mark.parametrize("as_collection", [False, True], ids=["spectrum", "collection"])
def test_non_scalar_offset_to_precursor_raises_value_error(as_collection):
    spectrum = _build_spectrum(
        precursor_mz=500.0,
        mzs=[100.0],
        intensities=[1.0],
    )

    with pytest.raises(ValueError, match="Expected 'offset_to_precursor' to be a scalar number"):
        run_filter_as_spectrum_or_collection(
            remove_peaks_relative_to_precursor_mz,
            spectrum,
            as_collection,
            offset_to_precursor=[-1.6],
        )


@pytest.mark.parametrize("as_collection", [False, True], ids=["spectrum", "collection"])
def test_empty_peaks_handled(as_collection):
    spectrum = _build_spectrum(precursor_mz=500.0, mzs=[], intensities=[])

    out = run_filter_as_spectrum_or_collection(
        remove_peaks_relative_to_precursor_mz,
        spectrum,
        as_collection,
    )

    assert out.peaks.mz.size == 0
    assert out.peaks.intensities.size == 0


@pytest.mark.parametrize("as_collection", [False, True], ids=["spectrum", "collection"])
def test_default_threshold_removes_greater_than_precursor_minus_1p6(as_collection):
    precursor = 500.0
    spectrum = _build_spectrum(
        precursor_mz=precursor,
        mzs=[100.0, 300.0, 498.3, 498.399, 498.5, 500.0, 501.0, 600.0],
        intensities=[1, 2, 3, 4, 5, 6, 7, 8],
    )

    out = run_filter_as_spectrum_or_collection(
        remove_peaks_relative_to_precursor_mz,
        spectrum,
        as_collection,
        clone=True,
    )

    # threshold = 498.4, keep peaks with m/z <= threshold
    _assert_peaks_equal(
        out,
        expected_mz=[100.0, 300.0, 498.3, 498.399],
        expected_intensities=[1, 2, 3, 4],
    )


@pytest.mark.parametrize("as_collection", [False, True], ids=["spectrum", "collection"])
def test_custom_relative_threshold(as_collection):
    precursor = 500.0
    spectrum = _build_spectrum(
        precursor_mz=precursor,
        mzs=[480.0, 489.9, 490.01, 490.1, 600.0],
        intensities=[10, 20, 30, 40, 50],
    )

    out = run_filter_as_spectrum_or_collection(
        remove_peaks_relative_to_precursor_mz,
        spectrum,
        as_collection,
        offset_to_precursor=-10.0,
    )

    _assert_peaks_equal(
        out,
        expected_mz=[480.0, 489.9],
        expected_intensities=[10, 20],
    )


def test_clone_true_does_not_modify_original():
    precursor = 200.0
    spectrum_in = _build_spectrum(
        precursor_mz=precursor,
        mzs=[150.0, 199.0, 199.0, 250.0],
        intensities=[1.0, 2.0, 3.0, 4.0],
    )

    out = remove_peaks_relative_to_precursor_mz(
        spectrum_in,
        offset_to_precursor=-1.0,
        clone=True,
    )

    assert out is not spectrum_in

    _assert_peaks_equal(
        spectrum_in,
        expected_mz=[150.0, 199.0, 199.0, 250.0],
        expected_intensities=[1.0, 2.0, 3.0, 4.0],
    )
    _assert_peaks_equal(
        out,
        expected_mz=[150.0, 199.0, 199.0],
        expected_intensities=[1.0, 2.0, 3.0],
    )


def test_clone_false_modifies_in_place():
    precursor = 300.0
    spectrum_in = _build_spectrum(
        precursor_mz=precursor,
        mzs=[100.0, 200.0, 298.5, 310.0],
        intensities=[1.0, 2.0, 3.0, 4.0],
    )

    out = remove_peaks_relative_to_precursor_mz(
        spectrum_in,
        offset_to_precursor=-1.0,
        clone=False,
    )

    assert out is spectrum_in

    _assert_peaks_equal(
        spectrum_in,
        expected_mz=[100.0, 200.0, 298.5],
        expected_intensities=[1.0, 2.0, 3.0],
    )


@pytest.mark.parametrize("as_collection", [False, True], ids=["spectrum", "collection"])
def test_intensity_array_kept_in_lockstep_with_mz(as_collection):
    spectrum = _build_spectrum(
        precursor_mz=250.0,
        mzs=[100.0, 200.0, 248.4, 248.499, 400.0],
        intensities=[0.1, 0.2, 0.3, 0.4, 0.5],
    )

    out = run_filter_as_spectrum_or_collection(
        remove_peaks_relative_to_precursor_mz,
        spectrum,
        as_collection,
        offset_to_precursor=-1.5,
    )

    _assert_peaks_equal(
        out,
        expected_mz=[100.0, 200.0, 248.4, 248.499],
        expected_intensities=[0.1, 0.2, 0.3, 0.4],
    )
