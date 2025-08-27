import numpy as np
import pytest
from matchms.filtering import remove_precursor_mz
from matchms.Fragments import Fragments
from ..builder_Spectrum import SpectrumBuilder


def _build_spectrum(precursor_mz, mzs, intensities, extra_meta=None):
    """Helper to create a Spectrum with metadata and peaks."""
    md = {"precursor_mz": precursor_mz}
    if extra_meta:
        md.update(extra_meta)
    s = SpectrumBuilder().with_metadata(md).build()
    s.peaks = Fragments(mz=np.asarray(mzs, dtype=float),
                        intensities=np.asarray(intensities, dtype=float))
    return s


def test_none_input_returns_none():
    assert remove_precursor_mz(None) is None


def test_missing_precursor_mz_raises_assertion():
    s = SpectrumBuilder().with_metadata({}).build()
    s.peaks = Fragments(mz=np.array([100.0]), intensities=np.array([1.0]))
    with pytest.raises(AssertionError, match="Precursor mz absent"):
        remove_precursor_mz(s)


def test_non_scalar_precursor_mz_raises_assertion():
    s = SpectrumBuilder().with_metadata({"precursor_mz": [500.0]}).build()
    s.peaks = Fragments(mz=np.array([100.0]), intensities=np.array([1.0]))
    with pytest.raises(AssertionError, match="Expected 'precursor_mz' to be a scalar number"):
        remove_precursor_mz(s)


def test_negative_tolerance_raises_assertion():
    s = _build_spectrum(precursor_mz=400.0, mzs=[390.0, 400.0, 410.0], intensities=[1, 2, 3])
    with pytest.raises(AssertionError, match="mz_tolerance must be a positive scalar"):
        remove_precursor_mz(s, mz_tolerance=-0.1)


def test_empty_peaks_kept_safely():
    s = _build_spectrum(precursor_mz=500.0, mzs=[], intensities=[])
    out = remove_precursor_mz(s)
    assert out.peaks.mz.size == 0
    assert out.peaks.intensities.size == 0


def test_custom_tolerance_behavior():
    """
    With tolerance = 5, remove peaks within ±5 Da (inclusive).
    """
    precursor = 300.0
    s = _build_spectrum(
        precursor_mz=precursor,
        mzs=[294.9, 295.0, 295.1, 300.0, 304.9, 305.0, 305.1, 320.0],
        intensities=[1, 2, 3, 4, 5, 6, 7, 8],
    )
    out = remove_precursor_mz(s, mz_tolerance=5.0)
    # Removed: 295.0..305.0 inclusive plus 300.0 itself
    assert np.allclose(out.peaks.mz, [294.9, 305.1, 320.0])
    assert np.allclose(out.peaks.intensities, [1, 7, 8])


def test_clone_true_returns_new_object_and_keeps_original_intact():
    precursor = 250.0
    s_in = _build_spectrum(
        precursor_mz=precursor,
        mzs=[230.0, 233.0, 240.0, 250.0, 266.9, 267.0, 270.0],
        intensities=[1, 2, 3, 4, 5, 6, 7],
    )
    out = remove_precursor_mz(s_in, mz_tolerance=17.0, clone=True)

    # New object when cloned
    assert out is not s_in

    # Original unchanged
    assert np.allclose(s_in.peaks.mz, [230.0, 233.0, 240.0, 250.0, 266.9, 267.0, 270.0])
    assert np.allclose(s_in.peaks.intensities, [1, 2, 3, 4, 5, 6, 7])

    # Filtered: remove within [233.0, 267.0] inclusive
    assert np.allclose(out.peaks.mz, [230.0, 270.0])
    assert np.allclose(out.peaks.intensities, [1, 7])


def test_clone_false_modifies_in_place():
    precursor = 100.0
    s_in = _build_spectrum(
        precursor_mz=precursor,
        mzs=[80.0, 83.0, 117.0, 118.0],
        intensities=[10.0, 20.0, 30.0, 40.0],
    )
    out = remove_precursor_mz(s_in, mz_tolerance=17.0, clone=False)

    # Same object when not cloned
    assert out is s_in

    # Remove peaks with |mz-100| <= 17 → 83.0 and 117.0 removed; 118.0 kept
    assert np.allclose(s_in.peaks.mz, [80.0, 118.0])
    assert np.allclose(s_in.peaks.intensities, [10.0, 40.0])


def test_intensities_remain_aligned_with_mz_after_filtering():
    s = _build_spectrum(
        precursor_mz=200.0,
        mzs=[182.9, 183.0, 183.1, 200.0, 216.9, 217.0, 217.1],
        intensities=[0.1, 0.2, 0.3, 9.9, 1.1, 1.2, 1.3],
    )
    out = remove_precursor_mz(s, mz_tolerance=17.0)
    # Remove 183.0..217.0 inclusive and 200.0; keep 182.9 and 217.1
    assert np.allclose(out.peaks.mz, [182.9, 217.1])
    assert np.allclose(out.peaks.intensities, [0.1, 1.3])
