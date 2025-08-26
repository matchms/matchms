import numpy as np
import pytest
from matchms.filtering import remove_peaks_relative_to_precursor_mz
from matchms.Fragments import Fragments
from ..builder_Spectrum import SpectrumBuilder


def _build_spectrum(precursor_mz, mzs, intensities, clone=True, extra_meta=None):
    """Helper to create a Spectrum with metadata and peaks."""
    md = {"precursor_mz": precursor_mz}
    if extra_meta:
        md.update(extra_meta)
    s = SpectrumBuilder().with_metadata(md).build()
    s.peaks = Fragments(mz=np.asarray(mzs, dtype=float),
                        intensities=np.asarray(intensities, dtype=float))
    return s


def test_none_input_returns_none():
    assert remove_peaks_relative_to_precursor_mz(None) is None


def test_missing_precursor_mz_raises_value_error():
    s = SpectrumBuilder().with_metadata({}).build()
    s.peaks = Fragments(mz=np.array([100.0]), intensities=np.array([1.0]))
    with pytest.raises(ValueError, match="Undefined 'precursor_mz'"):
        remove_peaks_relative_to_precursor_mz(s)


def test_non_scalar_precursor_mz_raises_value_error():
    s = SpectrumBuilder().with_metadata({"precursor_mz": [500.0]}).build()
    s.peaks = Fragments(mz=np.array([100.0]), intensities=np.array([1.0]))
    with pytest.raises(ValueError, match="Expected 'precursor_mz' to be a scalar"):
        remove_peaks_relative_to_precursor_mz(s)


def test_empty_peaks_handled():
    s = _build_spectrum(precursor_mz=500.0, mzs=[], intensities=[])
    out = remove_peaks_relative_to_precursor_mz(s)
    assert out.peaks.mz.size == 0
    assert out.peaks.intensities.size == 0


def test_default_threshold_removes_greater_than_precursor_minus_1p6():
    """
    With default offset_to_precursor = -1.6:
      threshold = precursor_mz + (-1.6) = precursor_mz - 1.6
      Peaks with mz > threshold are removed.
    """
    precursor = 500.0
    threshold = precursor - 1.6  # 498.4
    s = _build_spectrum(
        precursor_mz=precursor,
        mzs=[100.0, 300.0, 498.3, 498.4, 498.5, 500.0, 501.0, 600.0],
        intensities=[1, 2, 3, 4, 5, 6, 7, 8],
    )
    out = remove_peaks_relative_to_precursor_mz(s, clone=True)
    # Expect to keep peaks with mz <= 498.4
    assert np.allclose(out.peaks.mz, [100.0, 300.0, 498.3, 498.4])
    assert np.allclose(out.peaks.intensities, [1, 2, 3, 4])


def test_custom_relative_threshold():
    """
    With offset_to_precursor = -10:
      threshold = precursor - 10
      Remove peaks with mz > threshold.
    """
    precursor = 500.0
    s = _build_spectrum(
        precursor_mz=precursor,
        mzs=[480.0, 489.9, 490.0, 490.1, 600.0],
        intensities=[10, 20, 30, 40, 50],
    )
    out = remove_peaks_relative_to_precursor_mz(s, offset_to_precursor=-10.0)
    # Keep <= 490.0
    assert np.allclose(out.peaks.mz, [480.0, 489.9, 490.0])
    assert np.allclose(out.peaks.intensities, [10, 20, 30])


def test_clone_true_does_not_modify_original():
    precursor = 200.0
    s_in = _build_spectrum(
        precursor_mz=precursor,
        mzs=[150.0, 199.0, 199.0, 250.0],
        intensities=[1.0, 2.0, 3.0, 4.0],
    )
    out = remove_peaks_relative_to_precursor_mz(
        s_in, offset_to_precursor=-1.0, clone=True
        )

    # New object when cloned
    assert out is not s_in

    # Original unchanged
    assert np.allclose(s_in.peaks.mz, [150.0, 199.0, 199.0, 250.0])
    assert np.allclose(s_in.peaks.intensities, [1.0, 2.0, 3.0, 4.0])

    # Output filtered correctly: threshold = 199.0
    assert np.allclose(out.peaks.mz, [150.0, 199.0, 199.0])
    assert np.allclose(out.peaks.intensities, [1.0, 2.0, 3.0])


def test_clone_false_modifies_in_place():
    precursor = 300.0
    s_in = _build_spectrum(
        precursor_mz=precursor,
        mzs=[100.0, 200.0, 298.5, 310.0],
        intensities=[1.0, 2.0, 3.0, 4.0],
    )
    out = remove_peaks_relative_to_precursor_mz(
        s_in, offset_to_precursor=-1.0, clone=False
        )

    # Same object when not cloned
    assert out is s_in

    # threshold = 299.0 → drop 310.0 only
    assert np.allclose(s_in.peaks.mz, [100.0, 200.0, 298.5])
    assert np.allclose(s_in.peaks.intensities, [1.0, 2.0, 3.0])


def test_intensity_array_kept_in_lockstep_with_mz():
    """
    Ensure indexing keeps intensities aligned with mz after filtering.
    """
    s = _build_spectrum(
        precursor_mz=250.0,
        mzs=[100.0, 200.0, 248.4, 248.5, 400.0],
        intensities=[0.1, 0.2, 0.3, 0.4, 0.5],
    )
    out = remove_peaks_relative_to_precursor_mz(
        s, offset_to_precursor=-1.5
        )  # threshold 248.5
    # Remove > 248.5 → drop 400.0, keep 248.5 (equal) and below
    assert np.allclose(out.peaks.mz, [100.0, 200.0, 248.4, 248.5])
    assert np.allclose(out.peaks.intensities, [0.1, 0.2, 0.3, 0.4])
