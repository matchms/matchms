import math
import numpy as np
import pytest
from matchms.similarity.flash_utils import (
    _build_library_index,
    _clean_and_weight,
    _entropy_weight,
    _LibraryIndex,
    _merge_within,
)


# ----------------------------
# Entropy weighting
# ----------------------------

def test_entropy_weight_behaviour():
    intens = np.array([1.0, 1.0, 1.0, 1.0])
    # entropy of uniform over 4 bins = 2 bits -> w = 0.25 + 0.25 * 2 = 0.75
    w = 0.25 + 0.25 * 2.0
    expected = np.power(intens, w).astype(np.float32)
    got = _entropy_weight(intens, np.float32)
    assert np.allclose(got, expected, rtol=0, atol=1e-7)


def test_entropy_weight_zero_total_returns_input():
    intens = np.array([0.0, 0.0, 0.0, 0.0], dtype=float)
    got = _entropy_weight(intens, np.float32)
    # Should be returned as-is (dtype conversion allowed)
    assert got.dtype == np.float32
    assert np.allclose(got, 0.0, atol=0.0)


def test_entropy_weight_saturation_at_ge3_bits():
    # 8 equal bins => entropy = 3.0 bits => w = 1.0 (saturation)
    intens = np.ones(8, dtype=float)
    got = _entropy_weight(intens, np.float32)
    assert np.allclose(got, intens.astype(np.float32), atol=0.0)


# ----------------------------
# Merge-within
# ----------------------------

def test_merge_within_weighted_average():
    # two close peaks at 100.00 and 100.03 within 0.05 should merge
    peaks = np.column_stack([
        np.array([100.00, 100.03, 120.0], dtype=np.float32),
        np.array([1.0,    3.0,    2.0], dtype=np.float32),
    ])
    out = _merge_within(peaks, max_delta_da=0.05)
    # first two merged into weighted center: (100*1 + 100.03*3) / (1+3) = 100.0225
    assert out.shape == (2, 2)
    assert math.isclose(out[0, 0], np.array(100.0225, dtype=np.float32), rel_tol=0, abs_tol=1e-6)
    assert math.isclose(out[0, 1], 4.0, rel_tol=0, abs_tol=1e-6)
    assert math.isclose(out[1, 0], 120.0, rel_tol=0, abs_tol=1e-6)
    assert math.isclose(out[1, 1], 2.0, rel_tol=0, abs_tol=1e-6)


def test_merge_within_no_merge_and_zero_total_behaviour():
    # No merge if beyond threshold
    peaks_far = np.array([[100.0, 1.0], [100.2, 2.0]], dtype=np.float32)
    out_far = _merge_within(peaks_far, max_delta_da=0.05)
    assert np.allclose(out_far, peaks_far)

    # Merge two zero-intensity peaks within threshold -> intensity remains 0,
    # mz becomes last seen (code path total <= 0)
    peaks_zero = np.array([[100.00, 0.0], [100.02, 0.0]], dtype=np.float32)
    out_zero = _merge_within(peaks_zero, max_delta_da=0.05)
    assert out_zero.shape == (1, 2)
    assert math.isclose(out_zero[0, 0], 100.02, rel_tol=1e-6)
    assert math.isclose(out_zero[0, 1], 0.0, abs_tol=0.0)


# ----------------------------
# Preprocessing pipeline
# ----------------------------

def test_clean_and_weight_pipeline_precursor_noise_norm_merge():
    # Note: pass peaks as shape (N, 2)
    mz = np.array([100, 199.0, 199.5, 199.9, 210], dtype=float)
    intens = np.array([0.05, 0.2, 0.01, 0.2,  0.01], dtype=float)
    pmz = 200.0
    out = _clean_and_weight(
        np.column_stack([mz, intens]),
        precursor_mz=pmz,
        remove_precursor=True,
        precursor_window=1.6,   # keep only <= 198.4
        noise_cutoff=0.05,      # remove peaks below 5% of max (max AFTER precursor filter)
        normalize_to_half=True,
        merge_within_da=0.5,    # merge anything within 0.5 Da
        weighing_type="entropy",
        dtype=np.float32
    )
    # Only m/z=100 survives precursor filter (<=198.4); others removed
    assert out.shape[0] == 1
    assert math.isclose(out[0, 0], 100.0, rel_tol=0, abs_tol=1e-12)
    # Intensities are entropy-weighted then normalized to sum 0.5
    assert math.isclose(float(out[:, 1].sum()), 0.5, rel_tol=0, abs_tol=1e-7)


def test_clean_and_weight_cosine_weighting_no_change_when_not_normalized():
    peaks = np.array([[100.0, 1.0], [150.0, 2.0], [200.0, 3.0]], dtype=float)
    out = _clean_and_weight(peaks,
                            precursor_mz=None,
                            remove_precursor=False,
                            precursor_window=1.6,
                            noise_cutoff=0.0,
                            normalize_to_half=False,   # ensure no scaling applied
                            merge_within_da=0.0,
                            weighing_type="cosine",
                            dtype=np.float32)
    assert out.dtype == np.float32
    assert np.allclose(out[:, 0], peaks[:, 0])
    assert np.allclose(out[:, 1], peaks[:, 1])


def test_clean_and_weight_raises_on_unknown_weighing_type():
    peaks = np.array([[100.0, 1.0]], dtype=float)
    with pytest.raises(ValueError, match="Score type '.*' not recognized"):
        _clean_and_weight(peaks,
                          precursor_mz=None,
                          remove_precursor=False,
                          precursor_window=1.6,
                          noise_cutoff=0.0,
                          normalize_to_half=False,
                          merge_within_da=0.0,
                          weighing_type="nope",
                          dtype=np.float32)


# ----------------------------
# _LibraryIndex / _build_library_index
# ----------------------------

def test_library_index_empty_with_neutral_loss_and_l2():
    idx = _build_library_index([],
                               [],
                               compute_neutral_loss=True,
                               compute_l2_norm=True,
                               dtype=np.float32)
    assert isinstance(idx, _LibraryIndex)
    assert idx.n_specs == 0
    assert idx.peaks_mz.size == 0 and idx.peaks_int.size == 0 and idx.peaks_spec_idx.size == 0
    assert idx.nl_mz.size == 0 and idx.nl_int.size == 0 and idx.nl_spec_idx.size == 0 and idx.nl_product_idx.size == 0
    assert idx.spec_l2.size == 0
    assert idx.precursor_mz.size == 0
    assert idx.dtype == np.float32


def test_library_index_single_spec_no_precursor_no_nl():
    # One spectrum, no precursor -> NL arrays empty; L2 present if requested.
    p = np.array([[100.0, 1.0], [200.0, 2.0]], dtype=np.float32)
    idx = _build_library_index([p], [None],
                               compute_neutral_loss=True,
                               compute_l2_norm=True,
                               dtype=np.float32)
    assert idx.n_specs == 1
    # Peaks sorted ascending
    assert np.allclose(idx.peaks_mz, [100.0, 200.0])
    assert np.allclose(idx.peaks_int, [1.0, 2.0])
    assert idx.peaks_spec_idx.dtype == np.int32
    assert np.all(idx.peaks_spec_idx == np.array([0, 0], dtype=np.int32))
    # NL empty due to missing pmz
    assert idx.nl_mz.size == 0 and idx.nl_int.size == 0
    # L2 norm sqrt(1^2+2^2)
    assert math.isclose(float(idx.spec_l2[0]), math.sqrt(5.0), rel_tol=0, abs_tol=1e-6)
    # precursor array filled with NaN
    assert np.isnan(idx.precursor_mz[0])


def test_library_index_two_specs_with_pmz_nl_sorted_and_product_mapping():
    # Spec 0: pmz=500 peaks -> products [100, 300], NL [400, 200]
    # Spec 1: pmz=510 peaks -> product [150], NL [360]
    p0 = np.array([[100.0, 0.5], [300.0, 0.2]], dtype=np.float32)
    p1 = np.array([[150.0, 0.8]], dtype=np.float32)
    idx = _build_library_index([p0, p1], [500.0, 510.0],
                               compute_neutral_loss=True,
                               compute_l2_norm=True,
                               dtype=np.float32)

    # Product arrays are globally sorted by m/z
    assert np.allclose(idx.peaks_mz, [100.0, 150.0, 300.0], atol=1e-12)
    assert np.all(idx.peaks_spec_idx == np.array([0, 1, 0], dtype=np.int32))
    assert np.allclose(idx.peaks_int, [0.5, 0.8, 0.2], atol=1e-12)

    # Neutral-loss values (unsorted would be [400,200,360]); verify sorted ascending
    assert np.allclose(np.sort(idx.nl_mz), idx.nl_mz, atol=0.0)
    assert idx.nl_mz.size == 3
    # Spec indices carried along and product index maps to corresponding product peak
    for k in range(idx.nl_mz.size):
        col = int(idx.nl_spec_idx[k])
        pmz = float(idx.precursor_mz[col])
        loss = float(idx.nl_mz[k])
        # product m/z that generated this loss
        expected_prod_mz = pmz - loss
        prod_pos = int(idx.nl_product_idx[k])
        assert math.isclose(float(idx.peaks_mz[prod_pos]), expected_prod_mz, abs_tol=1e-6)

    # L2 per spectrum
    # spec0: sqrt(0.5^2 + 0.2^2) ; spec1: sqrt(0.8^2)
    expected0 = math.sqrt(0.25 + 0.04)
    expected1 = 0.8
    assert math.isclose(float(idx.spec_l2[0]), expected0, abs_tol=1e-6)
    assert math.isclose(float(idx.spec_l2[1]), expected1, abs_tol=1e-6)
    # dtype preserved
    assert idx.dtype == np.float32
    assert idx.peaks_mz.dtype == np.float32
    assert idx.nl_mz.dtype == np.float32


def test_library_index_dtype_float64_and_values():
    p = np.array([[100.0, 1.0], [200.0, 2.0]], dtype=np.float64)
    idx = _build_library_index([p], [500.0],
                               compute_neutral_loss=True,
                               compute_l2_norm=True,
                               dtype=np.float64)
    assert idx.dtype == np.float64
    assert idx.peaks_mz.dtype == np.float64
    assert idx.nl_mz.dtype == np.float64
    # NL values should be [400, 300] (sorted -> [300, 400])
    assert np.allclose(idx.nl_mz, [300.0, 400.0], atol=1e-12)
    # L2 = sqrt(1^2 + 2^2)
    assert math.isclose(float(idx.spec_l2[0]), math.sqrt(5.0), abs_tol=1e-12)
