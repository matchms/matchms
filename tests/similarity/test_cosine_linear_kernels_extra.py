"""Focused edge and independent-oracle tests for the Numba linear cosine kernels.

Run with JIT enabled for correctness, and with NUMBA_DISABLE_JIT=1 for
meaningful branch coverage in Python mode.
"""

from functools import lru_cache
import numpy as np
import pytest
from matchms.similarity.cosine_linear_functions import (
    _is_well_separated,
    _linear_partners,
    _select_path_edges,
    _sorted_peak_order,
    _weighted_products,
    linear_cosine_score,
    modified_linear_cosine_score,
    sirius_merge_close_peaks,
)


pytestmark = pytest.mark.numba_coverage


def _peaks(*rows):
    return np.asarray(rows, dtype=np.float64).reshape(-1, 2)


def _modified_cosine_oracle(left, right, precursor_left, precursor_right, tolerance):
    """Small exhaustive maximum-weight matching, independent of Numba's path DP."""
    w1 = left[:, 1]
    w2 = right[:, 1]
    normalizer = np.linalg.norm(w1) * np.linalg.norm(w2)
    if normalizer == 0:
        return 0.0
    shift_allowed = abs(precursor_left - precursor_right) > tolerance
    adjacency = []
    for mz1, intensity1 in left:
        options = []
        for j, (mz2, intensity2) in enumerate(right):
            direct = abs(mz1 - mz2) <= tolerance
            shifted = shift_allowed and abs((mz1 - precursor_left) - (mz2 - precursor_right)) <= tolerance
            if direct or shifted:
                options.append((j, intensity1 * intensity2))
        adjacency.append(tuple(options))

    @lru_cache(None)
    def maximum(i, occupied):
        if i == len(adjacency):
            return 0.0
        result = maximum(i + 1, occupied)
        for j, benefit in adjacency[i]:
            if not (occupied & (1 << j)):
                result = max(result, benefit + maximum(i + 1, occupied | (1 << j)))
        return result

    return maximum(0, 0) / normalizer


@pytest.mark.parametrize(
    "rows,expected_indices",
    [
        ([(100, 1), (110, 3), (120, 2)], [0, 2, 1]),
        ([(100, 1), (110, 1), (120, 1)], [2, 1, 0]),
        ([], []),
    ],
)
def test_sorted_peak_order_ties(rows, expected_indices):
    np.testing.assert_array_equal(_sorted_peak_order(_peaks(*rows)), expected_indices)


def test_merge_ties_and_consumed_peaks_are_deterministic():
    original = _peaks((100, 1), (100.1, 1), (100.15, 1), (110, 2), (110.1, 1))
    before = original.copy()
    merged = sirius_merge_close_peaks(original, 0.1)
    np.testing.assert_array_equal(merged, _peaks((100, 3), (110, 3)))
    np.testing.assert_array_equal(original, before)
    np.testing.assert_array_equal(sirius_merge_close_peaks(_peaks(), 0.1), _peaks())


@pytest.mark.parametrize("mz,expected", [([100, 100.125], False), ([100, 100.12501], True), ([], True)])
def test_well_separated_inclusive_boundary(mz, expected):
    assert bool(_is_well_separated(np.asarray(mz, float), 0.0625)) is expected


def test_linear_partners_mark_unmatched_and_advance_once():
    left = np.array([100.0, 110.0, 120.0])
    right = np.array([99.9, 110.0, 130.0])
    first = np.empty(left.size, np.int64)
    second = np.empty(right.size, np.int64)
    _linear_partners(left, right, 0.125, first, second)
    np.testing.assert_array_equal(first, [0, 1, -1])
    np.testing.assert_array_equal(second, [0, 1, -1])


@pytest.mark.parametrize(
    "weights,chosen", [([1.0], [True]), ([1.0, 2.0], [False, True]), ([3.0, 1.0, 3.0], [True, False, True])]
)
def test_select_path_edges_finds_optimum(weights, chosen):
    benefits = np.asarray(weights, dtype=float)
    selected = np.zeros(benefits.size, dtype=np.bool_)
    dp = np.zeros(benefits.size + 1, dtype=float)
    _select_path_edges(benefits.size, benefits, dp, selected)
    np.testing.assert_array_equal(selected, chosen)


def test_weighted_products_and_linear_zero_norm():
    products, norm = _weighted_products(_peaks((100, 0.25), (200, 0.5)), 1.0, 1.0)
    np.testing.assert_allclose(products, [25.0, 100.0])
    assert norm == pytest.approx(np.hypot(25, 100))
    assert linear_cosine_score(_peaks((100, 0)), _peaks((100, 1)), 0.1, 0, 1) == (0.0, 0)
    assert linear_cosine_score(_peaks(), _peaks((100, 1)), 0.1, 0, 1) == (0.0, 0)


def test_modified_linear_rejects_unmerged_input_and_zero_norm():
    bad = _peaks((100, 1), (100.1, 1))
    good = _peaks((100, 1), (200, 1))
    with pytest.raises(ValueError, match="well-separated"):
        modified_linear_cosine_score(bad, good, 500, 510, 0.1, 0, 1)
    assert modified_linear_cosine_score(_peaks(), good, 500, 510, 0.1, 0, 1) == (0.0, 0)
    assert modified_linear_cosine_score(_peaks((100, 0)), good, 500, 510, 0.1, 0, 1) == (0.0, 0)


@pytest.mark.parametrize("seed", range(12))
def test_modified_linear_agrees_with_exhaustive_matching(seed):
    rng = np.random.default_rng(seed)
    # Well-separated m/z, but deliberately allow precursor-shift/direct conflicts.
    masses = np.array([100.0, 120.0, 140.0, 160.0, 180.0, 200.0])
    left_idx = rng.choice(len(masses), 4, replace=False)
    right_idx = rng.choice(len(masses), 4, replace=False)
    left = _peaks(*sorted(zip(masses[left_idx], rng.choice([0.0, 0.25, 0.5, 1.0], 4), strict=True)))
    right = _peaks(*sorted(zip(masses[right_idx], rng.choice([0.0, 0.25, 0.5, 1.0], 4), strict=True)))
    p_left, p_right = 500.0, (500.0 if seed % 3 == 0 else 520.0)
    score, matches = modified_linear_cosine_score(left, right, p_left, p_right, 0.1, 0.0, 1.0)
    expected = _modified_cosine_oracle(left, right, p_left, p_right, 0.1)
    assert score == pytest.approx(expected, abs=1e-12)
    assert 0 <= matches <= min(len(left), len(right))


def test_modified_linear_direct_and_shifted_compete_for_one_peak():
    left = _peaks((100, 1), (120, 2))
    right = _peaks((100, 2), (140, 1))
    actual, matches = modified_linear_cosine_score(left, right, 500, 520, 0.1, 0, 1)
    expected = _modified_cosine_oracle(left, right, 500, 520, 0.1)
    assert actual == pytest.approx(expected, abs=1e-12)
    assert matches == 2
