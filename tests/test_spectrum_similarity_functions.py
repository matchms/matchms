"""Test function to collect matching peaks. Run tests both on numba compiled and
pure Python version."""
import numpy as np
import pytest
from matchms.similarity.spectrum_similarity_functions import (
    collect_peak_pairs, find_matches, score_best_matches)
from .builder_Spectrum import SpectrumBuilder


@pytest.fixture
def spectra():
    builder = SpectrumBuilder()
    spec1 = builder.with_mz([100, 200, 300, 500]).with_intensities([0.1, 0.1, 1.0, 1.0]).build()
    spec2 = builder.with_mz([105, 205.1, 300, 500.1]).with_intensities([0.1, 0.1, 1.0, 1.0]).build()
    return spec1.peaks.to_numpy, spec2.peaks.to_numpy

    # spec1 = np.array([[100, 200, 300, 500],
    #                      [0.1, 0.1, 1.0, 1.0]], dtype="float").T

    # spec2 = np.array([[105, 205.1, 300, 500.1],
    #                      [0.1, 0.1, 1.0, 1.0]], dtype="float").T

    # return spec1, spec2


def get_function(numba_compiled, f):
    if numba_compiled:
        return f
    return f.py_func


@pytest.mark.parametrize("numba_compiled", [True, False])
@pytest.mark.parametrize("shift, expected_pairs, expected_matches", [
    (0.0, [[2., 2., 1.], [3., 3., 1.]], (2, 3)),
    (-5.0, [[0., 0., 0.01], [1., 1., 0.01]], (2, 3)),
    (-20.0, None, None)
])
def test_collect_peak_pairs(numba_compiled, shift, expected_pairs, expected_matches, spectra):
    """Test finding expected peak matches for given tolerance."""
    spec1, spec2 = spectra

    func = get_function(numba_compiled, collect_peak_pairs)
    matching_pairs = func(spec1, spec2, tolerance=0.2, shift=shift)

    if expected_matches is not None:
        matching_pairs = np.array(matching_pairs)
        assert matching_pairs.shape == expected_matches, "Expected different number of matching peaks"
        assert np.allclose(matching_pairs, np.array(expected_pairs), atol=1e-8), "Expected different values."
    else:
        assert matching_pairs is None, "Expected pairs to be None."


@pytest.mark.parametrize("numba_compiled", [True, False])
@pytest.mark.parametrize("shift, expected_matches", [
    (-5.0, [(0, 0), (1, 1), (2, 3)]),
    (-20.0, [])
])
def test_find_matches(numba_compiled, shift, expected_matches):
    """Test finding matches with shifted peaks."""
    spec1_mz = np.array([100, 200, 300, 500], dtype="float")
    spec2_mz = np.array([105, 205.1, 300, 304.99, 500.1], dtype="float")

    func = get_function(numba_compiled, find_matches)
    matches = func(spec1_mz, spec2_mz, tolerance=0.2, shift=shift)

    assert expected_matches == matches, "Expected different matches."


@pytest.mark.parametrize("numba_compiled", [True, False])
@pytest.mark.parametrize("matching_pairs, expected_score",
                         [([[2., 2., 1.], [3., 3., 1.]], (0.990099009900, 2)),
                          ([[0., 0., 0.01], [1., 1., 0.01]], (0.009900990099, 2))])
def test_score_best_matches(numba_compiled, matching_pairs, expected_score, spectra):
    """Test finding expected peak matches for given tolerance."""
    matching_pairs = np.array(matching_pairs)
    spec1, spec2 = spectra

    func = get_function(numba_compiled, score_best_matches)

    score, matches = func(matching_pairs, spec1, spec2)
    assert score == pytest.approx(expected_score[0], 1e-6), "Expected different score"
    assert matches == expected_score[1], "Expected different matches."
