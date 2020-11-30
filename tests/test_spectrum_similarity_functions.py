"""Test function to collect matching peaks. Run tests both on numba compiled and
pure Python version."""
import numpy
import pytest
from matchms.similarity.spectrum_similarity_functions import collect_peak_pairs
from matchms.similarity.spectrum_similarity_functions import score_best_matches


@pytest.mark.parametrize("shift, expected_pairs",
                         [(0.0, [(2, 2, 1.0), (3, 3, 1.0)]),
                          (-5.0, [(0, 0, 0.01), (1, 1, 0.01)])])
def test_collect_peak_pairs_compiled(shift, expected_pairs):
    """Test finding expected peak matches for given tolerance."""
    shift = 0.0
    expected_pairs = [[2., 2., 1.], [3., 3., 1.]]
    spec1 = numpy.array([[100, 200, 300, 500],
                         [0.1, 0.1, 1.0, 1.0]], dtype="float").T

    spec2 = numpy.array([[105, 205.1, 300, 500.1],
                         [0.1, 0.1, 1.0, 1.0]], dtype="float").T

    matching_pairs = collect_peak_pairs(spec1, spec2, tolerance=0.2, shift=shift)
    assert matching_pairs.shape == (2, 3), "Expected different number of matching peaks"
    assert np.allclose(matching_pairs, np.array(expected_pairs), atol=1e-8), "Expected different values."


@pytest.mark.parametrize("shift, expected_pairs",
                         [(0.0, [[2., 2., 1.], [3., 3., 1.]]),
                          (-5.0, [[0., 0., 0.01], [1., 1., 0.01]])])
def test_collect_peak_pairs(shift, expected_pairs):
    """Test finding expected peak matches for tolerance=0.2 and given shift."""
    spec1 = numpy.array([[100, 200, 300, 500],
                         [0.1, 0.1, 1.0, 1.0]], dtype="float").T

    spec2 = numpy.array([[105, 205.1, 300, 500.1],
                         [0.1, 0.1, 1.0, 1.0]], dtype="float").T

    matching_pairs = collect_peak_pairs.py_func(spec1, spec2, tolerance=0.2, shift=shift)
    assert matching_pairs.shape == (2, 3), "Expected different number of matching peaks"
    assert np.allclose(matching_pairs, np.array(expected_pairs), atol=1e-8), "Expected different values."


@pytest.mark.parametrize("shift, matching_pairs", "expected_score",
                         [(0.0, [[2., 2., 1.], [3., 3., 1.]], (0.9900990099, 2)),
                          (-5.0, [[0., 0., 0.01], [1., 1., 0.01]], (0.0099009900, 2))])
def test_score_best_matches_compiled(shift, matching_pairs, expected_score):
    """Test finding expected peak matches for given tolerance."""
    shift = -5.0
    expected_pairs = [[2., 2., 1.], [3., 3., 1.]]
    spec1 = numpy.array([[100, 200, 300, 500],
                         [0.1, 0.1, 1.0, 1.0]], dtype="float").T

    spec2 = numpy.array([[105, 205.1, 300, 500.1],
                         [0.1, 0.1, 1.0, 1.0]], dtype="float").T

    score, matches = score_best_matches(matching_pairs, spec1, spec2)
    assert score == pytest.approx(expected_score[0], 1e-8), "Expected different score"
    assert matches == expected_score[1], "Expected different matches."


@pytest.mark.parametrize("shift, matching_pairs", "expected_score",
                         [(0.0, [[2., 2., 1.], [3., 3., 1.]], (0.9900990099, 2)),
                          (-5.0, [[0., 0., 0.01], [1., 1., 0.01]], (0.0099009900, 2))])
def test_score_best_matches(shift, matching_pairs, expected_score):
    """Test finding expected peak matches for given tolerance."""
    shift = -5.0
    expected_pairs = [[2., 2., 1.], [3., 3., 1.]]
    spec1 = numpy.array([[100, 200, 300, 500],
                         [0.1, 0.1, 1.0, 1.0]], dtype="float").T

    spec2 = numpy.array([[105, 205.1, 300, 500.1],
                         [0.1, 0.1, 1.0, 1.0]], dtype="float").T

    score, matches = .py_funcscore_best_matches(matching_pairs, spec1, spec2)
    assert score == pytest.approx(expected_score[0], 1e-8), "Expected different score"
    assert matches == expected_score[1], "Expected different matches."
