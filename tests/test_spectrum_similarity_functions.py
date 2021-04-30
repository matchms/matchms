"""Test function to collect matching peaks. Run tests both on numba compiled and
pure Python version."""
import numpy
import pytest
from matchms.similarity.spectrum_similarity_functions import collect_peak_pairs
from matchms.similarity.spectrum_similarity_functions import find_matches
from matchms.similarity.spectrum_similarity_functions import score_best_matches


@pytest.mark.parametrize("shift, expected_pairs, expected_matches",
                         [(0.0, [[2., 2., 1.], [3., 3., 1.]], (2, 3)),
                          (-5.0, [[0., 0., 0.01], [1., 1., 0.01]], (2, 3))])
def test_collect_peak_pairs_compiled(shift, expected_pairs, expected_matches):
    """Test finding expected peak matches for given tolerance."""
    spec1 = numpy.array([[100, 200, 300, 500],
                         [0.1, 0.1, 1.0, 1.0]], dtype="float").T

    spec2 = numpy.array([[105, 205.1, 300, 500.1],
                         [0.1, 0.1, 1.0, 1.0]], dtype="float").T

    matching_pairs = numpy.array(collect_peak_pairs(spec1, spec2, tolerance=0.2, shift=shift))
    assert matching_pairs.shape == expected_matches, "Expected different number of matching peaks"
    assert numpy.allclose(matching_pairs, numpy.array(expected_pairs), atol=1e-8), "Expected different values."


@pytest.mark.parametrize("shift, expected_pairs, expected_matches",
                         [(0.0, [[2., 2., 1.], [3., 3., 1.]], (2, 3)),
                          (-5.0, [[0., 0., 0.01], [1., 1., 0.01]], (2, 3))])
def test_collect_peak_pairs(shift, expected_pairs, expected_matches):
    """Test finding expected peak matches for tolerance=0.2 and given shift."""
    spec1 = numpy.array([[100, 200, 300, 500],
                         [0.1, 0.1, 1.0, 1.0]], dtype="float").T

    spec2 = numpy.array([[105, 205.1, 300, 500.1],
                         [0.1, 0.1, 1.0, 1.0]], dtype="float").T

    matching_pairs = numpy.array(collect_peak_pairs.py_func(spec1, spec2, tolerance=0.2, shift=shift))
    assert matching_pairs.shape == expected_matches, "Expected different number of matching peaks"
    assert numpy.allclose(matching_pairs, numpy.array(expected_pairs), atol=1e-8), "Expected different values."


@pytest.mark.parametrize("numba_compiled", [True, False])
def test_collect_peak_pairs_no_matches(numba_compiled):
    """Test function for no matching peaks."""
    shift = -20.0
    spec1 = numpy.array([[100, 200, 300, 500],
                         [0.1, 0.1, 1.0, 1.0]], dtype="float").T

    spec2 = numpy.array([[105, 205.1, 300, 500.1],
                         [0.1, 0.1, 1.0, 1.0]], dtype="float").T
    if numba_compiled:
        matching_pairs = collect_peak_pairs(spec1, spec2, tolerance=0.2, shift=shift)
    else:
        matching_pairs = collect_peak_pairs.py_func(spec1, spec2, tolerance=0.2, shift=shift)
    assert matching_pairs is None, "Expected pairs to be None."


@pytest.mark.parametrize("numba_compiled", [True, False])
def test_find_matches_shifted(numba_compiled):
    """Test finding matches with shifted peaks."""
    shift = -5.0
    spec1_mz = numpy.array([100, 200, 300, 500], dtype="float")

    spec2_mz = numpy.array([105, 205.1, 300, 304.99, 500.1], dtype="float")

    expected_matches = [(0, 0), (1, 1), (2, 3)]
    if numba_compiled:
        matches = find_matches(spec1_mz, spec2_mz, tolerance=0.2, shift=shift)
    else:
        matches = find_matches.py_func(spec1_mz, spec2_mz, tolerance=0.2, shift=shift)
    assert expected_matches == matches, "Expected different matches."


@pytest.mark.parametrize("numba_compiled", [True, False])
def test_find_matches_no_matches(numba_compiled):
    """Test function for no matching peaks."""
    shift = -20.0
    spec1_mz = numpy.array([100, 200, 300, 500], dtype="float")

    spec2_mz = numpy.array([105, 205.1, 300, 500.1], dtype="float")
    if numba_compiled:
        matches = find_matches(spec1_mz, spec2_mz, tolerance=0.2, shift=shift)
    else:
        matches = find_matches.py_func(spec1_mz, spec2_mz, tolerance=0.2, shift=shift)
    assert matches == [], "Expected empty list of matches."


@pytest.mark.parametrize("matching_pairs, expected_score",
                         [([[2., 2., 1.], [3., 3., 1.]], (0.990099009900, 2)),
                          ([[0., 0., 0.01], [1., 1., 0.01]], (0.009900990099, 2))])
def test_score_best_matches_compiled(matching_pairs, expected_score):
    """Test finding expected peak matches for given tolerance."""
    matching_pairs = numpy.array(matching_pairs)
    spec1 = numpy.array([[100, 200, 300, 500],
                         [0.1, 0.1, 1.0, 1.0]], dtype="float").T

    spec2 = numpy.array([[105, 205.1, 300, 500.1],
                         [0.1, 0.1, 1.0, 1.0]], dtype="float").T

    score, matches = score_best_matches(matching_pairs, spec1, spec2)
    assert score == pytest.approx(expected_score[0], 1e-6), "Expected different score"
    assert matches == expected_score[1], "Expected different matches."


@pytest.mark.parametrize("matching_pairs, expected_score",
                         [([[2., 2., 1.], [3., 3., 1.]], (0.990099009900, 2)),
                          ([[0., 0., 0.01], [1., 1., 0.01]], (0.009900990099, 2))])
def test_score_best_matches(matching_pairs, expected_score):
    """Test finding expected peak matches for given tolerance."""
    matching_pairs = numpy.array(matching_pairs)
    spec1 = numpy.array([[100, 200, 300, 500],
                         [0.1, 0.1, 1.0, 1.0]], dtype="float").T

    spec2 = numpy.array([[105, 205.1, 300, 500.1],
                         [0.1, 0.1, 1.0, 1.0]], dtype="float").T

    score, matches = score_best_matches.py_func(matching_pairs, spec1, spec2)
    assert score == pytest.approx(expected_score[0], 1e-6), "Expected different score"
    assert matches == expected_score[1], "Expected different matches."
