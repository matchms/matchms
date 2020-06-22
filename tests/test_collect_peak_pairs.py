"""Test function to collect matching peaks. Run tests both on numba compiled and
pure Python version."""
import numpy
import pytest
from matchms.similarity.collect_peak_pairs import collect_peak_pairs


@pytest.mark.parametrize("tolerance, expected_pairs",
                         [(0.2, [(2, 2, 1.0), (3, 3, 1.0)]),
                          (5.0, [(0, 0, 1.0), (1, 1, 1.0)])])
def test_cosine_hungarian_compiled(tolerance, expected_pairs):
    """Test finding expected peak matches for given tolerance."""
    spec1 = numpy.array([[100, 200, 300, 500],
                         [0.1, 0.1, 1.0, 1.0]], dtype="float").T

    spec2 = numpy.array([[105, 205.1, 300, 500.1],
                         [0.1, 0.1, 1.0, 1.0]], dtype="float").T

    matching_pairs = collect_peak_pairs(spec1, spec2, tolerance=tolerance)
    assert len(matching_pairs) == 2, "Expected different number of matching peaks"
    assert matching_pairs == expected_pairs, "Expected different matchin pairs."


@pytest.mark.parametrize("tolerance, expected_pairs",
                         [(0.2, [(2, 2, 1.0), (3, 3, 1.0)]),
                          (5.0, [(0, 0, 1.0), (1, 1, 1.0)])])
def test_cosine_hungarian(tolerance, expected_pairs):
    """Test finding expected peak matches for given tolerance."""
    spec1 = numpy.array([[100, 200, 300, 500],
                         [0.1, 0.1, 1.0, 1.0]], dtype="float").T

    spec2 = numpy.array([[105, 205.1, 300, 500.1],
                         [0.1, 0.1, 1.0, 1.0]], dtype="float").T

    matching_pairs = collect_peak_pairs.py_func(spec1, spec2, tolerance=tolerance)
    assert len(matching_pairs) == 2, "Expected different number of matching peaks"
    assert matching_pairs == expected_pairs, "Expected different matchin pairs."
