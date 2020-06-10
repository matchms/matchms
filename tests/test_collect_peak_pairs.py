"""Test function to collect matching peaks. Run tests both on numba compiled and
pure Python version."""
import numpy
from matchms.similarity.collect_peak_pairs import collect_peak_pairs


def test_cosine_hungarian_tolerance_01_compiled():
    """Test finding expected peak matches within tolerance=0.2."""
    spec1 = numpy.array([[100, 200, 300, 500],
                         [0.1, 0.1, 1.0, 1.0]], dtype="float").T

    spec2 = numpy.array([[105, 205.1, 300, 500.1],
                         [0.1, 0.1, 1.0, 1.0]], dtype="float").T

    matching_pairs = collect_peak_pairs(spec1, spec2, tolerance=0.2)
    assert len(matching_pairs) == 2, "Expected different number of matching peaks"
    assert matching_pairs == [(2, 2, 1.0), (3, 3, 1.0)], "Expected different matchin pairs."


def test_cosine_hungarian_tolerance_01_shift_min5_compiled():
    """Test finding expected peak matches when given a mass_shift of -5.0."""
    spec1 = numpy.array([[100, 200, 300, 500],
                         [1.0, 1.0, 0.1, 0.1]], dtype="float").T

    spec2 = numpy.array([[105, 205.1, 300, 500.1],
                         [1.0, 1.0, 0.1, 0.1]], dtype="float").T

    matching_pairs = collect_peak_pairs(spec1, spec2, tolerance=0.2, shift=-5.0)
    assert len(matching_pairs) == 2, "Expected different number of matching peaks"
    assert matching_pairs == [(0, 0, 1.0), (1, 1, 1.0)], "Expected different matchin pairs."


def test_cosine_hungarian_tolerance_01():
    """Test finding expected peak matches within tolerance=0.2."""
    spec1 = numpy.array([[100, 200, 300, 500],
                         [0.1, 0.1, 1.0, 1.0]], dtype="float").T

    spec2 = numpy.array([[105, 205.1, 300, 500.1],
                         [0.1, 0.1, 1.0, 1.0]], dtype="float").T

    matching_pairs = collect_peak_pairs.py_func(spec1, spec2, tolerance=0.2)
    assert len(matching_pairs) == 2, "Expected different number of matching peaks"
    assert matching_pairs == [(2, 2, 1.0), (3, 3, 1.0)], "Expected different matchin pairs."


def test_cosine_hungarian_tolerance_01_shift_min5():
    """Test finding expected peak matches when given a mass_shift of -5.0."""
    spec1 = numpy.array([[100, 200, 300, 500],
                         [1.0, 1.0, 0.1, 0.1]], dtype="float").T

    spec2 = numpy.array([[105, 205.1, 300, 500.1],
                         [1.0, 1.0, 0.1, 0.1]], dtype="float").T

    matching_pairs = collect_peak_pairs.py_func(spec1, spec2, tolerance=0.2, shift=-5.0)
    assert len(matching_pairs) == 2, "Expected different number of matching peaks"
    assert matching_pairs == [(0, 0, 1.0), (1, 1, 1.0)], "Expected different matchin pairs."
