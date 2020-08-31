import numpy
import pytest
from matchms import Spectrum
from matchms.similarity import IntersectMz


def test_intersect_mz_without_parameters():
    """Compare score with expected value."""
    spectrum_1 = Spectrum(mz=numpy.array([100, 200, 300, 500], dtype="float"),
                          intensities=numpy.array([1.0, 1.0, 1.0, 1.0], dtype="float"))

    spectrum_2 = Spectrum(mz=numpy.array([100, 200, 290, 499.9], dtype="float"),
                          intensities=numpy.array([1.0, 1.0, 1.0, 1.0], dtype="float"))
    similarity_score = IntersectMz()
    score = similarity_score.pair(spectrum_1, spectrum_2)

    assert score == pytest.approx(1/3, 0.0001), "Expected different score."
