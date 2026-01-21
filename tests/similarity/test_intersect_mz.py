import numpy as np
import pytest
from matchms.similarity import IntersectMz
from ..builder_Spectrum import SpectrumBuilder


def test_intersect_mz_without_parameters():
    """Compare score with expected value."""
    intensities = np.array([1.0, 1.0, 1.0, 1.0], dtype="float")
    builder = SpectrumBuilder().with_intensities(intensities)

    spectrum_1 = builder.with_mz(np.array([100, 200, 300, 500], dtype="float")).build()
    spectrum_2 = builder.with_mz(np.array([100, 200, 290, 499.9], dtype="float")).build()

    similarity_score = IntersectMz()
    score = similarity_score.pair(spectrum_1, spectrum_2)

    assert score == pytest.approx(1 / 3, 0.0001), "Expected different score."
