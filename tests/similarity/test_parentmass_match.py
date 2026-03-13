import numpy as np
import pytest
from matchms.similarity import ParentMassMatch
from ..builder_Spectrum import SpectrumBuilder, spectra_factory


@pytest.mark.parametrize('parent_mass, tolerance, expected', [
    [[100.0, 101.0], 0.1, False],
    [[100.0, 101.0], 2.0, True]
])
def test_parentmass_match_parameterized(parent_mass, tolerance, expected):
    s0, s1 = spectra_factory('parent_mass', parent_mass)
    similarity_score = ParentMassMatch(tolerance=tolerance)
    scores = similarity_score.pair(s0, s1)
    assert np.all(scores == np.array(expected)), "Expected different scores."


def test_parentmass_match_missing_parentmass():
    """Missing parentmass entries should return False."""
    builder = SpectrumBuilder()
    spectrum_1 = builder.with_metadata({"parent_mass": 100.0}).build()
    spectrum_2 = builder.with_metadata({}).build()

    similarity_score = ParentMassMatch(tolerance=2.0)

    score = similarity_score.pair(spectrum_1, spectrum_2)
    assert score == np.array(False, dtype=bool)


@pytest.mark.parametrize('parent_mass, tolerance, expected', [
    [[100.0, 101.0, 99.0, 98.0], 0.1, [[False, False], [False, False]]],
    [[100.0, 101.0, 99.0, 98.0], 2.0, [[True, True], [True, False]]]
])
def test_parentmass_match_array_parameterized(parent_mass, tolerance, expected):
    s0, s1, s2, s3 = spectra_factory('parent_mass', parent_mass)
    similarity_score = ParentMassMatch(tolerance=tolerance)
    scores = similarity_score.matrix([s0, s1], [s2, s3])
    assert np.all(scores == np.array(expected)), "Expected different scores."
