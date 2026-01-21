import numpy as np
import pytest
from matchms.similarity import ParentMassMatch
from ..builder_Spectrum import SpectrumBuilder, spectra_factory


@pytest.mark.parametrize(
    "parent_mass, tolerance, expected", [[[100.0, 101.0], 0.1, False], [[100.0, 101.0], 2.0, True]]
)
def test_parentmass_match_parameterized(parent_mass, tolerance, expected):
    s0, s1 = spectra_factory("parent_mass", parent_mass)
    similarity_score = ParentMassMatch(tolerance=tolerance)
    scores = similarity_score.pair(s0, s1)
    assert np.all(scores == np.array(expected)), "Expected different scores."


def test_parentmass_match_missing_parentmass():
    "Test with missing parentmass."
    builder = SpectrumBuilder()
    spectrum_1 = builder.with_metadata({"parent_mass": 100.0}).build()
    spectrum_2 = builder.with_metadata({}).build()

    similarity_score = ParentMassMatch(tolerance=2.0)

    with pytest.raises(AssertionError) as msg:
        _ = similarity_score.pair(spectrum_1, spectrum_2)

    expected_message_part = "Missing parent mass."
    assert expected_message_part in str(msg.value), "Expected particular error message."


@pytest.mark.parametrize(
    "parent_mass, tolerance, expected",
    [
        [[100.0, 101.0, 99.0, 98.0], 0.1, [[False, False], [False, False]]],
        [[100.0, 101.0, 99.0, 98.0], 2.0, [[True, True], [True, False]]],
    ],
)
def test_parentmass_match_array_parameterized(parent_mass, tolerance, expected):
    s0, s1, s2, s3 = spectra_factory("parent_mass", parent_mass)
    similarity_score = ParentMassMatch(tolerance=tolerance)
    scores = similarity_score.matrix([s0, s1], [s2, s3])
    assert np.all(scores == np.array(expected)), "Expected different scores."


def test_parentmass_match_array_symmetric():
    """Test with array and is_symmetric=True."""
    builder = SpectrumBuilder()
    spectrum_1 = builder.with_metadata({"parent_mass": 100.0}).build()
    spectrum_2 = builder.with_metadata({"parent_mass": 101.0}).build()
    spectrum_3 = builder.with_metadata({"parent_mass": 99.95}).build()
    spectrum_4 = builder.with_metadata({"parent_mass": 98.0}).build()

    spectra = [spectrum_1, spectrum_2, spectrum_3, spectrum_4]
    similarity_score = ParentMassMatch()
    scores = similarity_score.matrix(spectra, spectra, is_symmetric=True)
    scores2 = similarity_score.matrix(spectra, spectra, is_symmetric=False)

    assert np.all(scores == scores2), "Expected identical scores"
    assert np.all(
        scores
        == np.array(
            [
                [True, False, True, False],
                [False, True, False, False],
                [True, False, True, False],
                [False, False, False, True],
            ]
        )
    ), "Expected different scores"
