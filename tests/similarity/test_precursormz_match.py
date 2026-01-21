import numpy as np
import pytest
from matchms.similarity import PrecursorMzMatch
from ..builder_Spectrum import SpectrumBuilder, spectra_factory


# Local global variables to make the test easier to read
T = True
F = False


@pytest.mark.parametrize(
    "precursor_mz, tolerance, tolerance_type, expected",
    [[[100.0, 101.0], 0.1, "Dalton", F], [[100.0, 101.0], 2.0, "Dalton", T], [[600.0, 600.001], 2.0, "ppm", T]],
)
def test_precursormz_match_parameterized(precursor_mz, tolerance, tolerance_type, expected):
    s0, s1 = spectra_factory("precursor_mz", precursor_mz)
    similarity_score = PrecursorMzMatch(tolerance=tolerance, tolerance_type=tolerance_type)
    scores = similarity_score.pair(s0, s1)
    assert np.all(scores == np.array(expected)), "Expected different scores."


def test_precursormz_match_missing_precursormz():
    """Test with missing precursormz."""
    builder = SpectrumBuilder()
    spectrum_1 = builder.with_metadata({"precursor_mz": 100.0}).build()
    spectrum_2 = builder.with_metadata({}).build()

    similarity_score = PrecursorMzMatch(tolerance=2.0)

    with pytest.raises(AssertionError) as msg:
        _ = similarity_score.pair(spectrum_1, spectrum_2)

    expected_message_part = "Missing precursor m/z."
    assert expected_message_part in str(msg.value), "Expected particular error message."


@pytest.mark.parametrize(
    "precursor_mz, tolerance, tolerance_type, expected",
    [
        [[100.0, 101.0, 99.0, 98.0], 0.1, "Dalton", [[F, F], [F, F]]],
        [[100.0, 101.0, 99.0, 98.0], 2.0, "Dalton", [[T, T], [T, F]]],
        [[100.0, 101.0, 99.99, 98.0], 101.0, "ppm", [[T, F], [F, F]]],
    ],
)
def test_precursormz_match_array_parameterized(precursor_mz, tolerance, tolerance_type, expected):
    s0, s1, s2, s3 = spectra_factory("precursor_mz", precursor_mz)
    similarity_score = PrecursorMzMatch(tolerance=tolerance, tolerance_type=tolerance_type)
    scores = similarity_score.matrix([s0, s1], [s2, s3])
    assert np.all(scores == np.array(expected)), "Expected different scores."


@pytest.mark.parametrize(
    "precursor_mz, tolerance, tolerance_type, expected",
    [
        [[100.0, 101.0, 99.95, 98.0], 0.1, "Dalton", [[T, F, T, F], [F, T, F, F], [T, F, T, F], [F, F, F, T]]],
        [[100.0, 101.0, 99.99999, 99.9], 5.0, "ppm", [[T, F, T, F], [F, T, F, F], [T, F, T, F], [F, F, F, T]]],
    ],
)
def test_precursormz_match_array_symmetric_parameterized(precursor_mz, tolerance, tolerance_type, expected):
    spectra = spectra_factory("precursor_mz", precursor_mz)
    similarity_score = PrecursorMzMatch(tolerance=tolerance, tolerance_type=tolerance_type)
    scores = similarity_score.matrix(spectra, spectra, is_symmetric=T)
    scores2 = similarity_score.matrix(spectra, spectra, is_symmetric=F)

    assert np.all(scores == scores2), "Expected identical scores"
    assert np.all(scores == np.array(expected)), "Expected different scores"
