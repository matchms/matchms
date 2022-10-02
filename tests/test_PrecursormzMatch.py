import numpy as np
import pytest
from matchms.similarity import PrecursorMzMatch
from matchms.similarity.PrecursorMzMatch import (
    precursormz_scores, precursormz_scores_ppm, precursormz_scores_symmetric,
    precursormz_scores_symmetric_ppm)
from .builder_Spectrum import SpectrumBuilder, spectra_factory


@pytest.mark.parametrize('precursor_mz, tolerance, tolerance_type, expected', [
    [[100.0, 101.0], 0.1, "Dalton", False],
    [[100.0, 101.0], 2.0, "Dalton", True],
    [[600.0, 600.001], 2.0, "ppm", True]
])
def test_precursormz_match_parameterized(precursor_mz, tolerance, tolerance_type, expected):
    s0, s1 = spectra_factory('precursor_mz', precursor_mz)
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


@pytest.mark.parametrize('precursor_mz, tolerance, tolerance_type, expected', [
    [[100.0, 101.0, 99.0, 98.0], 0.1, "Dalton", [[False, False], [False, False]]],
    [[100.0, 101.0, 99.0, 98.0], 2.0, "Dalton", [[True, True], [True, False]]],
    [[100.0, 101.0, 99.99, 98.0], 101.0, "ppm", [[True, False], [False, False]]]
])
def test_precursormz_match_array_parameterized(precursor_mz, tolerance, tolerance_type, expected):
    s0, s1, s2, s3 = spectra_factory('precursor_mz', precursor_mz)
    similarity_score = PrecursorMzMatch(tolerance=tolerance, tolerance_type=tolerance_type)
    scores = similarity_score.matrix([s0, s1], [s2, s3])
    assert np.all(scores == np.array(expected)), "Expected different scores."


@pytest.mark.parametrize('precursor_mz, tolerance, tolerance_type, expected', [
    [[100.0, 101.0, 99.95, 98.0], 0.1, "Dalton",
     [[True, False, True, False], [False, True, False, False], [True, False, True, False], [False, False, False, True]]],
    [[100.0, 101.0, 99.99999, 99.9], 5.0, "ppm",
     [[True, False, True, False], [False, True, False, False], [True, False, True, False], [False, False, False, True]]]
])
def test_precursormz_match_array_symmetric_parameterized(precursor_mz, tolerance, tolerance_type, expected):
    spectrums = spectra_factory('precursor_mz', precursor_mz)
    similarity_score = PrecursorMzMatch(tolerance=tolerance, tolerance_type=tolerance_type)
    scores = similarity_score.matrix(spectrums, spectrums, is_symmetric=True)
    scores2 = similarity_score.matrix(spectrums, spectrums, is_symmetric=False)

    assert np.all(scores == scores2), "Expected identical scores"
    assert np.all(scores == np.array(expected)), "Expected different scores"


@pytest.mark.parametrize("numba_compiled", [True, False])
def test_precursormz_scores(numba_compiled):
    """Test the underlying score function (pure Python and numba compiled)."""
    precursors_ref = np.asarray([101, 200, 300])
    precursors_query = np.asarray([100, 301])
    if numba_compiled:
        scores = precursormz_scores(precursors_ref, precursors_query, tolerance=2.0)
    else:
        scores = precursormz_scores.py_func(precursors_ref, precursors_query, tolerance=2.0)
    assert np.all(scores == np.array([[1., 0.],
                                            [0., 0.],
                                            [0., 1.]])), "Expected different scores."


@pytest.mark.parametrize("numba_compiled", [True, False])
def test_precursormz_scores_symmetric(numba_compiled):
    """Test the underlying score function (non-compiled)."""
    precursors = np.asarray([101, 100, 200])
    if numba_compiled:
        scores = precursormz_scores_symmetric(precursors, precursors, tolerance=2.0)
    else:
        scores = precursormz_scores_symmetric.py_func(precursors, precursors, tolerance=2.0)
    assert np.all(scores == np.array([[1., 1., 0.],
                                            [1., 1., 0.],
                                            [0., 0., 1.]])), "Expected different scores."


@pytest.mark.parametrize("numba_compiled", [True, False])
def test_precursormz_scores_ppm(numba_compiled):
    """Test the underlying score function (pure Python and numba compiled)."""
    precursors_ref = np.asarray([100.00001, 200, 300])
    precursors_query = np.asarray([100, 300.00001])
    if numba_compiled:
        scores = precursormz_scores_ppm(precursors_ref, precursors_query, tolerance_ppm=2.0)
    else:
        scores = precursormz_scores_ppm.py_func(precursors_ref, precursors_query, tolerance_ppm=2.0)
    assert np.all(scores == np.array([[1., 0.],
                                            [0., 0.],
                                            [0., 1.]])), "Expected different scores."


@pytest.mark.parametrize("numba_compiled", [True, False])
def test_precursormz_scores_symmetric_ppm(numba_compiled):
    """Test the underlying score function (non-compiled)."""
    precursors = np.asarray([100.00001, 100, 200])
    if numba_compiled:
        scores = precursormz_scores_symmetric_ppm(precursors, precursors, tolerance_ppm=2.0)
    else:
        scores = precursormz_scores_symmetric_ppm.py_func(precursors, precursors, tolerance_ppm=2.0)
    assert np.all(scores == np.array([[1., 1., 0.],
                                            [1., 1., 0.],
                                            [0., 0., 1.]])), "Expected different scores."
