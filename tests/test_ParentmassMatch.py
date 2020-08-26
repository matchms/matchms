import numpy
import pytest
from matchms import Spectrum
from matchms.similarity import ParentmassMatch
from matchms.similarity.ParentmassMatch import parentmass_scores
from matchms.similarity.ParentmassMatch import parentmass_scores_symmetric


def test_parentmass_match():
    "Test with default tolerance."
    spectrum_1 = Spectrum(mz=numpy.array([], dtype="float"),
                          intensities=numpy.array([], dtype="float"),
                          metadata={"parent_mass": 100.0})

    spectrum_2 = Spectrum(mz=numpy.array([], dtype="float"),
                          intensities=numpy.array([], dtype="float"),
                          metadata={"parent_mass": 101.0})

    similarity_score = ParentmassMatch()
    score = similarity_score.compute_score(spectrum_1, spectrum_2)
    assert not score, "Expected different score."


def test_parentmass_match_tolerance2():
    "Test with tolerance > difference."
    spectrum_1 = Spectrum(mz=numpy.array([], dtype="float"),
                          intensities=numpy.array([], dtype="float"),
                          metadata={"parent_mass": 100.0})

    spectrum_2 = Spectrum(mz=numpy.array([], dtype="float"),
                          intensities=numpy.array([], dtype="float"),
                          metadata={"parent_mass": 101.0})

    similarity_score = ParentmassMatch(tolerance=2.0)
    score = similarity_score.compute_score(spectrum_1, spectrum_2)
    assert score, "Expected different score."


def test_parentmass_match_missing_parentmass():
    "Test with missing parentmass."
    spectrum_1 = Spectrum(mz=numpy.array([], dtype="float"),
                          intensities=numpy.array([], dtype="float"),
                          metadata={"parent_mass": 100.0})

    spectrum_2 = Spectrum(mz=numpy.array([], dtype="float"),
                          intensities=numpy.array([], dtype="float"),
                          metadata={})

    similarity_score = ParentmassMatch(tolerance=2.0)

    with pytest.raises(AssertionError) as msg:
        _ = similarity_score.compute_score(spectrum_1, spectrum_2)

    expected_message_part = "Missing parent mass."
    assert expected_message_part in str(msg.value), "Expected particular error message."


def test_parentmass_match_array():
    "Test with array and default tolerance."
    spectrum_1 = Spectrum(mz=numpy.array([], dtype="float"),
                          intensities=numpy.array([], dtype="float"),
                          metadata={"parent_mass": 100.0})

    spectrum_2 = Spectrum(mz=numpy.array([], dtype="float"),
                          intensities=numpy.array([], dtype="float"),
                          metadata={"parent_mass": 101.0})

    spectrum_a = Spectrum(mz=numpy.array([], dtype="float"),
                          intensities=numpy.array([], dtype="float"),
                          metadata={"parent_mass": 99.0})

    spectrum_b = Spectrum(mz=numpy.array([], dtype="float"),
                          intensities=numpy.array([], dtype="float"),
                          metadata={"parent_mass": 98.0})

    similarity_score = ParentmassMatch()
    scores = similarity_score.compute_score_matrix([spectrum_1, spectrum_2],
                                                   [spectrum_a, spectrum_b])
    assert numpy.all(scores == numpy.array([[False, False],
                                            [False, False]])), "Expected different scores."


def test_parentmass_match_tolerance2_array():
    """Test with array and tolerance=2."""
    spectrum_1 = Spectrum(mz=numpy.array([], dtype="float"),
                          intensities=numpy.array([], dtype="float"),
                          metadata={"parent_mass": 100.0})

    spectrum_2 = Spectrum(mz=numpy.array([], dtype="float"),
                          intensities=numpy.array([], dtype="float"),
                          metadata={"parent_mass": 101.0})

    spectrum_a = Spectrum(mz=numpy.array([], dtype="float"),
                          intensities=numpy.array([], dtype="float"),
                          metadata={"parent_mass": 99.0})

    spectrum_b = Spectrum(mz=numpy.array([], dtype="float"),
                          intensities=numpy.array([], dtype="float"),
                          metadata={"parent_mass": 98.0})

    similarity_score = ParentmassMatch(tolerance=2.0)
    scores = similarity_score.compute_score_matrix([spectrum_1, spectrum_2],
                                                   [spectrum_a, spectrum_b])
    assert numpy.all(scores == numpy.array([[True, True],
                                            [True, False]])), "Expected different scores."


def test_parentmass_match_array_symmetric():
    """Test with array and is_symmetric=True."""
    spectrum_1 = Spectrum(mz=numpy.array([], dtype="float"),
                          intensities=numpy.array([], dtype="float"),
                          metadata={"parent_mass": 100.0})

    spectrum_2 = Spectrum(mz=numpy.array([], dtype="float"),
                          intensities=numpy.array([], dtype="float"),
                          metadata={"parent_mass": 101.0})

    spectrum_3 = Spectrum(mz=numpy.array([], dtype="float"),
                          intensities=numpy.array([], dtype="float"),
                          metadata={"parent_mass": 99.95})

    spectrum_4 = Spectrum(mz=numpy.array([], dtype="float"),
                          intensities=numpy.array([], dtype="float"),
                          metadata={"parent_mass": 98.0})

    spectrums = [spectrum_1, spectrum_2, spectrum_3, spectrum_4]
    similarity_score = ParentmassMatch()
    scores = similarity_score.compute_score_matrix(spectrums, spectrums,
                                                   is_symmetric=True)
    scores2 = similarity_score.compute_score_matrix(spectrums, spectrums,
                                                    is_symmetric=False)

    assert numpy.all(scores == scores2), "Expected identical scores"
    assert numpy.all(scores == numpy.array(
        [[True, False, True, False],
         [False, True, False, False],
         [True, False, True, False],
         [False, False, False, True]])), "Expected different scores"


def test_parentmass_scores_compiled():
    """Test the underlying score function (numba compiled)."""
    parentmasses_ref = numpy.asarray([101, 200, 300])
    parentmasses_query = numpy.asarray([100, 301])
    scores = parentmass_scores(parentmasses_ref, parentmasses_query, tolerance=2.0)
    assert numpy.all(scores == numpy.array([[1., 0.],
                                            [0., 0.],
                                            [0., 1.]])), "Expected different scores."


def test_parentmass_scores():
    """Test the underlying score function (non-compiled)."""
    parentmasses_ref = numpy.asarray([101, 200, 300])
    parentmasses_query = numpy.asarray([100, 301])
    scores = parentmass_scores.py_func(parentmasses_ref, parentmasses_query, tolerance=2.0)
    assert numpy.all(scores == numpy.array([[True, False],
                                            [False, False],
                                            [False, True]])), "Expected different scores."


def test_parentmass_scores_symmetric_compliled():
    """Test the underlying score function (non-compiled)."""
    parentmasses = numpy.asarray([101, 100, 200])
    scores = parentmass_scores_symmetric(parentmasses, parentmasses, tolerance=2.0)
    assert numpy.all(scores == numpy.array([[1., 1., 0.],
                                            [1., 1., 0.],
                                            [0., 0., 1.]])), "Expected different scores."


def test_parentmass_scores_symmetric():
    """Test the underlying score function (non-compiled)."""
    parentmasses = numpy.asarray([101, 100, 200])
    scores = parentmass_scores_symmetric.py_func(parentmasses, parentmasses, tolerance=2.0)
    assert numpy.all(scores == numpy.array([[1., 1., 0.],
                                            [1., 1., 0.],
                                            [0., 0., 1.]])), "Expected different scores."
