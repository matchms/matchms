import numpy as np
import pytest
from matchms import calculate_scores
from matchms.similarity.MetadataMatch import entries_scores
from matchms.similarity.MetadataMatch import entries_scores_symmetric
from matchms.similarity.MetadataMatch import MetadataMatch
from .builder_Spectrum import SpectrumBuilder


@pytest.fixture
def spectrums():
    metadata1 = {"instrument_type": "orbitrap",
                 "retention_time": 100.}
    metadata2 = {"instrument_type": "qtof",
                 "retention_time": 100.5}
    metadata3 = {"instrument_type": "orbitrap",
                 "retention_time": 105.}
    metadata4 = {"retention_time": 99.1}

    s1 = SpectrumBuilder().with_metadata(metadata1).build()
    s2 = SpectrumBuilder().with_metadata(metadata2).build()
    s3 = SpectrumBuilder().with_metadata(metadata3).build()
    s4 = SpectrumBuilder().with_metadata(metadata4).build()
    return [s1, s2, s3, s4]


@pytest.mark.parametrize("entries1, entries2, result", [
    [np.array([100, 101, 102]), np.array([102, 104]),
     np.array([[1., 0.], [1., 0.], [1., 1.]])],
    [np.array([98, 105.5]), np.array([102, 104]),
     np.array([[0., 0.], [0., 1.]])]
])
def test_entries_scores(entries1, entries2, result):
    scores = entries_scores(entries1, entries2, 2)
    assert np.array_equal(scores, result)
    # non-compiled run
    scores = entries_scores.py_func(entries1, entries2, 2)
    assert np.array_equal(scores, result)


@pytest.mark.parametrize("entries, tolerance, result", [
    [np.array([100, 101, 102]), 0.9,
     np.array([[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]])],
    [np.array([100, 101, 102]), 1.0,
     np.array([[1., 1., 0.], [1., 1., 1.], [0., 1., 1.]])]
])
def test_entries_scores_symmetric(entries, tolerance, result):
    scores = entries_scores_symmetric(entries, entries, tolerance)
    assert np.array_equal(scores, result)
    # non-compiled run
    scores = entries_scores_symmetric.py_func(entries, entries, tolerance)
    assert np.array_equal(scores, result)


def test_metadata_match_strings(spectrums):
    """Test basic metadata matching between string entries."""
    references = spectrums[:2]
    queries = spectrums[2:]

    similarity_score = MetadataMatch(field="instrument_type")
    scores = calculate_scores(references, queries, similarity_score)
    assert np.all(scores.scores == [[1, 0], [0, 0]]), "Expected different scores."


def test_metadata_match_strings_pair(spectrums):
    """Test basic metadata matching between string entries."""
    similarity_score = MetadataMatch(field="instrument_type")
    score = similarity_score.pair(spectrums[0], spectrums[1])
    assert score == np.array(False, dtype=bool), "Expected different score."
    score = similarity_score.pair(spectrums[0], spectrums[3])
    assert score == np.array(False, dtype=bool), "Expected different score."
    score = similarity_score.pair(spectrums[0], spectrums[2])
    assert score == np.array(True, dtype=bool), "Expected different score."


def test_metadata_match_strings_wrong_method(spectrums, caplog):
    """Test basic metadata matching between string entries."""
    references = spectrums[:2]
    queries = spectrums[2:]

    similarity_score = MetadataMatch(field="instrument_type", matching_type="difference")
    scores = calculate_scores(references, queries, similarity_score)
    assert np.all(scores.scores == [[0, 0], [0, 0]]), "Expected different scores."
    msg = "not compatible with 'difference' method"
    assert msg in caplog.text


def test_metadata_match_numerical_pair(spectrums):
    """Test basic metadata matching between string entries."""
    similarity_score = MetadataMatch(field="retention_time",
                                     matching_type="difference",
                                     tolerance=0.6)
    score = similarity_score.pair(spectrums[0], spectrums[1])
    assert score == 1, "Expected different score."


@pytest.mark.parametrize("tolerance, expected", [
    [1.0, [[0, 1], [0, 0]]],
    [2.0, [[0, 1], [0, 1]]],
    [10.0, [[1, 1], [1, 1]]],
    [0.1, [[0, 0], [0, 0]]]
])
def test_metadata_match_numerical(spectrums, tolerance, expected):
    """Test basic metadata matching between numerical entries."""
    references = spectrums[:2]
    queries = spectrums[2:]

    similarity_score = MetadataMatch(field="retention_time",
                                     matching_type="difference", tolerance=tolerance)
    scores = calculate_scores(references, queries, similarity_score)
    assert np.all(scores.scores == expected), "Expected different scores."
