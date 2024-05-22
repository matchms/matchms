import numpy as np
import pytest
from matchms import calculate_scores
from matchms.similarity.MetadataMatch import MetadataMatch
from tests.builder_Spectrum import SpectrumBuilder


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


def test_metadata_match_strings(spectrums):
    """Test basic metadata matching between string entries."""
    references = spectrums[:2]
    queries = spectrums[2:]

    similarity_score = MetadataMatch(field="instrument_type")
    scores = calculate_scores(references, queries, similarity_score)
    assert np.all(scores.scores.to_array() == [[1, 0], [0, 0]]), "Expected different scores."


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
    assert np.all(scores.scores.to_array() == [[0, 0], [0, 0]]), "Expected different scores."
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
    assert np.all(scores.scores.to_array().tolist() == expected), "Expected different scores."


def test_metadata_match_invalid_array_type(spectrums):
    """Test value error if array_type is not 'numpy' or 'sparse' in metadata matching."""
    references = spectrums[:2]
    queries = spectrums[2:]

    similarity_score = MetadataMatch(field="instrument_type")

    try:
        scores = calculate_scores(references, queries, similarity_score, array_type = "scipy")
    except ValueError as e:
        assert str(e) == "array_type must be 'numpy' or 'sparse'.", "The error message did not match the expected output"
