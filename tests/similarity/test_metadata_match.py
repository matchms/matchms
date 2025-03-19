import numpy as np
import pytest
from matchms import calculate_scores
from matchms.similarity.MetadataMatch import MetadataMatch
from tests.builder_Spectrum import SpectrumBuilder


@pytest.fixture
def spectra():
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


def test_metadata_match_strings(spectra):
    """Test basic metadata matching between string entries."""
    references = spectra[:2]
    queries = spectra[2:]

    similarity_score = MetadataMatch(field="instrument_type")
    scores = calculate_scores(references, queries, similarity_score)
    assert np.all(scores.scores.to_array() == [[1, 0], [0, 0]]), "Expected different scores."


def test_metadata_match_strings_pair(spectra):
    """Test basic metadata matching between string entries."""
    similarity_score = MetadataMatch(field="instrument_type")
    score = similarity_score.pair(spectra[0], spectra[1])
    assert score == np.array(False, dtype=bool), "Expected different score."
    score = similarity_score.pair(spectra[0], spectra[3])
    assert score == np.array(False, dtype=bool), "Expected different score."
    score = similarity_score.pair(spectra[0], spectra[2])
    assert score == np.array(True, dtype=bool), "Expected different score."


def test_metadata_match_strings_wrong_method(spectra, caplog):
    """Test basic metadata matching between string entries."""
    references = spectra[:2]
    queries = spectra[2:]

    similarity_score = MetadataMatch(field="instrument_type", matching_type="difference")
    scores = calculate_scores(references, queries, similarity_score)
    assert np.all(scores.scores.to_array() == [[0, 0], [0, 0]]), "Expected different scores."
    msg = "not compatible with 'difference' method"
    assert msg in caplog.text


def test_metadata_match_numerical_pair(spectra):
    """Test basic metadata matching between string entries."""
    similarity_score = MetadataMatch(field="retention_time",
                                     matching_type="difference",
                                     tolerance=0.6)
    score = similarity_score.pair(spectra[0], spectra[1])
    assert score == 1, "Expected different score."


@pytest.mark.parametrize("tolerance, expected", [
    [1.0, [[0, 1], [0, 0]]],
    [2.0, [[0, 1], [0, 1]]],
    [10.0, [[1, 1], [1, 1]]],
    [0.1, [[0, 0], [0, 0]]]
])
def test_metadata_match_numerical(spectra, tolerance, expected):
    """Test basic metadata matching between numerical entries."""
    references = spectra[:2]
    queries = spectra[2:]

    similarity_score = MetadataMatch(field="retention_time",
                                     matching_type="difference", tolerance=tolerance)
    scores = calculate_scores(references, queries, similarity_score)
    assert np.all(scores.scores.to_array().tolist() == expected), "Expected different scores."
