import numpy as np
import pytest
from .builder_Spectrum import SpectrumBuilder
from matchms import calculate_scores
from matchms.similarity.MetadataMatch import MetadataMatch


@pytest.fixture
def spectrums():
    metadata1 = {"instrument_type": "orbitrap",
                 "retention_time": 100.}
    metadata2 = {"instrument_type": "qtof",
                 "retention_time": 100.5}
    metadata3 = {"instrument_type": "orbitrap",
                 "retention_time": 105.}
    metadata4 = {"instrument_type": "unknown",
                 "retention_time": 99.1}

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
    assert np.all(scores.scores == [[1, 0], [0, 0]]), "Expected different scores."


def test_metadata_match_strings_wrong_method(spectrums, caplog):
    """Test basic metadata matching between string entries."""
    references = spectrums[:2]
    queries = spectrums[2:]

    similarity_score = MetadataMatch(field="instrument_type", matching_type="difference")
    scores = calculate_scores(references, queries, similarity_score)
    assert np.all(scores.scores == [[0, 0], [0, 0]]), "Expected different scores."
    msg = "Matching_type was set to 'difference' but no difference could be computed between"
    assert msg in caplog.text


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
