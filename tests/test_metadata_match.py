import numpy as np
from .builder_Spectrum import SpectrumBuilder
from matchms import calculate_scores
from matchms.similarity.MetadataMatch import MetadataMatch


def test_metadata_match_strings():
    metadata1 = {"instrument_type": "orbitrap"}
    metadata2 = {"instrument_type": "qtof"}
    metadata3 = {"instrument_type": "orbitrap"}
    metadata4 = {"instrument_type": "unknown"}

    spectrum_1 = SpectrumBuilder().with_metadata(metadata1).build()
    spectrum_2 = SpectrumBuilder().with_metadata(metadata2).build()
    spectrum_3 = SpectrumBuilder().with_metadata(metadata3).build()
    spectrum_4 = SpectrumBuilder().with_metadata(metadata4).build()
    references = [spectrum_1, spectrum_2]
    queries = [spectrum_3, spectrum_4]

    similarity_score = MetadataMatch(field="instrument_type")
    scores = calculate_scores(references, queries, similarity_score)
    assert np.all(scores.scores == [[1, 0], [0, 0]]), "Expected different scores."
