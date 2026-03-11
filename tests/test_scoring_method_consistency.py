"""Test if the available methods to compute scores give the same results.

Scoring functions can be called individually (`.pair()`) or with `.matrix()`.
Some scoring functions also support `.sparse_matrix()`. The optimized methods
should yield the same results as repeated pairwise computations.

Some scoring methods are omitted because they would require additional processing
or dependencies (for example fingerprint similarity).
"""
import os
import numpy as np
import pytest
import matchms.filtering as msfilter
import matchms.similarity as mssimilarity
from matchms import Pipeline
from matchms.importing import load_from_json
from matchms.Pipeline import create_workflow
from matchms.similarity.BaseSimilarity import BaseSimilarityWithSparse


module_root = os.path.dirname(__file__)
json_file = os.path.join(module_root, "testdata", "gnps_spectra.json")

# Get similarity measures available in matchms
_score_functions = {
    "modifiedcosinegreedy": [mssimilarity.ModifiedCosineGreedy, {}],
    "cosinegreedy": [mssimilarity.CosineGreedy, {}],
    "cosinehungarian": [mssimilarity.CosineHungarian, {}],
    "modifiedcosinehungarian": [mssimilarity.ModifiedCosineHungarian, {}],
    "precursormzmatch": [mssimilarity.PrecursorMzMatch, {}],
    "metadatamatch": [mssimilarity.MetadataMatch, {"field": "spectrum_status"}],
}


@pytest.fixture
def spectra():
    """Import spectra and apply basic filters."""
    def processing(s):
        s = msfilter.default_filters(s)
        s = msfilter.add_parent_mass(s)
        s = msfilter.normalize_intensities(s)
        return s

    spectra = load_from_json(json_file)
    spectra = [processing(s) for s in spectra]
    spectra = [s for s in spectra if s is not None]
    return spectra


def _primary_field(similarity_measure) -> str:
    """Return the primary score field for a similarity measure."""
    return similarity_measure.score_fields[0]


def _score_to_scalar(score, similarity_measure):
    """Convert a pair() result to the primary scalar score."""
    if isinstance(score, np.ndarray) and score.dtype.names is not None:
        return score[_primary_field(similarity_measure)]
    return score


@pytest.mark.parametrize("similarity_measure", list(_score_functions.values()))
def test_all_scores_and_methods(spectra, similarity_measure):
    """Compute similarities between all spectra and compare across different method calls.

    pair() will be compared to matrix() results.
    If sparse_matrix() is supported, its results will also be compared to matrix().
    """
    similarity_measure = similarity_measure[0](**similarity_measure[1])
    primary_field = _primary_field(similarity_measure)

    # Run pair() method
    computed_scores_pair = np.zeros((len(spectra), len(spectra)))
    for i, spec1 in enumerate(spectra):
        for j, spec2 in enumerate(spectra):
            score = similarity_measure.pair(spec1, spec2)
            computed_scores_pair[i, j] = _score_to_scalar(score, similarity_measure)

    # Run matrix() method
    computed_scores_matrix = similarity_measure.matrix(spectra)
    computed_scores_matrix_array = computed_scores_matrix.to_array(primary_field)

    assert np.allclose(computed_scores_pair, computed_scores_matrix_array)

    # Run sparse_matrix() method when supported
    if isinstance(similarity_measure, BaseSimilarityWithSparse):
        computed_scores_sparse = similarity_measure.sparse_matrix(spectra)
        computed_scores_sparse_array = computed_scores_sparse.to_array(primary_field)
        assert np.allclose(computed_scores_sparse_array, computed_scores_matrix_array)


@pytest.mark.parametrize("similarity_measure", list(_score_functions.values()))
def test_consistency_scoring_and_pipeline(spectra, similarity_measure):
    """Compare direct scoring with pipeline scoring output."""
    scoring_method = similarity_measure[0](**similarity_measure[1])
    primary_field = _primary_field(scoring_method)

    # Run matrix() method
    computed_scores_matrix = scoring_method.matrix(spectra)
    computed_scores_matrix_array = computed_scores_matrix.to_array(primary_field)

    # Run pipeline
    workflow = create_workflow(
        query_filters=[["add_parent_mass"], ["normalize_intensities"]],
        score_computations=[similarity_measure],
    )
    pipeline = Pipeline(workflow)
    pipeline.run(json_file)

    assert np.allclose(
        pipeline.scores.to_array(primary_field),
        computed_scores_matrix_array,
    )
