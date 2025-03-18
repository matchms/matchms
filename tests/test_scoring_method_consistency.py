"""Test if the available methods to compute scores give the same results.
Scoring functions can be called individually (`.pair()`) or with `.sparse_array()`
or `.matrix()`. The latter are optimized for bigger datasets but should yield
the same results.

Some scoring methods will be omitted because they would require additional processing
or depencendies (e.g. fingerprint similarity).
"""
import os
import numpy as np
import pytest
import matchms.filtering as msfilter
import matchms.similarity as mssimilarity
from matchms import Pipeline
from matchms.importing import load_from_json
from matchms.Pipeline import create_workflow
from matchms.similarity.Mask import Mask

module_root = os.path.dirname(__file__)
json_file = os.path.join(module_root, "testdata", "gnps_spectra.json")
# Get similarity measures available in matchms
_score_functions = {
    "cosinegreedy": [mssimilarity.CosineGreedy, {}],
    "cosinehungarian": [mssimilarity.CosineHungarian, {}],
    #"parentmassmatch": [mssimilarity.ParentMassMatch, {}],
    "precursormzmatch": [mssimilarity.PrecursorMzMatch, {}],
    "metadatamatch": [mssimilarity.MetadataMatch, {"field": "spectrum_status"}],
    "modifiedcosine": [mssimilarity.ModifiedCosine, {}]}


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


@pytest.mark.parametrize("similarity_measure", list(_score_functions.values()))
def test_all_scores_and_methods(spectra, similarity_measure):
    """Compute similarites between all spectra and compare across different method calls.
    .pair() will be compared to .matrix() results
    .matrix() results will be compared to .sparse_array() results
    """
    similarity_measure = similarity_measure[0](**similarity_measure[1])

    # Run pair() method
    computed_scores_pair = np.zeros((len(spectra), len(spectra)))
    for i, spec1 in enumerate(spectra):
        for j, spec2 in enumerate(spectra):
            score = similarity_measure.pair(spec1, spec2)
            if isinstance(score, np.ndarray) and score.dtype.names is not None:
                score = score[score.dtype.names[0]]
            computed_scores_pair[i, j] = score

    # Run matrix() method
    computed_scores_matrix = similarity_measure.matrix(spectra, spectra)
    if computed_scores_matrix.dtype.names is not None:
        computed_scores_matrix = computed_scores_matrix[computed_scores_matrix.dtype.names[0]]
    assert np.allclose(computed_scores_pair, computed_scores_matrix)

    # Run sparse_array() method
    idx_row, idx_col = np.where(computed_scores_matrix)
    mask = Mask(idx_row, idx_col)
    computed_scores_sparse = similarity_measure.sparse_array(spectra, spectra,
                                                             mask)
    if computed_scores_sparse.dtype.names is None:
        assert np.allclose(computed_scores_sparse, computed_scores_matrix[idx_row, idx_col])
    else:
        assert np.allclose(computed_scores_sparse[computed_scores_sparse.dtype.names[0]],
            computed_scores_matrix[idx_row, idx_col])


@pytest.mark.parametrize("similarity_measure", list(_score_functions.values()))
def test_consistency_scoring_and_pipeline(spectra, similarity_measure):
    scoring_method = similarity_measure[0](**similarity_measure[1])
    # Run matrix() method
    computed_scores_matrix = scoring_method.matrix(spectra, spectra)

    # Run pipeline
    workflow = create_workflow(query_filters=[["add_parent_mass"], ["normalize_intensities"]],
                               score_computations=[similarity_measure])
    pipeline = Pipeline(workflow)
    pipeline.run(json_file)

    if computed_scores_matrix.dtype.names is None:
        assert np.allclose(pipeline.scores.to_array(), computed_scores_matrix)
    else:
        assert np.allclose(pipeline.scores.to_array(pipeline.scores.score_names[0]),
            computed_scores_matrix[computed_scores_matrix.dtype.names[0]])
