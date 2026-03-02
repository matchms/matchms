import sys
import numpy as np
from matchms import create_scores_object_and_calculate_scores
from matchms.networking.networking_functions import get_top_hits
from matchms.similarity import FingerprintSimilarity
from .test_SimilarityNetwork import create_dummy_spectra


def create_dummy_scores():
    spectra = create_dummy_spectra()
    references = spectra[:5]
    queries = spectra[5:]

    # Create Scores object by calculating dice scores
    similarity_measure = FingerprintSimilarity("dice")
    scores = create_scores_object_and_calculate_scores(references, queries, similarity_measure)
    return scores


def test_get_top_hits_by_references():
    scores = create_dummy_scores()
    idx_ref, scores_ref = get_top_hits(scores, top_n=10, search_by="references")

    expected_scores_ref = {'ref_spec_0': np.array([0.66666667, 0.5]),
                           'ref_spec_1': np.array([0.66666667, 0.5]),
                           'ref_spec_2': np.array([0.66666667, 0.66666667, 0.5]),
                           'ref_spec_3': np.array([0.8, 0.5, 0.5]),
                           'ref_spec_4': np.array([1., 0.8, 0.5])}
    expected_idx_ref = {'ref_spec_0': np.array([0, 2], dtype=np.int64),
                        'ref_spec_1': np.array([1, 2], dtype=np.int64),
                        'ref_spec_2': np.array([1, 0, 2], dtype=np.int64),
                        'ref_spec_3': np.array([2, 1, 0], dtype=np.int64),
                        'ref_spec_4': np.array([0, 2, 1], dtype=np.int64)}
    for key, value in scores_ref.items():
        assert np.allclose(value, expected_scores_ref[key], atol=1e-5), \
            "Expected different selected scores"
    for key, value in idx_ref.items():
        assert np.array_equal(value, expected_idx_ref[key]), \
            "Expected different selected indices"

    # Test lower top_n
    idx_ref, scores_ref = get_top_hits(scores, top_n=2, search_by="references")
    for key, value in scores_ref.items():
        assert np.allclose(value, expected_scores_ref[key][:2], atol=1e-5), \
            "Expected different selected scores"
    for key, value in idx_ref.items():
        assert np.array_equal(value, expected_idx_ref[key][:2]), \
            "Expected different selected indices"


def test_get_top_hits_by_queries():
    scores = create_dummy_scores()
    idx_query, scores_query = get_top_hits(scores, top_n=10, search_by="queries")

    expected_scores_query = {'query_spec_0': np.array([1., 0.66666667, 0.66666667, 0.5]),
                             'query_spec_1': np.array([0.66666667, 0.66666667, 0.5, 0.5]),
                             'query_spec_2': np.array([0.8, 0.8, 0.5, 0.5, 0.5])}
    expected_idx_query = {'query_spec_0': np.array([4, 2, 0, 3], dtype=np.int64),
                          'query_spec_1': np.array([1, 2, 3, 4], dtype=np.int64),
                          'query_spec_2': np.array([4, 3, 2, 1, 0], dtype=np.int64)}

    # this is needed due to different sorting algorithms in case of ties apparently
    if sys.platform == 'darwin':
        expected_idx_query['query_spec_1'] = np.array([2, 1, 4, 3], dtype=np.int64)

    for key, value in scores_query.items():
        assert np.allclose(value, expected_scores_query[key], atol=1e-5), \
            "Expected different selected scores"
    for key, value in idx_query.items():
        assert np.array_equal(value, expected_idx_query[key]), \
            "Expected different selected indices"

    # Test lower top_n
    idx_query, scores_query = get_top_hits(scores, top_n=2, search_by="queries")
    for key, value in scores_query.items():
        assert np.allclose(value, expected_scores_query[key][:2], atol=1e-5), \
            "Expected different selected scores"
    for key, value in idx_query.items():
        assert np.array_equal(value, expected_idx_query[key][:2]), \
            "Expected different selected indices"
