import os

import numpy as np
import pytest

from matchms import Pipeline
from matchms.Pipeline import create_workflow


def _score_matrix(scores):
    return scores["score"].to_array() if "score" in scores.score_fields else scores.to_array()


def _matches_matrix(scores):
    if "matches" not in scores.score_fields:
        raise KeyError(f"Expected 'matches' field, available fields: {scores.score_fields}")
    return scores["matches"].to_array()


def test_user_workflow():
    module_root = os.path.join(os.path.dirname(__file__), "..")
    spectra_file = os.path.join(module_root, "tests", "testdata", "pesticides.mgf")

    workflow = create_workflow(
        spectra_1_filters=[
            ["add_parent_mass"],
            ["normalize_intensities"],
            ["select_by_relative_intensity", {"intensity_from": 0.0, "intensity_to": 1.0}],
            ["select_by_mz", {"mz_from": 0, "mz_to": 1000}],
            ["require_minimum_number_of_peaks", {"n_required": 5}],
        ],
        score_computations=[["cosinegreedy", {"tolerance": 0.3}]],
    )

    pipeline = Pipeline(workflow)
    pipeline.run(spectra_file)

    scores = pipeline.scores
    score_arr = _score_matrix(scores)
    matches_arr = _matches_matrix(scores)

    assert scores is not None
    assert pipeline.is_symmetric is True
    assert pipeline.spectra_2 is None

    n_spectra = len(pipeline.spectra_1)
    assert n_spectra > 10
    assert scores.shape == (n_spectra, n_spectra)
    assert score_arr.shape == (n_spectra, n_spectra)
    assert matches_arr.shape == (n_spectra, n_spectra)

    # CosineGreedy self-comparisons should be on the diagonal and essentially perfect.
    assert np.allclose(np.diag(score_arr), 1.0, atol=1e-10)
    assert np.all(np.diag(matches_arr) >= 5)

    # Symmetric all-vs-all workflow should produce symmetric score and matches matrices.
    assert np.allclose(score_arr, score_arr.T, atol=1e-12)
    assert np.array_equal(matches_arr, matches_arr.T)

    # Ignore diagonal and collect strong non-self hits.
    offdiag_mask = ~np.eye(n_spectra, dtype=bool)
    strong_mask = offdiag_mask & (matches_arr >= 20)

    assert np.any(strong_mask), "Expected at least one strong non-self match with >=20 matching peaks."

    pairs = []
    rows, cols = np.where(strong_mask)
    for i, j in zip(rows, cols):
        pairs.append((i, j, score_arr[i, j], matches_arr[i, j]))

    pairs_sorted = sorted(pairs, key=lambda x: (x[2], x[3]), reverse=True)
    top10 = pairs_sorted[:10]

    assert len(top10) == 10

    # Top pairs should be non-self, high-scoring, and satisfy the filter.
    assert all(i != j for i, j, _, _ in top10)
    assert all(m >= 20 for _, _, _, m in top10)

    # Since sorting is by score then matches descending, that ordering should hold.
    top10_keys = [(score, matches) for _, _, score, matches in top10]
    assert top10_keys == sorted(top10_keys, reverse=True)

    # All top scores should be very high in this known dataset/workflow.
    assert min(score for _, _, score, _ in top10) > 0.95

    # Because the run is symmetric, each strong pair should have a mirrored counterpart.
    top10_pairs = {(i, j) for i, j, _, _ in top10}
    mirrored_count = sum((j, i) in top10_pairs for i, j in top10_pairs)
    assert mirrored_count >= 6, "Expected several mirrored high-scoring pairs in the top results."

    # The best non-self hit should be nearly identical and have many matching peaks.
    best_i, best_j, best_score, best_matches = top10[0]
    assert best_i != best_j
    assert best_score > 0.99
    assert best_matches >= 20
