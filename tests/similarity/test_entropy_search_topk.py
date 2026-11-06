from __future__ import annotations
import numpy as np
import pytest
from matchms.similarity._entropy_search_topk import (
    _MAX_EPOCH,
    _scatter_add_topk,
    _TopKWorkspace,
)
from matchms.similarity._topk import (
    select_topk_candidates,
    select_topk_row,
)
from matchms.similarity.entropy_search import EntropySearch
from matchms.similarity.topk_results import TopKResults


def _peaks(*pairs):
    return np.asarray(pairs, dtype=np.float64).reshape((-1, 2))


def _make_scorer(dtype=np.float32, *, index_step=0.01):
    return EntropySearch(
        tolerance=0.01,
        max_tolerance=0.02,
        peak_separation="raise",
        noise_cutoff=0,
        remove_precursor=False,
        dtype=dtype,
        index_step=index_step,
    )


def _library_and_queries():
    references = [
        _peaks((100, 3), (150, 1)),
        _peaks((100, 3), (150, 1)),  # exact tie: ID 0 precedes ID 1
        _peaks((100, 1)),
        _peaks((105, 1), (250, 2)),
        _peaks((800, 2)),
        _peaks(),
        _peaks((101, 2), (200, 1)),
    ]
    queries = [
        _peaks((100, 3), (150, 1)),
        _peaks((105, 1), (250, 2)),
        _peaks((999, 1)),
        _peaks(),
        _peaks((100, 5)),
        _peaks((800, 1)),
        _peaks((101, 3), (200, 2)),
    ]
    return references, queries


def _expected_from_dense(matrix, k, min_score):
    expected_ids = np.full((matrix.shape[0], k), -1, dtype=np.int64)
    expected_scores = np.zeros((matrix.shape[0], k), dtype=matrix.dtype)
    counts = np.zeros(matrix.shape[0], dtype=np.int64)
    for i, dense_row in enumerate(matrix):
        candidate_ids = np.flatnonzero(dense_row > min_score)
        order = np.lexsort((candidate_ids, -dense_row[candidate_ids]))
        best_ids = candidate_ids[order[:k]]
        counts[i] = len(best_ids)
        expected_ids[i, : len(best_ids)] = best_ids
        expected_scores[i, : len(best_ids)] = dense_row[best_ids]
    return expected_ids, expected_scores, counts


def _assert_matches_dense(results: TopKResults, matrix, k, min_score):
    ids, scores, counts = _expected_from_dense(matrix, k, min_score)
    assert results.shape == (matrix.shape[0], k)
    assert results.n_references == matrix.shape[1]
    assert results.n_queries == matrix.shape[0]
    assert results.k == k
    assert len(results) == matrix.shape[0]
    assert results.scores.dtype == matrix.dtype
    expected_id_dtype = np.int32 if matrix.shape[1] <= np.iinfo(np.int32).max else np.int64
    assert results.indices.dtype == expected_id_dtype
    np.testing.assert_array_equal(results.indices, ids)
    np.testing.assert_array_equal(results.scores, scores)
    np.testing.assert_array_equal(results.valid_counts, counts)
    for row in range(results.n_queries):
        n = int(counts[row])
        hit_ids, hit_scores = results.get_hits(row)
        np.testing.assert_array_equal(hit_ids, ids[row, :n])
        np.testing.assert_array_equal(hit_scores, scores[row, :n])


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("k", [1, 2, 4, 12])
@pytest.mark.parametrize("min_score", [0, 0.05, 0.8, 1.1])
def test_topk_exactly_matches_dense_with_ties_padding_and_threshold(dtype, k, min_score):
    scorer = _make_scorer(dtype)
    references, queries = _library_and_queries()
    index = scorer.build_index_prepared(scorer.prepare_peak_arrays(references))
    prepared = scorer.prepare_peak_arrays(queries)
    dense = scorer.search_prepared(prepared, index, progress_bar=False).to_array()
    result = scorer.search_prepared_topk(
        prepared,
        index,
        k=k,
        min_score=min_score,
        batch_size=3,
        progress_bar=False,
        n_jobs=2,
    )
    _assert_matches_dense(result, dense, k, min_score)


def test_search_topk_prepares_queries_and_agrees_with_prepared_path():
    scorer = _make_scorer()
    references, queries = _library_and_queries()
    index = scorer.build_index_prepared(scorer.prepare_peak_arrays(references))
    # The public prepare_queries path accepts matchms Spectrum instances.
    from matchms.spectrum import Spectrum

    spectra = [Spectrum(mz=row[:, 0], intensities=row[:, 1], metadata_harmonization=False) for row in queries]
    result = scorer.search_topk(spectra, index, k=4, batch_size=2, progress_bar=False)
    prepared = scorer.prepare_peak_arrays(queries)
    expected = scorer.search_prepared_topk(prepared, index, k=4, batch_size=2)
    np.testing.assert_array_equal(result.indices, expected.indices)
    np.testing.assert_array_equal(result.scores, expected.scores)
    np.testing.assert_array_equal(result.valid_counts, expected.valid_counts)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_large_matching_windows_use_existing_vector_kernel(dtype):
    scorer = _make_scorer(dtype, index_step=0)
    references = [_peaks((100, 1)) for _ in range(700)]
    queries = [_peaks((100, 1)), _peaks((200, 1)), _peaks((100, 2))]
    index = scorer.build_index_prepared(scorer.prepare_peak_arrays(references))
    prepared = scorer.prepare_peak_arrays(queries)
    dense = scorer.search_prepared(prepared, index, n_jobs=2).to_array()
    result = scorer.search_prepared_topk(
        prepared,
        index,
        k=15,
        min_score=0.5,
        batch_size=2,
        n_jobs=2,
    )
    _assert_matches_dense(result, dense, 15, 0.5)
    np.testing.assert_array_equal(result.indices[0], np.arange(15))
    assert result.valid_counts[1] == 0


def test_candidate_topk_never_calls_dense_scoring(monkeypatch):
    scorer = _make_scorer()
    references, queries = _library_and_queries()
    queries = queries * 3
    index = scorer.build_index_prepared(scorer.prepare_peak_arrays(references))
    prepared = scorer.prepare_peak_arrays(queries)

    def forbidden_dense_score(*args, **kwargs):
        raise AssertionError("top-k called the dense scorer")

    monkeypatch.setattr(scorer, "_score_into", forbidden_dense_score)
    result = scorer.search_prepared_topk(prepared, index, k=2, batch_size=4, n_jobs=2)
    assert result.shape == (21, 2)
    assert result.valid_counts[0] == 2


def test_batch_size_controls_scheduling_not_dense_buffer(monkeypatch):
    import matchms.similarity.entropy_search as entropy_search_module

    scorer = _make_scorer()
    references, queries = _library_and_queries()
    index = scorer.build_index_prepared(scorer.prepare_peak_arrays(references))
    prepared = scorer.prepare_peak_arrays(queries * 4)
    calls = []
    score_block = entropy_search_module._score_topk_block

    def wrapped_score_block(*args):
        # Function signature: (..., first_row, last_row, k, min_score, ...)
        first, last = args[7], args[8]
        calls.append((first, last))
        return score_block(*args)

    monkeypatch.setattr(entropy_search_module, "_score_topk_block", wrapped_score_block)
    result = scorer.search_prepared_topk(prepared, index, k=3, batch_size=3, n_jobs=2)
    assert result.shape == (28, 3)
    assert sum(b - a for a, b in calls) == 28
    assert sorted(r for a, b in calls for r in range(a, b)) == list(range(28))
    assert all(b - a <= 3 for a, b in calls)


def test_empty_queries_empty_index_and_zero_hits():
    scorer = _make_scorer()
    empty = scorer.prepare_peak_arrays([])
    reference = scorer.prepare_peak_arrays([_peaks(), _peaks()])
    no_peak_index = scorer.build_index_prepared(reference)
    result = scorer.search_prepared_topk(empty, no_peak_index, k=3)
    assert result.shape == (0, 3)
    assert result.valid_counts.shape == (0,)

    query = scorer.prepare_peak_arrays([_peaks((100, 1)), _peaks()])
    result = scorer.search_prepared_topk(query, no_peak_index, k=3)
    np.testing.assert_array_equal(result.indices, -np.ones((2, 3), dtype=np.int32))
    np.testing.assert_array_equal(result.scores, np.zeros((2, 3), dtype=scorer.dtype))
    np.testing.assert_array_equal(result.valid_counts, [0, 0])


def test_index_round_trip_without_format_change(tmp_path):
    scorer = _make_scorer()
    references, queries = _library_and_queries()
    index = scorer.build_index_prepared(scorer.prepare_peak_arrays(references))
    path = tmp_path / "entropy-search-index.npz"
    scorer.save_index(index, path)
    loaded = scorer.load_index(path)
    prepared = scorer.prepare_peak_arrays(queries)
    original = scorer.search_prepared_topk(prepared, index, k=5, batch_size=2)
    restored = scorer.search_prepared_topk(prepared, loaded, k=5, batch_size=2)
    np.testing.assert_array_equal(original.indices, restored.indices)
    np.testing.assert_array_equal(original.scores, restored.scores)
    np.testing.assert_array_equal(original.valid_counts, restored.valid_counts)


@pytest.mark.parametrize(
    "k, min_score, batch_size, exception",
    [
        (0, 0, None, ValueError),
        (-1, 0, None, ValueError),
        (True, 0, None, TypeError),
        (1.5, 0, None, TypeError),
        (1, -0.1, None, ValueError),
        (1, float("nan"), None, ValueError),
        (1, float("inf"), None, ValueError),
        (1, True, None, TypeError),
        (1, 0, 0, ValueError),
        (1, 0, -2, ValueError),
        (1, 0, 1.2, TypeError),
        (1, 0, False, TypeError),
    ],
)
def test_invalid_topk_parameters(k, min_score, batch_size, exception):
    scorer = _make_scorer()
    index = scorer.build_index_prepared(scorer.prepare_peak_arrays([_peaks((100, 1))]))
    prepared = scorer.prepare_peak_arrays([_peaks((100, 1))])
    with pytest.raises(exception):
        scorer.search_prepared_topk(prepared, index, k=k, min_score=min_score, batch_size=batch_size)


@pytest.mark.parametrize("n_jobs", [0, -2, True, 1.0, "2"])
def test_invalid_worker_count_matches_dense_search(n_jobs):
    scorer = _make_scorer()
    index = scorer.build_index_prepared(scorer.prepare_peak_arrays([_peaks((100, 1))]))
    prepared = scorer.prepare_peak_arrays([_peaks((100, 1))])
    with pytest.raises(ValueError):
        scorer.search_prepared_topk(prepared, index, n_jobs=n_jobs)
    with pytest.raises(ValueError):
        scorer.search_prepared(prepared, index, n_jobs=n_jobs)


def test_selector_uses_deterministic_library_id_for_kth_score_ties():
    row = np.array([0.8, 0.6, 0.6, 0.95, 0.6, 0.0, 0.6], dtype=np.float32)
    for k, expected in [(1, [3]), (2, [3, 0]), (3, [3, 0, 1]), (5, [3, 0, 1, 2, 4])]:
        indices, values = select_topk_row(row, k, 0.0)
        np.testing.assert_array_equal(indices, expected)
        np.testing.assert_array_equal(values, row[indices])


def test_results_are_read_only_and_have_explicit_padding():
    scorer = _make_scorer()
    index = scorer.build_index_prepared(scorer.prepare_peak_arrays([_peaks((100, 1))]))
    result = scorer.search_prepared_topk(
        scorer.prepare_peak_arrays([_peaks((100, 1))]),
        index,
        k=4,
    )
    assert result.shape == (1, 4)
    assert result.valid_counts[0] == 1
    np.testing.assert_array_equal(result.indices[0], [0, -1, -1, -1])
    np.testing.assert_array_equal(result.scores[0], [1.0, 0.0, 0.0, 0.0])
    with pytest.raises(ValueError):
        result.indices[0, 0] = 5
    with pytest.raises(ValueError):
        result.scores[0, 0] = 5
    with pytest.raises(ValueError):
        result.valid_counts[0] = 5


def test_randomized_exact_ranking_regression():
    rng = np.random.default_rng(1234)
    scorer = _make_scorer(np.float32)
    references = []
    for _ in range(191):
        # Distinct mass sites guarantee separation, while repeated sites across
        # spectra produce realistic candidate overlap and score ties.
        masses = np.sort(rng.choice(np.arange(80, 430, 7), size=3, replace=False)).astype(float)
        references.append(np.column_stack((masses, rng.uniform(0.1, 5, 3))))
    queries = []
    for _ in range(29):
        masses = np.sort(rng.choice(np.arange(80, 430, 7), size=4, replace=False)).astype(float)
        queries.append(np.column_stack((masses, rng.uniform(0.1, 5, 4))))
    index = scorer.build_index_prepared(scorer.prepare_peak_arrays(references))
    prepared = scorer.prepare_peak_arrays(queries)
    dense = scorer.search_prepared(prepared, index, n_jobs=2).to_array()
    for k in (1, 3, 9, 23):
        for threshold in (0.0, 0.05, 0.3):
            result = scorer.search_prepared_topk(prepared, index, k=k, min_score=threshold, batch_size=7, n_jobs=2)
            _assert_matches_dense(result, dense, k, threshold)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_dynamic_touched_array_grows_past_chunk_size_and_matches_dense(dtype):
    # More than one 65,536-posting vector chunk and >4096 unique candidates.
    scorer = _make_scorer(dtype, index_step=0)
    references = [_peaks((100, 1)) for _ in range(67_000)]
    queries = [_peaks((100, 1)), _peaks((500, 1))]
    index = scorer.build_index_prepared(scorer.prepare_peak_arrays(references))
    prepared = scorer.prepare_peak_arrays(queries)
    dense = scorer.search_prepared(prepared, index, n_jobs=1).to_array()
    result = scorer.search_prepared_topk(prepared, index, k=13, n_jobs=2, batch_size=1)
    _assert_matches_dense(result, dense, 13, 0)
    np.testing.assert_array_equal(result.indices[0], np.arange(13))
    assert result.valid_counts[1] == 0


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_mixed_short_long_windows_and_repeated_library_updates(dtype):
    scorer = _make_scorer(dtype)
    # Common peaks are long windows. Only a handful match the middle peak.
    # One library ID receives contributions from both matching paths.
    references = [_peaks((100, i % 7 + 1), (200 if i < 9 else 240, 1), (300, i % 5 + 1)) for i in range(641)]
    queries = [
        _peaks((100, 5), (200, 4), (300, 3)),
        _peaks((100, 3), (300, 9)),
        _peaks((200, 2)),
    ]
    index = scorer.build_index_prepared(scorer.prepare_peak_arrays(references))
    prepared = scorer.prepare_peak_arrays(queries)
    dense = scorer.search_prepared(prepared, index, n_jobs=2).to_array()
    for k in (1, 7, 23, 100):
        result = scorer.search_prepared_topk(prepared, index, k=k, n_jobs=3, batch_size=1)
        _assert_matches_dense(result, dense, k, 0)


def test_touched_candidate_selector_resolves_ties_independent_of_encounter_order():
    scores = np.array([0.8, 0.6, 0.6, 0.95, 0.6, 0.0, 0.6], dtype=np.float32)
    for ids in (
        np.array([6, 2, 3, 1, 0, 4], dtype=np.int32),
        np.array([4, 1, 6, 0, 2, 3], dtype=np.int32),
    ):
        for k in (1, 2, 3, 5, 9):
            ids_found, values_found = select_topk_candidates(ids, scores, k, 0)
            full_row = np.zeros_like(scores)
            full_row[ids] = scores[ids]
            expected_ids, expected_values = select_topk_row(full_row, k, 0)
            np.testing.assert_array_equal(ids_found, expected_ids)
            np.testing.assert_array_equal(values_found, expected_values)


def test_scatter_updates_each_touched_candidate_once_with_bounded_growth():
    workspace = _TopKWorkspace(20_000, np.float32)
    assert workspace.touched_ids.size == 4096
    ids = np.arange(19_000, dtype=np.int32)
    ids = np.concatenate((ids, ids[::-1]))
    contributions = np.ones(ids.size, dtype=np.float32)
    epoch = workspace.next_epoch()
    touched, count = _scatter_add_topk(
        workspace.scores,
        workspace.seen_epochs,
        workspace.touched_ids,
        0,
        epoch,
        ids,
        contributions,
    )
    workspace.touched_ids = touched
    assert count == 19_000
    assert 19_000 <= touched.size <= 20_000
    np.testing.assert_array_equal(touched[:count], np.arange(19_000))
    np.testing.assert_array_equal(workspace.scores[:19_000], 2.0)
    # Next query has only one touched reference; stale scores must be cleared.
    epoch = workspace.next_epoch()
    touched, count = _scatter_add_topk(
        workspace.scores,
        workspace.seen_epochs,
        workspace.touched_ids,
        0,
        epoch,
        np.array([14], dtype=np.uint32),
        np.array([0.25], dtype=np.float32),
    )
    assert count == 1
    assert touched[0] == 14
    assert workspace.scores[14] == np.float32(0.25)


def test_epoch_wraparound_does_not_allow_stale_score_reuse():
    workspace = _TopKWorkspace(10, np.float32)
    workspace.seen_epochs[:] = 1
    workspace.scores[:] = 77.0
    workspace.epoch = _MAX_EPOCH
    assert workspace.next_epoch() == 1
    np.testing.assert_array_equal(workspace.seen_epochs, np.zeros(10, dtype=np.uint32))
    touched, count = _scatter_add_topk(
        workspace.scores,
        workspace.seen_epochs,
        workspace.touched_ids,
        0,
        workspace.epoch,
        np.array([3], dtype=np.uint32),
        np.array([0.75], dtype=np.float32),
    )
    assert count == 1
    assert touched[0] == 3
    assert workspace.scores[3] == np.float32(0.75)


@pytest.mark.parametrize("n_jobs", [1, 2, 4, -1])
@pytest.mark.parametrize("batch_size", [None, 1, 3])
def test_rankings_invariant_to_worker_count_and_scheduling(n_jobs, batch_size):
    rng = np.random.default_rng(987)
    scorer = _make_scorer(np.float32)
    references = []
    for _ in range(501):
        masses = np.sort(rng.choice(np.arange(90, 470, 7), size=4, replace=False))
        references.append(np.column_stack((masses, rng.uniform(0.1, 5, 4))))
    queries = []
    for _ in range(19):
        masses = np.sort(rng.choice(np.arange(90, 470, 7), size=4, replace=False))
        queries.append(np.column_stack((masses, rng.uniform(0.1, 5, 4))))
    index = scorer.build_index_prepared(scorer.prepare_peak_arrays(references))
    prepared = scorer.prepare_peak_arrays(queries)
    dense = scorer.search_prepared(prepared, index, n_jobs=1).to_array()
    found = scorer.search_prepared_topk(prepared, index, k=11, min_score=0.04, n_jobs=n_jobs, batch_size=batch_size)
    _assert_matches_dense(found, dense, 11, 0.04)
