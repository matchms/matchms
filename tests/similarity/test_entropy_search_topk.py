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
from matchms.similarity.entropy_search import EntropySearch, _prepare_packed, _validate_separation
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
@pytest.mark.numba_coverage
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


@pytest.mark.numba_coverage
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


@pytest.mark.numba_coverage
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


@pytest.mark.numba_coverage
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


@pytest.mark.numba_coverage
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

# ---------------------------------------------------------------------------
# EntropySearch: configuration, preparation, indexing and dense search
# ---------------------------------------------------------------------------


@pytest.mark.numba_coverage
class TestEntropySearchConfiguration:
    @pytest.mark.parametrize(
        "options, message",
        [
            ({"matching_mode": "hybrid"}, "fragment matching"),
            ({"use_ppm": True}, "fragment matching"),
            ({"tolerance": -0.01}, "finite"),
            ({"tolerance": np.inf}, "finite"),
            ({"tolerance": np.nan}, "finite"),
            ({"tolerance": 0.1, "max_tolerance": 0.01}, "finite"),
            ({"max_tolerance": np.inf}, "finite"),
            ({"peak_separation": "ignore"}, "peak_separation"),
            ({"noise_cutoff": -0.01}, "noise_cutoff"),
            ({"noise_cutoff": 1.01}, "noise_cutoff"),
            ({"noise_cutoff": np.nan}, "noise_cutoff"),
            ({"offset_to_precursor": np.inf}, "offset_to_precursor"),
            ({"dtype": np.float16}, "dtype"),
            ({"index_step": -1.0}, "index_step"),
            ({"index_step": np.nan}, "index_step"),
        ],
    )
    def test_invalid_constructor_settings(self, options, message):
        with pytest.raises(ValueError, match=message):
            EntropySearch(**options)

    @pytest.mark.parametrize("dtype", [np.float32, np.float64])
    def test_settings_serialization_and_zero_tolerance(self, dtype):
        scorer = EntropySearch(
            tolerance=0,
            max_tolerance=0,
            peak_separation="raise",
            noise_cutoff=None,
            remove_precursor=False,
            intensity_weighting=False,
            index_step=0,
            dtype=dtype,
        )
        settings = scorer.to_dict()
        assert settings["__Similarity__"] == "EntropySearch"
        assert settings["dtype"] == np.dtype(dtype).name
        assert settings["max_tolerance"] == 0
        assert settings["noise_cutoff"] == 0
        assert scorer.score_datatype == np.dtype(dtype)
        assert scorer._minimum_gap() == 0
        assert dict(scorer._settings()) == scorer._config()
        assert scorer._config()["normalize_to_half"] is True

    def test_minimum_gap_uses_maximum_tolerance(self):
        scorer = _make_scorer()
        gap = scorer._minimum_gap()
        assert gap > 2 * scorer.max_tolerance
        assert gap < 2.001 * scorer.max_tolerance

    def test_precursor_conversion_and_missing_values(self):
        convert = EntropySearch._precursors
        np.testing.assert_array_equal(np.isnan(convert(None, 3)), [True] * 3)
        precursors = convert(["500.5", "missing", 0, -2, np.inf, np.nan], 6)
        assert precursors[0] == pytest.approx(500.5)
        assert np.all(np.isnan(precursors[1:]))
        with pytest.raises(ValueError, match="one value per spectrum"):
            convert([500.0], 2)
        with pytest.raises(ValueError, match="one value per spectrum"):
            convert([[500.0, 510.0]], 1)


@pytest.mark.numba_coverage
class TestEntropySearchPreparation:
    @pytest.mark.parametrize("dtype", [np.float32, np.float64])
    def test_pack_multiple_rows_normalizes_and_keeps_inputs_immutable(self, dtype):
        scorer = _make_scorer(dtype)
        rows = [_peaks((100, 2), (200, 2)), _peaks(), _peaks((150, 9))]
        original = [r.copy() for r in rows]
        packed = scorer.prepare_peak_arrays(rows, precursor_mz=[400, None, 450])
        assert packed.n_specs == 3 and packed.n_peaks == 3 and len(packed) == 3
        assert packed.n_input_peaks == 3
        assert packed.n_filtered_peaks == packed.n_merged_peaks == 0
        np.testing.assert_array_equal(packed.spec_offsets, [0, 2, 2, 3])
        np.testing.assert_array_equal(packed.spec_mz, [100, 200, 150])
        np.testing.assert_allclose(packed.spec_int, [0.25, 0.25, 0.5])
        assert packed.spec_int.dtype == np.dtype(dtype)
        assert np.isnan(packed.precursor_mz[1])
        for array in (packed.spec_offsets, packed.spec_mz, packed.spec_int, packed.precursor_mz):
            assert not array.flags.writeable
            with pytest.raises(ValueError):
                array.flat[0] = 123.0
        for before, after in zip(original, rows, strict=True):
            np.testing.assert_array_equal(before, after)

    @pytest.mark.parametrize("bad", [np.empty(0), np.ones((2, 3)), np.ones((2, 1))])
    def test_peak_arrays_require_two_columns(self, bad):
        with pytest.raises(ValueError, match="shape"):
            _make_scorer().prepare_peak_arrays([bad])

    @pytest.mark.parametrize(
        "offsets",
        [
            np.array([], dtype=np.int64),
            np.array([1, 2]),
            np.array([0, 2, 1]),
            np.array([0.0, 2.0]),
            np.array([[0, 2]]),
        ],
    )
    def test_invalid_offsets(self, offsets):
        with pytest.raises(ValueError, match="Peak offsets"):
            _make_scorer()._prepare_arrays(offsets, [100, 200], [1, 1], [500])

    @pytest.mark.parametrize(
        "offsets, mz, intensity",
        [
            ([0, 2], [100], [1]),
            ([0, 1], [100], [1, 2]),
            ([0, 1], [[100]], [1]),
        ],
    )
    def test_invalid_packed_shapes(self, offsets, mz, intensity):
        with pytest.raises(ValueError, match="incompatible shapes"):
            _make_scorer()._prepare_arrays(offsets, mz, intensity, [500])

    def test_empty_input_and_empty_spectra(self):
        scorer = _make_scorer()
        empty = scorer.prepare_peak_arrays([])
        assert empty.n_specs == 0 and empty.n_peaks == 0
        np.testing.assert_array_equal(empty.spec_offsets, [0])
        rows = scorer.prepare_peak_arrays([_peaks(), _peaks()])
        assert rows.n_specs == 2 and rows.n_peaks == 0
        np.testing.assert_array_equal(rows.spec_offsets, [0, 0, 0])

    def test_preparation_filters_invalid_zero_noise_and_precursor_peaks(self):
        scorer = EntropySearch(
            tolerance=0.01,
            max_tolerance=0.02,
            peak_separation="raise",
            noise_cutoff=0.2,
            remove_precursor=True,
            offset_to_precursor=-1.6,
            intensity_weighting=False,
            dtype=np.float64,
        )
        row = _peaks((100, 10), (120, 0), (150, 1), (198.4, 8), (199, 100))
        packed = scorer.prepare_peak_arrays([row], precursor_mz=[200])
        np.testing.assert_array_equal(packed.spec_mz, [100, 198.4])
        np.testing.assert_allclose(packed.spec_int, [5 / 18, 4 / 18])
        assert packed.n_input_peaks == 5
        assert packed.n_filtered_peaks == 3
        assert packed.n_merged_peaks == 0

    def test_invalid_peaks_and_missing_precursor_do_not_poison_packed_arrays(self):
        scorer = EntropySearch(tolerance=0.01, noise_cutoff=0, remove_precursor=True)
        raw = _peaks((100, 3), (110, 0), (120, -1), (130, np.nan), (np.inf, 2), (np.nan, 1), (-1, 1))
        packed = scorer.prepare_peak_arrays([raw])
        np.testing.assert_array_equal(packed.spec_mz, [100])
        np.testing.assert_allclose(packed.spec_int, [0.5])
        assert packed.n_filtered_peaks == 6
        no_precursor = scorer.prepare_peak_arrays([_peaks((100, 1), (900, 1))])
        np.testing.assert_array_equal(no_precursor.spec_mz, [100, 900])

    @pytest.mark.parametrize("policy", ["merge", "raise"])
    def test_unsorted_peaks_are_rejected_even_when_merging(self, policy):
        scorer = EntropySearch(tolerance=0.02, peak_separation=policy)
        with pytest.raises(ValueError, match="row 0.*unsorted"):
            scorer.prepare_peak_arrays([_peaks((200, 1), (100, 2))])

    def test_close_peaks_raise_or_merge_around_strongest_peak(self):
        raw = _peaks((100, 1), (100.015, 3), (100.03, 1), (200, 2))
        with pytest.raises(ValueError, match="row 0.*separated"):
            _make_scorer().prepare_peak_arrays([raw])
        merged = EntropySearch(
            tolerance=0.01,
            max_tolerance=0.02,
            peak_separation="merge",
            intensity_weighting=False,
        ).prepare_peak_arrays([raw])
        np.testing.assert_allclose(merged.spec_mz, [100.015, 200], rtol=1e-7, atol=1e-6)
        np.testing.assert_allclose(merged.spec_int, [5 / 7 * 0.5, 2 / 7 * 0.5])
        assert merged.n_merged_peaks == 2

    def test_merge_ties_prefer_lower_mz_and_inclusive_boundary_rejected(self):
        scorer = EntropySearch(tolerance=0.0625, max_tolerance=0.0625, peak_separation="merge")
        merged = scorer.prepare_peak_arrays([_peaks((100, 1), (100.125, 1))])
        np.testing.assert_array_equal(merged.spec_mz, [100.0])
        assert merged.n_merged_peaks == 1
        with pytest.raises(ValueError, match="separated"):
            EntropySearch(
                tolerance=0.0625, max_tolerance=0.0625, peak_separation="raise"
            ).prepare_peak_arrays([_peaks((100, 1), (100.125, 1))])

    def test_weighted_and_unweighted_intensities(self):
        raw = [_peaks((100, 9), (200, 1))]
        plain = EntropySearch(tolerance=0.01, intensity_weighting=False).prepare_peak_arrays(raw)
        weighted = EntropySearch(tolerance=0.01, intensity_weighting=True).prepare_peak_arrays(raw)
        np.testing.assert_allclose(plain.spec_int, [0.45, 0.05])
        np.testing.assert_allclose(weighted.spec_int.sum(), 0.5)
        assert weighted.spec_int[1] > plain.spec_int[1]
        assert weighted.spec_int[0] < plain.spec_int[0]

    def test_intensity_rescaling_avoids_float64_overflow(self):
        packed = EntropySearch(tolerance=0.01, dtype=np.float64).prepare_peak_arrays(
            [_peaks((100, 1e308), (200, 1e308))]
        )
        np.testing.assert_allclose(packed.spec_int, [0.25, 0.25])
        assert np.all(np.isfinite(packed.spec_int))

    def test_underflowed_float32_intensities_are_removed(self):
        smallest = float(np.nextafter(np.float32(0), np.float32(1)))
        scorer = EntropySearch(
            tolerance=0.01, dtype=np.float32, noise_cutoff=0,
            intensity_weighting=False,
        )
        prepared = scorer.prepare_peak_arrays([_peaks((100, smallest), (200, 1.0))])
        np.testing.assert_array_equal(prepared.spec_mz, [200])
        np.testing.assert_allclose(prepared.spec_int, [0.5])
        assert prepared.n_filtered_peaks == 1

    def test_prepared_slices_rebase_offsets_share_peaks_and_check_bounds(self):
        prepared = _make_scorer().prepare_peak_arrays(
            [_peaks((100, 1), (200, 1)), _peaks(), _peaks((300, 2))]
        )
        selected = prepared.slice(1, 3)
        np.testing.assert_array_equal(selected.spec_offsets, [0, 0, 1])
        np.testing.assert_array_equal(selected.spec_mz, [300])
        assert np.shares_memory(prepared.spec_mz, selected.spec_mz)
        assert selected.n_input_peaks == 1
        assert selected.n_filtered_peaks == selected.n_merged_peaks == 0
        assert not selected.spec_mz.flags.writeable
        for first, last in [(-1, 1), (2, 1), (0, 4)]:
            with pytest.raises(ValueError, match="slice"):
                prepared.slice(first, last)

    def test_direct_packed_kernel_reports_specific_failure_codes(self):
        scorer = _make_scorer()

        def call(mz, intensity, merge):
            return _prepare_packed(
                np.array([0, len(mz)], np.int64),
                np.asarray(mz, np.float64),
                np.asarray(intensity, np.float64),
                np.array([500.0]),
                0.0, False, -1.6, 0.05, scorer._minimum_gap(), merge, False,
            )

        offsets, mz, intensity, filtered, merged, row, code = call([100, 200], [1, 2], False)
        np.testing.assert_array_equal(offsets, [0, 2])
        np.testing.assert_allclose(intensity, [1 / 6, 1 / 3])
        assert (filtered, merged, row, code) == (0, 0, -1, 0)
        assert call([200, 100], [1, 1], True)[-2:] == (0, 1)
        assert call([100, 100.01], [1, 1], False)[-2:] == (0, 2)

    @pytest.mark.parametrize(
        "mass, intensity, expected_bad_row",
        [
            ([100, 200, 300], [0.25, 0.25, 0.25], 1),  # wrong sum in second row
            ([100, 100.01, 300], [0.25, 0.25, 0.5], 0),  # close peaks
            ([100, 200, 300], [0.0, 0.5, 0.5], 0),
            ([100, 200, 300], [np.nan, 0.5, 0.5], 0),
            ([100, 200, 300], [0.1, 0.1, 0.5], 0),
        ],
    )
    def test_validate_stored_spectra_rejects_invalid_rows(self, mass, intensity, expected_bad_row):
        offsets = np.array([0, 2, 3], np.int64)
        assert _validate_separation(offsets, np.asarray(mass), np.asarray(intensity), 0.04) == expected_bad_row

    def test_validate_stored_spectra_accepts_empty_rows(self):
        offsets = np.array([0, 2, 2, 3], np.int64)
        mz = np.array([100.0, 200.0, 300.0])
        intensity = np.array([0.25, 0.25, 0.5])
        assert _validate_separation(offsets, mz, intensity, 0.04) == -1


@pytest.mark.numba_coverage
class TestEntropySearchIndexAndDenseScoring:
    @pytest.mark.parametrize("with_directory", [True, False])
    def test_index_statistics_directory_cache_and_persistence(self, tmp_path, with_directory):
        scorer = _make_scorer(np.float64, index_step=0.1 if with_directory else 0)
        prepared = scorer.prepare_peak_arrays([_peaks((100, 1), (200, 3)), _peaks(), _peaks((300, 1))])
        index = scorer.build_index_prepared(prepared)
        first = scorer.prime_index(index)
        assert scorer.prime_index(index) is first
        assert not first.terms.flags.writeable
        assert not first.starts.flags.writeable
        assert bool(first.starts.size) is with_directory
        stats = scorer.index_statistics(index)
        assert stats["n_reference"] == 3
        assert stats["n_peaks"] == 3
        assert stats["dtype"] == "float64"
        assert stats["directory_entries"] == first.starts.size
        assert stats["directory_bytes"] == first.starts.nbytes
        assert stats["cached_entropy_bytes"] == first.terms.nbytes
        assert stats["prepared_statistics"]["input_peaks"] == 3
        path = tmp_path / "entropy_index.npz"
        scorer.save_index(index, path)
        restored = scorer.load_index(path)
        assert restored is not index
        np.testing.assert_array_equal(restored.peaks_mz, index.peaks_mz)
        original = scorer.search_prepared(prepared, index).to_array()
        loaded = scorer.search_prepared(prepared, restored).to_array()
        np.testing.assert_allclose(original, loaded, atol=1e-14)

    def test_index_and_prepared_require_compatible_settings(self):
        scorer = _make_scorer()
        prepared = scorer.prepare_peak_arrays([_peaks((100, 1))])
        index = scorer.build_index_prepared(prepared)
        different = EntropySearch(
            tolerance=0.01, max_tolerance=0.02, peak_separation="raise", noise_cutoff=0,
            remove_precursor=False, intensity_weighting=False,
        )
        with pytest.raises(ValueError, match="prepare_queries"):
            different.build_index_prepared(prepared)
        with pytest.raises(ValueError, match="compatible"):
            different.search_prepared(different.prepare_peak_arrays([_peaks((100, 1))]), index)
        with pytest.raises(ValueError, match="prepare_queries"):
            scorer.search_prepared(object(), index)
        with pytest.raises(TypeError, match="FlashIndex"):
            scorer.search_prepared(prepared, object())

    def test_prepared_library_reused_with_smaller_query_tolerance(self):
        broad = _make_scorer(np.float64)
        narrow = EntropySearch(
            tolerance=0.005, max_tolerance=0.02, peak_separation="raise",
            noise_cutoff=0, remove_precursor=False, dtype=np.float64,
        )
        lib = broad.build_index_prepared(broad.prepare_peak_arrays([_peaks((100, 1))]))
        queries = broad.prepare_peak_arrays([_peaks((100.007, 1))])
        assert broad.search_prepared(queries, lib).to_array()[0, 0] == pytest.approx(1.0)
        assert narrow.search_prepared(queries, lib).to_array()[0, 0] == 0.0

    def test_binary_search_and_window_boundaries_without_directory(self):
        from matchms.similarity.entropy_search import _bisect, _query_windows

        library = np.array([99.875, 100.0, 100.125, 100.125, 150.0, 150.125])
        values = [99.8, 99.875, 100, 100.125, 110, 150.125, 160]
        # Deliberately invalid directory brackets must trigger full-search fallback.
        broken_directory = np.zeros(12, dtype=np.int64)
        for value in values:
            for right in (False, True):
                expected = int(np.searchsorted(library, value, side="right" if right else "left"))
                assert _bisect(library, value, np.empty(0, np.int64), 0.0, 1.0, right) == expected
                assert _bisect(library, value, broken_directory, 80.0, 5.0, right) == expected
        query = np.array([100.0, 150.0, 180.0])
        found = _query_windows(query, library, 0.125, np.empty(0, np.int64), 0.0, 1.0)
        np.testing.assert_array_equal(found, [[0, 4], [4, 6], [6, 6]])

    def test_query_windows_trim_adjacent_floating_point_endpoints(self):
        from matchms.similarity.entropy_search import _query_windows

        lower = np.nextafter(99.875, -np.inf)
        upper = np.nextafter(100.125, np.inf)
        library = np.array([lower, 99.875, 100.125, upper])
        bounds = _query_windows(
            np.array([100.0]), library, 0.125,
            np.empty(0, dtype=np.int64), 0.0, 1.0,
        )
        np.testing.assert_array_equal(bounds, [[1, 3]])

    @pytest.mark.parametrize("dtype", [np.float32, np.float64])
    @pytest.mark.parametrize("index_step", [0, 0.25])
    def test_dense_search_matrix_pair_and_threading(self, dtype, index_step):
        from matchms.spectrum import Spectrum

        scorer = _make_scorer(dtype, index_step=index_step)

        def spectrum(row):
            return Spectrum(mz=row[:, 0], intensities=row[:, 1], metadata_harmonization=False)

        references = [_peaks((100, 1), (200, 1)), _peaks((100, 1), (300, 1)), _peaks()]
        queries = [_peaks((100, 1), (200, 1)), _peaks((900, 1)), _peaks()]
        index = scorer.build_index([spectrum(r) for r in references])
        prepared = scorer.prepare_peak_arrays(queries)
        serial = scorer.search_prepared(prepared, index, n_jobs=1).to_array()
        threaded = scorer.search_prepared(prepared, index, n_jobs=2).to_array()
        assert serial.dtype == np.dtype(dtype)
        assert serial.shape == (3, 3)
        np.testing.assert_allclose(serial, threaded, atol=1e-7)
        np.testing.assert_allclose(serial[0], [1.0, 0.5, 0.0], atol=1e-6)
        np.testing.assert_array_equal(serial[1:], np.zeros((2, 3)))
        matrix = scorer.matrix([spectrum(r) for r in queries], [spectrum(r) for r in references], progress_bar=False)
        np.testing.assert_allclose(matrix.to_array(), serial, atol=1e-6)
        assert scorer.pair(spectrum(queries[0]), spectrum(references[0])).dtype == np.dtype(dtype)
        assert scorer.pair(spectrum(queries[0]), spectrum(references[0])) == pytest.approx(1.0, abs=1e-6)

    def test_self_matrix_and_unprepared_search(self):
        from matchms.spectrum import Spectrum

        scorer = _make_scorer(np.float64)
        one = Spectrum(mz=np.array([100.0, 200.0]), intensities=np.array([1.0, 1.0]), metadata_harmonization=False)
        other = Spectrum(mz=np.array([100.0, 300.0]), intensities=np.array([1.0, 1.0]), metadata_harmonization=False)
        matrix = scorer.matrix([one, other], progress_bar=False).to_array()
        np.testing.assert_allclose(matrix, [[1, 0.5], [0.5, 1]], atol=1e-12)
        index = scorer.build_index([one, other])
        found = scorer.search([one, other], index, progress_bar=False).to_array()
        np.testing.assert_allclose(found, matrix, atol=1e-12)

    def test_prepare_queries_csr_and_non_csr_paths(self):
        from types import SimpleNamespace
        import pandas as pd

        scorer = _make_scorer(np.float64)
        # These two duck-typed collections exercise the packed CSR fast path and
        # the generic get_row fallback without involving a full collection constructor.
        csr = SimpleNamespace(
            format="csr",
            indptr=np.array([0, 2, 3], dtype=np.int64),
            indices=np.array([100, 200, 150], dtype=np.int32),
            data=np.array([1.0, 1.0, 3.0]),
        )
        fragments_csr = SimpleNamespace(array=csr, bin_to_mz=lambda indices: indices.astype(float))
        metadata = pd.DataFrame({"precursor_mz": [500.0, 600.0]})
        fake_csr = SimpleNamespace(fragments=fragments_csr, metadata=metadata)
        direct = scorer.prepare_queries(fake_csr)
        np.testing.assert_array_equal(direct.spec_offsets, [0, 2, 3])
        np.testing.assert_array_equal(direct.spec_mz, [100, 200, 150])
        np.testing.assert_allclose(direct.spec_int, [0.25, 0.25, 0.5])

        class Unpacked:
            array = SimpleNamespace(format="coo")

            @staticmethod
            def get_row(row):
                if row == 0:
                    return np.array([100.0, 200.0]), np.array([1.0, 1.0])
                return np.array([150.0]), np.array([3.0])

        fake_unpacked = SimpleNamespace(fragments=Unpacked(), metadata=metadata)
        # The fallback implementation asks the collection for its row count.
        class Collection:
            fragments = fake_unpacked.fragments
            metadata = fake_unpacked.metadata

            def __len__(self):
                return 2

        unpacked = scorer.prepare_queries(Collection())
        np.testing.assert_array_equal(unpacked.spec_mz, direct.spec_mz)
        np.testing.assert_allclose(unpacked.spec_int, direct.spec_int)

    def test_long_windows_use_small_vector_chunks(self, monkeypatch):
        import matchms.similarity.entropy_search as search_module

        # >512 hits forces the vector path. A small chunk size exercises its
        # bounded-memory loop without allocating tens of thousands of references.
        monkeypatch.setattr(search_module, "_VECTOR_CHUNK_SIZE", 19)
        scorer = _make_scorer(np.float64, index_step=0)
        reference = scorer.prepare_peak_arrays([_peaks((100, 1)) for _ in range(520)])
        query = scorer.prepare_peak_arrays([_peaks((100, 1))])
        index = scorer.build_index_prepared(reference)
        values = scorer.search_prepared(query, index).to_array()
        np.testing.assert_allclose(values, np.ones((1, 520)), atol=1e-12)

    @pytest.mark.parametrize("n_jobs", [0, -2, True, 1.5, "2"])
    def test_invalid_dense_worker_counts(self, n_jobs):
        scorer = _make_scorer()
        index = scorer.build_index_prepared(scorer.prepare_peak_arrays([_peaks((100, 1))]))
        query = scorer.prepare_peak_arrays([_peaks((100, 1))])
        with pytest.raises(ValueError, match="n_jobs"):
            scorer.search_prepared(query, index, n_jobs=n_jobs)

    def test_dense_empty_queries_and_all_empty_library(self):
        scorer = _make_scorer()
        library = scorer.build_index_prepared(scorer.prepare_peak_arrays([_peaks(), _peaks()]))
        empty_query = scorer.prepare_peak_arrays([])
        assert scorer.search_prepared(empty_query, library).to_array().shape == (0, 2)
        nonempty_query = scorer.prepare_peak_arrays([_peaks((100, 1))])
        np.testing.assert_array_equal(scorer.search_prepared(nonempty_query, library).to_array(), [[0, 0]])

