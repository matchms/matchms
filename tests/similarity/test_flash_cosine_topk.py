from __future__ import annotations
import numpy as np
import pytest
from matchms.similarity._flash_cosine_topk import _CosineTopKWorkspace
from matchms.similarity._flash_prepared import PreparedSpectra, readonly
from matchms.similarity._flash_topk_common import _MAX_EPOCH
from matchms.similarity.flash_similarity import CosineFlash
from matchms.similarity.topk_results import TopKResults


def _prepared(scorer, rows, precursors):
    mz, intensities, offsets, norms = [], [], [0], []
    for pairs in rows:
        row = np.asarray(pairs, dtype=np.float64).reshape((-1, 2))
        assert np.all(row[:, 1] >= 0) and np.all(row[1:, 0] >= row[:-1, 0])
        values = np.asarray(row[:, 1], dtype=scorer.dtype)
        mz.extend(row[:, 0])
        intensities.extend(values)
        offsets.append(len(mz))
        norms.append(np.linalg.norm(values.astype(np.float64)))
    return PreparedSpectra(
        spec_offsets=readonly(np.array(offsets, dtype=np.int64)),
        spec_mz=readonly(np.array(mz, dtype=scorer.dtype)),
        spec_int=readonly(np.array(intensities, dtype=scorer.dtype)),
        precursor_mz=readonly(np.array(precursors, dtype=np.float64)),
        spec_l2=readonly(np.array(norms, dtype=np.float64)),
        dtype=scorer.dtype,
        settings=scorer._settings(),
    )


def _scorer(mode="fragment", dtype=np.float32, *, ppm=False, gate=None,
            identity_ppm=False):
    return CosineFlash(
        matching_mode=mode, tolerance=55 if ppm else 0.045,
        use_ppm=ppm, identity_precursor_tolerance=gate,
        identity_use_ppm=identity_ppm, noise_cutoff=0,
        remove_precursor=False, dtype=dtype,
    )


def _expected(dense_scores, dense_matches, k, min_score):
    n_queries, _ = dense_scores.shape
    ids = np.full((n_queries, k), -1, dtype=np.int64)
    vals = np.zeros((n_queries, k), dtype=dense_scores.dtype)
    matches = np.zeros((n_queries, k), dtype=np.int32)
    counts = np.zeros(n_queries, dtype=np.intp)
    for q, row in enumerate(dense_scores):
        qualified = np.flatnonzero(row > min_score)
        sorted_ids = sorted(qualified, key=lambda i: (-row[i], int(i)))[:k]
        length = len(sorted_ids)
        counts[q] = length
        ids[q, :length] = sorted_ids
        vals[q, :length] = row[sorted_ids]
        matches[q, :length] = dense_matches[q, sorted_ids]
    return ids, vals, matches, counts


def _check_dense(scorer, ref, query, *, k=3, min_score=0.0, n_jobs=1,
                 batch_size=None, fields=None):
    index = scorer.build_index_prepared(ref)
    dense = scorer.search_prepared(query, index, n_jobs=1)
    expected = _expected(dense._data["score"], dense._data["matches"],
                         k, min_score)
    result = scorer.search_prepared_topk(
        query, index, k=k, min_score=min_score, n_jobs=n_jobs,
        batch_size=batch_size, score_fields=fields,
    )
    assert isinstance(result, TopKResults)
    assert result.shape == (query.n_specs, k)
    assert result.n_references == ref.n_specs
    np.testing.assert_array_equal(result.indices, expected[0])
    np.testing.assert_array_equal(result.scores, expected[1])
    np.testing.assert_array_equal(result.valid_counts, expected[3])
    if fields is None or "matches" in fields:
        assert result.matches is not None
        np.testing.assert_array_equal(result.matches, expected[2])
        for row in range(query.n_specs):
            got_ids, got_scores, got_matches = result.get_hits_with_matches(row)
            size = int(expected[3][row])
            np.testing.assert_array_equal(got_ids, expected[0][row, :size])
            np.testing.assert_array_equal(got_scores, expected[1][row, :size])
            np.testing.assert_array_equal(got_matches, expected[2][row, :size])
    else:
        assert result.matches is None
        with pytest.raises(ValueError, match="match counts"):
            result.get_hits_with_matches(0)
    return result


def _random_rows(seed, n, *, max_peaks=16):
    rng = np.random.default_rng(seed)
    rows = []
    for _ in range(n):
        n_peaks = int(rng.integers(0, max_peaks + 1))
        mz = rng.choice(np.arange(30, 400, 2), size=n_peaks, replace=True).astype(float)
        mz += rng.choice([0, 0.012, 0.035, 0.055], size=n_peaks)
        intensity = rng.choice([0.0, 0.15, 0.4, 0.7, 1.0, 1.3], size=n_peaks)
        order = np.argsort(mz, kind="stable")
        rows.append(np.column_stack((mz[order], intensity[order])))
    return rows


@pytest.mark.parametrize("mode", ["fragment", "neutral_loss", "hybrid"])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("ppm", [False, True])
@pytest.mark.parametrize("gate", [None, 0.1])
def test_randomized_exact_greedy_and_counts(mode, dtype, ppm, gate):
    scorer = _scorer(mode, dtype, ppm=ppm, gate=gate)
    refs = _random_rows(302, 24)
    queries = _random_rows(123, 9)
    # Intentionally reuse precursors to trigger hybrid direct-only gating.
    p_refs = np.array([260, 270, 280, np.nan, 260.03] * 5)[:len(refs)]
    p_queries = np.array([260, 260.03, np.nan, 280, 270] * 2)[:len(queries)]
    _check_dense(scorer, _prepared(scorer, refs, p_refs),
                 _prepared(scorer, queries, p_queries),
                 k=7, min_score=0.0, n_jobs=3, batch_size=2)


@pytest.mark.parametrize("mode", ["fragment", "neutral_loss", "hybrid"])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("k,min_score", [(1, 0), (2, .2), (10, .7), (3, 1.1)])
def test_competing_matches_ties_and_threshold(mode, dtype, k, min_score):
    scorer = _scorer(mode, dtype)
    refs = [
        [(100, 1), (100.01, 2), (200, 1)],
        [(100, 1), (100.01, 2), (200, 1)],
        [(100, 1), (200, 2)],
        [(99.99, 1), (100.01, 3), (150, 1)],
        [(150, 2)],
        [],
        [(380, 1)],
    ]
    queries = [
        [(100, 1), (100.02, 3), (200.02, 1)],
        [(99.99, 2), (200, 1)],
        [(150, 1)],
        [(999, 1)],
        [],
    ]
    _check_dense(scorer, _prepared(scorer, refs, [300, 300, 310, 300, 280, 300, 380]),
                 _prepared(scorer, queries, [300, 310, 280, np.nan, 300]),
                 k=k, min_score=min_score, n_jobs=2, batch_size=1)


@pytest.mark.parametrize("mode", ["fragment", "neutral_loss", "hybrid"])
def test_precursor_identity_ppm(mode):
    scorer = _scorer(mode, ppm=False, gate=30, identity_ppm=True)
    refs = _prepared(scorer, [[(100, 1)], [(100, 1)], [(100, 1)]],
                     [300, 300.006, 302])
    queries = _prepared(scorer, [[(100, 1)], [(100, 1)]],
                        [300, np.nan])
    _check_dense(scorer, refs, queries, k=4)


def test_hybrid_query_and_library_bitsets_exceed_64_peaks():
    scorer = _scorer("hybrid", np.float64)
    mz = np.arange(100, 380, dtype=float)
    ref_rows = [np.column_stack((mz[:190], np.ones(190))),
                np.column_stack((mz[2:200], np.arange(198) + 1)),
                [(100, 1), (150, 2)], []]
    query_rows = [np.column_stack((mz[:255], np.ones(255))),
                  [(100, 1)], np.column_stack((mz[40:180], np.ones(140)))]
    _check_dense(scorer, _prepared(scorer, ref_rows, [480, 490, 480, 480]),
                 _prepared(scorer, query_rows, [480, 490, 480]),
                 k=5, n_jobs=2, batch_size=1)


def test_matches_field_can_be_omitted():
    scorer = _scorer("hybrid")
    refs = _prepared(scorer, [[(100, 1)], [(100, 1)]], [250, 252])
    queries = _prepared(scorer, [[(100, 1)]], [250])
    _check_dense(scorer, refs, queries, fields=("score",))


def test_does_not_call_dense_scoring(monkeypatch):
    scorer = _scorer("hybrid")
    refs = _prepared(scorer, [[(100, 1)], [(100, 1)]], [250, 252])
    queries = _prepared(scorer, [[(100, 1)]], [250])
    index = scorer.build_index_prepared(refs)
    monkeypatch.setattr(scorer, "_score_prepared", lambda *a: pytest.fail("dense called"))
    got = scorer.search_prepared_topk(queries, index, k=1)
    assert got.valid_counts.tolist() == [1]
    assert got.matches[0, 0] == 1


def test_public_method_prepares_queries(monkeypatch):
    scorer = _scorer("fragment")
    refs = _prepared(scorer, [[(100, 1)]], [250])
    queries = _prepared(scorer, [[(100, 1)]], [250])
    index = scorer.build_index_prepared(refs)
    monkeypatch.setattr(scorer, "prepare_queries", lambda _: queries)
    got = scorer.search_topk(object(), index, k=2, progress_bar=False)
    assert got.indices.tolist() == [[0, -1]]


def test_saved_index_load_matches_original(tmp_path):
    scorer = _scorer("hybrid")
    refs = _prepared(scorer, [[(100, 1), (100.02, 1)], [(200, 1)]], [300, 301])
    queries = _prepared(scorer, [[(100, 1), (199.99, 1)]], [300])
    index = scorer.build_index_prepared(refs)
    path = tmp_path / "cosine.npz"
    scorer.save_index(index, path)
    loaded = scorer.load_index(path)
    old = scorer.search_prepared_topk(queries, index, k=3)
    new = scorer.search_prepared_topk(queries, loaded, k=3)
    np.testing.assert_array_equal(old.indices, new.indices)
    np.testing.assert_array_equal(old.scores, new.scores)
    np.testing.assert_array_equal(old.matches, new.matches)


def test_empty_cases_and_untouched_padding():
    scorer = _scorer("hybrid")
    empty = _prepared(scorer, [], [])
    nonempty = _prepared(scorer, [[(100, 1)]], [300])
    no_peaks = _prepared(scorer, [[]], [300])
    for refs, queries in [(empty, nonempty), (nonempty, empty),
                          (nonempty, no_peaks)]:
        _check_dense(scorer, refs, queries, k=4)


def test_epoch_wraparound_on_reused_workspace():
    state = _CosineTopKWorkspace(11, 40, np.float32, hybrid=True)
    state.epoch = _MAX_EPOCH
    state.epochs[:] = 77
    state.peak_word_epochs[:] = 77
    assert state.next_epoch() == 1
    assert not np.any(state.epochs)
    assert not np.any(state.peak_word_epochs)


def test_invalid_score_fields_and_parameters():
    scorer = _scorer("fragment")
    refs = _prepared(scorer, [[(100, 1)]], [250])
    index = scorer.build_index_prepared(refs)
    with pytest.raises(ValueError, match="requires the 'score'"):
        scorer.search_prepared_topk(refs, index, score_fields=("matches",))
    for k in (-1, 0, True, 1.2):
        with pytest.raises((TypeError, ValueError)):
            scorer.search_prepared_topk(refs, index, k=k)
    for cutoff in (-1, float("nan"), float("inf"), True):
        with pytest.raises((TypeError, ValueError)):
            scorer.search_prepared_topk(refs, index, min_score=cutoff)
    for batch_size in (-1, 0, True, 1.2):
        with pytest.raises((TypeError, ValueError)):
            scorer.search_prepared_topk(refs, index, batch_size=batch_size)


def test_topk_results_optional_match_counts_validation_and_immutability():
    from matchms.similarity.topk_results import TopKResults
    obj = TopKResults(
        np.array([[0, -1]], dtype=np.int32),
        np.array([[.7, 0]], dtype=np.float32),
        np.array([1], dtype=np.int32), 1,
        np.array([[3, 0]], dtype=np.int32),
    )
    assert obj.get_hits_with_matches(0)[2].tolist() == [3]
    assert not obj.matches.flags.writeable
    with pytest.raises(ValueError, match="zero"):
        TopKResults(obj.indices, obj.scores, obj.valid_counts, 1,
                    np.array([[3, 1]], dtype=np.int32))
