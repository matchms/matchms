from __future__ import annotations
import numpy as np
import pytest
from matchms.similarity._flash_entropy_topk import _EntropyTopKWorkspace
from matchms.similarity._flash_prepared import PreparedSpectra, readonly
from matchms.similarity._flash_topk_common import _MAX_EPOCH
from matchms.similarity.entropy import Entropy
from matchms.similarity.flash_similarity import EntropyFlash
from matchms.similarity.topk_results import TopKResults


pytestmark = pytest.mark.numba_coverage


def _prepared(scorer, spectra, precursors):
    """Build valid, normalized prepared rows with no full Spectrum objects."""
    mz, intensities, offsets = [], [], [0]
    for spectrum in spectra:
        row = np.asarray(spectrum, dtype=np.float64).reshape(-1, 2)
        assert np.all(row[:, 0] >= 0)
        assert np.all(row[:, 1] >= 0)
        assert np.all(row[1:, 0] >= row[:-1, 0])
        values = row[:, 1].copy()
        if len(values) and values.sum():
            values *= 0.5 / values.sum()
        mz.extend(row[:, 0])
        intensities.extend(values)
        offsets.append(len(mz))
    n_specs = len(spectra)
    return PreparedSpectra(
        spec_offsets=readonly(np.asarray(offsets, dtype=np.int64)),
        spec_mz=readonly(np.asarray(mz, dtype=scorer.dtype)),
        spec_int=readonly(np.asarray(intensities, dtype=scorer.dtype)),
        precursor_mz=readonly(np.asarray(precursors, dtype=np.float64)),
        spec_l2=readonly(np.zeros(n_specs, dtype=np.float64)),
        dtype=scorer.dtype,
        settings=scorer._settings(),
    )


def _make_scorer(mode="fragment", dtype=np.float32, ppm=False, identity=None, identity_ppm=False, cls=EntropyFlash):
    return cls(
        matching_mode=mode,
        tolerance=30 if ppm else 0.025,
        use_ppm=ppm,
        remove_precursor=False,
        noise_cutoff=0,
        identity_precursor_tolerance=identity,
        identity_use_ppm=identity_ppm,
        dtype=dtype,
    )


def _expected(matrix, k, min_score):
    ids = np.full((matrix.shape[0], k), -1, dtype=np.int64)
    vals = np.zeros((matrix.shape[0], k), dtype=matrix.dtype)
    counts = np.zeros(matrix.shape[0], dtype=np.intp)
    for r, row in enumerate(matrix):
        qualified = np.flatnonzero(row > min_score)
        ranked = sorted(qualified, key=lambda i: (-row[i], i))[:k]
        counts[r] = len(ranked)
        ids[r, : counts[r]] = ranked
        vals[r, : counts[r]] = row[ranked]
    return ids, vals, counts


def _check_matches_dense(scorer, refs, queries, *, k, min_score=0.0, jobs=1, batch_size=None):
    lib = scorer.build_index_prepared(refs)
    dense = scorer.search_prepared(queries, lib, progress_bar=False, n_jobs=1).to_array()
    got = scorer.search_prepared_topk(
        queries,
        lib,
        k=k,
        min_score=min_score,
        n_jobs=jobs,
        batch_size=batch_size,
        progress_bar=False,
    )
    want_ids, want_scores, want_counts = _expected(dense, k, min_score)
    np.testing.assert_array_equal(got.indices, want_ids)
    np.testing.assert_array_equal(got.scores, want_scores)
    np.testing.assert_array_equal(got.valid_counts, want_counts)
    assert got.n_references == len(refs.spec_offsets) - 1
    assert isinstance(got, TopKResults)
    return got


def _random_rows(seed, n, *, max_peaks=18, allow_empty=True):
    rng = np.random.default_rng(seed)
    rows = []
    for _ in range(n):
        p = int(rng.integers(0 if allow_empty else 1, max_peaks + 1))
        mz = rng.choice(np.arange(70, 470, 2), size=p, replace=True).astype(float)
        mz += rng.choice([0, 0.007, 0.015], size=p)
        intensity = rng.uniform(0.02, 1.0, size=p)
        order = np.argsort(mz, kind="stable")
        rows.append(np.column_stack([mz[order], intensity[order]]))
    return rows


@pytest.mark.parametrize("mode", ["fragment", "neutral_loss", "hybrid"])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("ppm", [False, True])
@pytest.mark.parametrize("identity", [None, 0.05])
def test_exact_randomized(mode, dtype, ppm, identity):
    scorer = _make_scorer(mode, dtype, ppm, identity)
    refs = _random_rows(1024, 75)
    queries = _random_rows(42, 21)
    # Inject cross-spectrum duplicates and ties, including more than one match.
    queries[0] = refs[0].copy()
    queries[1] = refs[2].copy()
    queries[2] = refs[18].copy()
    prec_ref = np.array([500.0 + (i % 3) for i in range(len(refs))])
    prec_q = np.array([500.0 + (i % 3) for i in range(len(queries))])
    prec_ref[[4, 10, 12]] = np.nan
    prec_q[[2, 6]] = np.nan
    reference = _prepared(scorer, refs, prec_ref)
    query = _prepared(scorer, queries, prec_q)
    for k, minimum in [(1, 0.0), (7, 0.05), (100, 0.2)]:
        _check_matches_dense(scorer, reference, query, k=k, min_score=minimum, jobs=2, batch_size=3)


@pytest.mark.parametrize("mode", ["fragment", "neutral_loss", "hybrid"])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_ties_empty_reference_and_query_precursor(mode, dtype):
    scorer = _make_scorer(mode, dtype)
    library = [
        [(100, 3), (150, 1)],
        [(100, 3), (150, 1)],
        [(100, 3), (150, 1)],
        [],
        [(255, 1)],
        [(100.01, 2), (150, 1)],
        [(100, 3), (151, 1)],
    ]
    queries = [[(100, 3), (150, 1)], [], [(777, 1)], [(100, 2)], [(100, 3), (150, 1)]]
    refs = _prepared(scorer, library, [300, 300, 300, np.nan, 300, 300, np.nan])
    qs = _prepared(scorer, queries, [300, np.nan, 300, np.nan, 300])
    _check_matches_dense(scorer, refs, qs, k=3, jobs=1)
    _check_matches_dense(scorer, refs, qs, k=50, jobs=2, batch_size=1)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_hybrid_many_query_peaks_and_overlaps(dtype):
    scorer = _make_scorer("hybrid", dtype)
    bases = np.arange(80.0, 190.0, 1.0)
    queries = [np.column_stack([bases, np.arange(1, bases.size + 1)])]
    references = [queries[0].copy(), queries[0].copy()]
    references += [np.column_stack([bases[::2], np.ones(bases[::2].size)])]
    references += [np.column_stack([bases[::-1], np.ones(bases.size)])[::-1]]
    refs = _prepared(scorer, references, [270.0] * len(references))
    qs = _prepared(scorer, queries, [270.0])
    _check_matches_dense(scorer, refs, qs, k=4, jobs=1)


@pytest.mark.parametrize("mode", ["fragment", "hybrid"])
def test_large_touched_growth_and_multiple_workers(mode):
    scorer = _make_scorer(mode)
    library = [[(100, 1), (150, 1)] for _ in range(650)]
    query = [[(100, 1), (150, 1)] for _ in range(13)]
    refs = _prepared(scorer, library, [200.0] * len(library))
    qs = _prepared(scorer, query, [200.0] * len(query))
    _check_matches_dense(scorer, refs, qs, k=17, jobs=4, batch_size=1)


def test_identity_precursor_ppm_gate():
    scorer = _make_scorer("hybrid", ppm=True, identity=40, identity_ppm=True)
    refs = _prepared(scorer, [[(100, 1)], [(100, 1)], [(100, 1)]], [300, 300.006, 301])
    qs = _prepared(scorer, [[(100, 1)], [(100, 1)]], [300, np.nan])
    _check_matches_dense(scorer, refs, qs, k=3)


def test_hybrid_query_bitmap_grows_between_rows():
    scorer = _make_scorer("hybrid", np.float32)
    long_mz = np.arange(100.0, 370.0, 1.0)
    rows = [
        [(100, 1)],
        np.column_stack([long_mz[:135], np.ones(135)]),
        [(100, 2), (200, 3)],
        np.column_stack([long_mz, np.ones(long_mz.size)]),
        [],
    ]
    refs = _prepared(scorer, [rows[1], rows[3], rows[2]], [500, 500, 500])
    qs = _prepared(scorer, rows, [500] * len(rows))
    _check_matches_dense(scorer, refs, qs, k=3, jobs=1, batch_size=1)


def test_score_conversion_ties_and_threshold():
    scorer = _make_scorer("fragment", np.float32)
    refs = _prepared(scorer, [[(100, 1)], [(100, 1)], [(100, 1)]], [300.0] * 3)
    qs = _prepared(scorer, [[(100, 1)]], [300.0])
    _check_matches_dense(scorer, refs, qs, k=2, min_score=0.0)
    _check_matches_dense(scorer, refs, qs, k=2, min_score=1.0)


def test_index_save_reload(tmp_path):
    scorer = _make_scorer("hybrid", np.float64)
    refs = _prepared(scorer, [[(100, 1), (110, 1)], [(90, 1)], []], [250, 250, np.nan])
    qs = _prepared(scorer, [[(100, 1)], [(90, 1)]], [250, 250])
    index = scorer.build_index_prepared(refs)
    filename = tmp_path / "saved.npz"
    scorer.save_index(index, filename)
    reloaded = scorer.load_index(filename)
    before = scorer.search_prepared_topk(qs, index, k=3)
    after = scorer.search_prepared_topk(qs, reloaded, k=3)
    np.testing.assert_array_equal(after.indices, before.indices)
    np.testing.assert_array_equal(after.scores, before.scores)


def test_entropy_subclass_inherits_topk():
    scorer = Entropy(matching_mode="hybrid", tolerance=0.025, remove_precursor=False, noise_cutoff=0, dtype=np.float32)
    refs = _prepared(scorer, [[(100, 1)], [(100, 1)]], [250, 251])
    qs = _prepared(scorer, [[(100, 1)]], [250])
    _check_matches_dense(scorer, refs, qs, k=1)


def test_public_api_prepares_queries(monkeypatch):
    scorer = _make_scorer("fragment")
    refs = _prepared(scorer, [[(100, 1)], [(110, 1)]], [250, 250])
    qs = _prepared(scorer, [[(100, 1)]], [250])
    index = scorer.build_index_prepared(refs)
    monkeypatch.setattr(scorer, "prepare_queries", lambda data: qs)
    got = scorer.search_topk(object(), index, k=2, progress_bar=False)
    assert got.get_hits(0)[0].tolist() == [0]


def test_does_not_call_dense_scoring(monkeypatch):
    scorer = _make_scorer("hybrid")
    refs = _prepared(scorer, [[(100, 1)], [(100, 1)]], [200.0, 200.0])
    qs = _prepared(scorer, [[(100, 1)]], [200.0])
    index = scorer.build_index_prepared(refs)
    monkeypatch.setattr(scorer, "_score_prepared", lambda *a: pytest.fail("dense scorer called"))
    result = scorer.search_prepared_topk(qs, index, k=1)
    assert result.valid_counts.tolist() == [1]


def test_empty_inputs_and_padding():
    scorer = _make_scorer("hybrid")
    empty = _prepared(scorer, [], [])
    nonempty = _prepared(scorer, [[(100, 1)]], [200.0])
    empty_refs = scorer.build_index_prepared(empty)
    small_refs = scorer.build_index_prepared(nonempty)
    assert scorer.search_prepared_topk(nonempty, empty_refs, k=4).shape == (1, 4)
    assert scorer.search_prepared_topk(empty, small_refs, k=4).shape == (0, 4)
    no_peaks = _prepared(scorer, [[]], [200.0])
    result = scorer.search_prepared_topk(no_peaks, small_refs, k=4)
    assert np.all(result.indices == -1)
    assert np.all(result.scores == 0)


def test_invalid_parameters_and_prepared_settings():
    scorer = _make_scorer("fragment")
    refs = _prepared(scorer, [[(100, 1)]], [200.0])
    index = scorer.build_index_prepared(refs)
    for k in (0, -1, True, 1.1):
        with pytest.raises((TypeError, ValueError)):
            scorer.search_prepared_topk(refs, index, k=k)
    for minimum in (-1, float("nan"), float("inf"), True):
        with pytest.raises((TypeError, ValueError)):
            scorer.search_prepared_topk(refs, index, min_score=minimum)
    for size in (0, -1, 1.4, True):
        with pytest.raises((TypeError, ValueError)):
            scorer.search_prepared_topk(refs, index, batch_size=size)
    with pytest.raises(TypeError):
        scorer.search_prepared_topk(refs, index, n_jobs=True)
    another = _make_scorer("fragment", np.float64)
    with pytest.raises(ValueError, match="preprocessing"):
        another.search_prepared_topk(refs, index)


def test_epoch_wraparound():
    work = _EntropyTopKWorkspace(7, 200, np.float32, hybrid=True)
    work.epoch = _MAX_EPOCH
    work.epochs.fill(_MAX_EPOCH)
    work.peak_word_epochs.fill(_MAX_EPOCH)
    assert work.next_epoch() == 1
    assert np.all(work.epochs == 0)
    assert np.all(work.peak_word_epochs == 0)
