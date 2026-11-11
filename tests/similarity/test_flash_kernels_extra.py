"""Small analytic edge cases for dense Numba Flash cosine/entropy kernels.

Unlike public API parity tests these exercise explicit output-block boundaries,
empty rows, greedy one-to-one conflicts and the optional match-count buffer.
"""

from types import SimpleNamespace
import numpy as np
import pytest
from matchms.similarity._flash_cosine import cosine_rows
from matchms.similarity._flash_cosine_topk import (
    _CosineTopKWorkspace,
    _score_cosine_topk_block,
)
from matchms.similarity._flash_entropy import entropy_rows
from matchms.similarity._flash_entropy_topk import (
    _EntropyTopKWorkspace,
    _score_entropy_topk_block,
)


def _packed(rows, precursors):
    sizes = [len(r) for r in rows]
    flat = np.concatenate([np.asarray(row, dtype=np.float64).reshape(-1, 2) for row in rows], axis=0)
    return (
        np.r_[0, np.cumsum(sizes)].astype(np.int64),
        flat[:, 0].copy(),
        flat[:, 1].copy(),
        np.asarray(precursors, dtype=np.float64),
    )


def _index(rows, precursors):
    offsets, mz, intensity, pmz = _packed(rows, precursors)
    owners = np.repeat(np.arange(len(rows), dtype=np.int32), np.diff(offsets))
    order = np.argsort(mz, kind="stable")
    products = (mz[order], intensity[order], owners[order])
    physical = np.empty(len(order), dtype=np.int64)
    physical[order] = np.arange(len(order), dtype=np.int64)
    valid = np.isfinite(pmz[owners])
    spec_major_positions = np.flatnonzero(valid)
    losses = pmz[owners[valid]] - mz[valid]
    loss_order = np.argsort(losses, kind="stable")
    product_positions = physical[spec_major_positions[loss_order]]
    loss_owner = owners[spec_major_positions[loss_order]]
    return products, (losses[loss_order], loss_owner, product_positions), pmz


def _run_cosine(
    queries, query_precursors, refs, ref_precursors, *, mode=0, gate=-1.0, ppm=False, with_matches=True, rows=None
):
    offsets, mz, intensities, pmz = _packed(queries, query_precursors)
    product, loss, ref_pmz = _index(refs, ref_precursors)
    norms = np.asarray([np.linalg.norm(np.asarray(r, dtype=float).reshape(-1, 2)[:, 1]) for r in queries])
    lib_norms = np.asarray([np.linalg.norm(np.asarray(r, dtype=float).reshape(-1, 2)[:, 1]) for r in refs])
    out = np.full((len(queries), len(refs)), -9.0)
    matched = np.full((len(queries), len(refs)), -9, dtype=np.int32) if with_matches else np.empty((0, 0), np.int32)
    first, stop = rows if rows is not None else (0, len(queries))
    cosine_rows(
        out,
        matched,
        offsets,
        mz,
        intensities,
        pmz,
        norms,
        *product,
        *loss,
        ref_pmz,
        lib_norms,
        0.02,
        ppm,
        mode,
        gate,
        False,
        first,
        stop,
    )
    return out, matched


def _run_entropy(queries, query_precursors, refs, ref_precursors, *, mode=0, gate=-1.0, ppm=False):
    offsets, mz, intensities, pmz = _packed(queries, query_precursors)
    product, loss, ref_pmz = _index(refs, ref_precursors)
    # These tests use explicitly entropy-weighted intensities summing to 0.5.
    product_mz, product_int, owner = product
    loss_mz, loss_owner, product_positions = loss
    physical_ids = np.arange(len(product_mz), dtype=np.int64)
    terms = np.where(product_int > 0, product_int * np.log2(np.where(product_int > 0, product_int, 1)), 0.0)
    fragment = (product_mz, product_int, terms, owner, physical_ids)
    neutral_loss = (
        loss_mz,
        product_int[product_positions],
        terms[product_positions],
        loss_owner,
        product_positions,
    )
    out = np.full((len(queries), len(refs)), -9.0)
    entropy_rows(
        out,
        offsets,
        mz,
        intensities,
        pmz,
        fragment,
        neutral_loss,
        ref_pmz,
        len(product_mz),
        0.02,
        ppm,
        mode,
        gate,
        False,
        0,
        len(queries),
    )
    return out


@pytest.mark.parametrize("mode", [0, 1, 2])
def test_dense_cosine_analytic_matching_modes(mode):
    refs = [[(100, 1), (200, 1)], [(100, 1), (210, 1)], [(800, 1)]]
    queries = [[(100, 1), (200, 1)], [(100, 1), (220, 1)], [], [(777, 1)]]
    out, matches = _run_cosine(queries, [500, 520, np.nan, np.nan], refs, [500, 510, 700], mode=mode)
    assert out.shape == (4, 3)
    assert out[0, 0] == pytest.approx(1.0)
    assert matches[0, 0] == 2
    assert out[0, 2] == 0.0
    np.testing.assert_array_equal(out[2:], 0.0)
    np.testing.assert_array_equal(matches[2:], 0)
    expected_ref1 = 0.5 if mode != 2 else 1.0
    assert out[0, 1] == pytest.approx(expected_ref1)
    assert matches[0, 1] == (1 if mode != 2 else 2)


def test_dense_cosine_competing_query_peaks_count_only_once():
    out, matched = _run_cosine([[(100.0, 1), (100.01, 1)]], [500], [[(100.005, 1)]], [500])
    assert out[0, 0] == pytest.approx(1 / np.sqrt(2))
    assert matched[0, 0] == 1


def test_dense_cosine_score_only_gate_and_disjoint_row_write():
    queries = [[(100, 1)], [(100, 1)], []]
    refs = [[(100, 1)], [(100, 1)]]
    out, matched = _run_cosine(queries, [500, 510, np.nan], refs, [500, 520], gate=0.1, with_matches=False, rows=(1, 2))
    assert matched.shape == (0, 0)
    assert out[0].tolist() == [-9.0, -9.0]
    assert out[1].tolist() == [0.0, 0.0]
    assert out[2].tolist() == [-9.0, -9.0]


@pytest.mark.parametrize("mode", [0, 1, 2])
def test_dense_entropy_analytic_matching_modes(mode):
    refs = [[(100, 0.25), (200, 0.25)], [(100, 0.25), (210, 0.25)], [(800, 0.5)]]
    queries = [[(100, 0.25), (200, 0.25)], [], [(777, 0.5)]]
    out = _run_entropy(queries, [500, np.nan, np.nan], refs, [500, 510, 700], mode=mode)
    assert out[0, 0] == pytest.approx(1.0)
    assert out[0, 1] == pytest.approx(1.0 if mode == 2 else 0.5)
    assert out[0, 2] == 0
    np.testing.assert_array_equal(out[1:], 0.0)


def test_dense_entropy_identity_and_no_precursor_loss():
    refs = [[(100, 0.5)], [(100, 0.5)]]
    queries = [[(100, 0.5)], [(100, 0.5)]]
    out = _run_entropy(queries, [500, np.nan], refs, [500, 510], mode=0, gate=0.1)
    np.testing.assert_allclose(out, [[1.0, 0.0], [0.0, 0.0]])
    loss_only = _run_entropy(queries, [500, np.nan], refs, [500, 510], mode=1)
    assert loss_only[1, 0] == 0.0
    assert loss_only[1, 1] == 0.0


def test_dense_entropy_competing_peaks_do_not_double_count():
    out = _run_entropy([[(100.0, 0.25), (100.01, 0.25)]], [500], [[(100.005, 0.5)]], [500])
    pair_score = 0.75 * np.log2(0.75) - 0.25 * np.log2(0.25) - 0.5 * np.log2(0.5)
    assert out[0, 0] == pytest.approx(pair_score)


def _sort_dense_row(row, k, threshold):
    candidates = np.flatnonzero(row > threshold)
    ids = candidates[np.lexsort((candidates, -row[candidates]))][:k]
    return ids


def _reference_example(entropy=False):
    if entropy:
        references = [[(100, 0.25), (200, 0.25)], [(100, 0.25), (210, 0.25)], [(800, 0.5)], [], [(100, 0.5)]]
        queries = [[(100, 0.25), (200, 0.25)], [(100, 0.25), (220, 0.25)], [], [(100, 0.5)], [(999, 0.5)]]
    else:
        references = [[(100, 1), (200, 1)], [(100, 1), (210, 1)], [(800, 1)], [], [(100, 1)]]
        queries = [[(100, 1), (200, 1)], [(100, 1), (220, 1)], [], [(100, 1)], [(999, 1)]]
    return references, queries, [500, 510, 700, np.nan, 500], [500, 520, np.nan, 500, 800]


def _prepare_topk_fixture(references, queries, reference_precursors, query_precursors, dtype):
    q_offsets, q_mz, q_int, q_pmz = _packed(queries, query_precursors)
    products, losses, lib_pmz = _index(references, reference_precursors)
    lib_norm = np.asarray([np.linalg.norm(np.asarray(r, float).reshape(-1, 2)[:, 1]) for r in references])
    q_norm = np.asarray([np.linalg.norm(np.asarray(r, float).reshape(-1, 2)[:, 1]) for r in queries])
    return (
        SimpleNamespace(
            spec_offsets=q_offsets,
            spec_mz=q_mz.astype(dtype),
            spec_int=q_int.astype(dtype),
            precursor_mz=q_pmz,
            spec_l2=q_norm,
        ),
        products,
        losses,
        lib_pmz,
        lib_norm,
    )


@pytest.mark.parametrize("mode", [0, 1, 2])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("gate", [-1.0, 0.1])
@pytest.mark.parametrize("include_matches", [False, True])
def test_cosine_topk_block_exactly_matches_dense_for_modes_and_matches(mode, dtype, gate, include_matches):
    refs, queries, ref_pmz, q_pmz = _reference_example()
    prepared, product, loss, lib_precursors, lib_norm = _prepare_topk_fixture(refs, queries, ref_pmz, q_pmz, dtype)
    n_references, n_queries, k = len(refs), len(queries), 3
    dense = np.zeros((n_queries, n_references), dtype=dtype)
    dense_matches = np.zeros((n_queries, n_references), dtype=np.int32)
    cosine_rows(
        dense,
        dense_matches,
        prepared.spec_offsets,
        prepared.spec_mz,
        prepared.spec_int,
        prepared.precursor_mz,
        prepared.spec_l2,
        *product,
        *loss,
        lib_precursors,
        lib_norm,
        0.02,
        False,
        mode,
        gate,
        False,
        0,
        n_queries,
    )
    workspace = _CosineTopKWorkspace(n_references, len(product[0]), dtype, mode == 2)
    ids = np.full((n_queries, k), -1, dtype=np.int32)
    scores = np.zeros((n_queries, k), dtype=dtype)
    matches = np.zeros((n_queries, k), dtype=np.int32) if include_matches else None
    counts = np.zeros(n_queries, np.int64)
    library = SimpleNamespace(
        peaks_mz=product[0].astype(dtype),
        peaks_int=product[1].astype(dtype),
        peaks_spec_idx=product[2],
        nl_mz=loss[0].astype(dtype),
        nl_spec_idx=loss[1],
        nl_product_idx=loss[2],
        precursor_mz=lib_precursors,
        spec_l2=lib_norm,
    )
    _score_cosine_topk_block(
        prepared, library, workspace, ids, scores, counts, matches, 0, n_queries, k, 0.0, mode, 0.02, False, gate, False
    )
    for row in range(n_queries):
        expected = _sort_dense_row(dense[row], k, 0.0)
        np.testing.assert_array_equal(ids[row, : len(expected)], expected)
        np.testing.assert_array_equal(scores[row, : len(expected)], dense[row, expected])
        if include_matches:
            np.testing.assert_array_equal(matches[row, : len(expected)], dense_matches[row, expected])
        assert counts[row] == len(expected)
        np.testing.assert_array_equal(ids[row, len(expected) :], -1)
        np.testing.assert_array_equal(scores[row, len(expected) :], 0)


@pytest.mark.parametrize("mode", [0, 1, 2])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_entropy_topk_block_exactly_matches_dense_with_reused_epochs(mode, dtype):
    refs, queries, ref_pmz, q_pmz = _reference_example(entropy=True)
    prepared, product, loss, lib_precursors, _ = _prepare_topk_fixture(refs, queries, ref_pmz, q_pmz, dtype)
    n_references, n_queries, k = len(refs), len(queries), 3
    dense = np.zeros((n_queries, n_references), dtype=dtype)
    mz, intensity, owners = product
    terms = np.where(intensity > 0, intensity * np.log2(np.where(intensity > 0, intensity, 1)), 0.0)
    fragment = (mz.astype(dtype), intensity.astype(dtype), terms, owners, np.arange(len(mz), dtype=np.int64))
    neutral_loss = (loss[0].astype(dtype), intensity[loss[2]].astype(dtype), terms[loss[2]], loss[1], loss[2])
    entropy_rows(
        dense,
        prepared.spec_offsets,
        prepared.spec_mz,
        prepared.spec_int,
        prepared.precursor_mz,
        fragment,
        neutral_loss,
        lib_precursors,
        len(mz),
        0.02,
        False,
        mode,
        -1.0,
        False,
        0,
        n_queries,
    )
    library = SimpleNamespace(
        fragment=fragment, neutral_loss=neutral_loss, precursor_mz=lib_precursors, n_peaks=len(mz)
    )
    workspace = _EntropyTopKWorkspace(n_references, len(mz), dtype, mode == 2)
    ids = np.full((n_queries, k), -1, dtype=np.int32)
    scores = np.zeros((n_queries, k), dtype=dtype)
    counts = np.zeros(n_queries, np.int64)
    _score_entropy_topk_block(
        prepared, library, workspace, ids, scores, counts, 0, n_queries, k, 0.0, mode, 0.02, False, -1.0, False
    )
    for row in range(n_queries):
        expected = _sort_dense_row(dense[row], k, 0.0)
        np.testing.assert_array_equal(ids[row, : len(expected)], expected)
        np.testing.assert_array_equal(scores[row, : len(expected)], dense[row, expected])
        assert counts[row] == len(expected)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize(
    "mode,reference,query,ref_precursor,query_precursor",
    [
        pytest.param(
            0, [(100.005, 1), (200, 1)], [(100, 1), (100.01, 2)], 500, 500, id="fragment_competing_query_peaks"
        ),
        pytest.param(1, [(100, 1), (200, 1)], [(120, 1), (120.01, 2)], 500, 520, id="loss_competing_query_peaks"),
        pytest.param(
            2, [(100, 2), (200, 1)], [(100, 1), (120, 3)], 500, 520, id="hybrid_direct_and_loss_same_physical_peak"
        ),
    ],
)
def test_cosine_topk_greedy_conflicts_match_dense(mode, reference, query, ref_precursor, query_precursor, dtype):
    """The compact conflict-edge path must exactly reproduce dense greedy assignment."""
    refs = [reference, [(400, 1)]]
    queries = [query, [], query]
    prepared, product, loss, lib_precursors, lib_norm = _prepare_topk_fixture(
        refs, queries, [ref_precursor, 600], [query_precursor, np.nan, query_precursor], dtype
    )
    n_queries, n_references, k = len(queries), len(refs), 2
    dense = np.zeros((n_queries, n_references), dtype=dtype)
    dense_matches = np.zeros((n_queries, n_references), dtype=np.int32)
    cosine_rows(
        dense,
        dense_matches,
        prepared.spec_offsets,
        prepared.spec_mz,
        prepared.spec_int,
        prepared.precursor_mz,
        prepared.spec_l2,
        *product,
        *loss,
        lib_precursors,
        lib_norm,
        0.02,
        False,
        mode,
        -1.0,
        False,
        0,
        n_queries,
    )
    library = SimpleNamespace(
        peaks_mz=product[0].astype(dtype),
        peaks_int=product[1].astype(dtype),
        peaks_spec_idx=product[2],
        nl_mz=loss[0].astype(dtype),
        nl_spec_idx=loss[1],
        nl_product_idx=loss[2],
        precursor_mz=lib_precursors,
        spec_l2=lib_norm,
    )
    work = _CosineTopKWorkspace(n_references, len(product[0]), dtype, mode == 2)
    indices = np.full((n_queries, k), -1, dtype=np.int32)
    scores = np.zeros((n_queries, k), dtype=dtype)
    matches = np.zeros((n_queries, k), dtype=np.int32)
    counts = np.zeros(n_queries, dtype=np.int64)
    _score_cosine_topk_block(
        prepared, library, work, indices, scores, counts, matches, 0, n_queries, k, 0.0, mode, 0.02, False, -1.0, False
    )
    for row in range(n_queries):
        expected = _sort_dense_row(dense[row], k, 0.0)
        np.testing.assert_array_equal(indices[row, : len(expected)], expected)
        np.testing.assert_array_equal(scores[row, : len(expected)], dense[row, expected])
        np.testing.assert_array_equal(matches[row, : len(expected)], dense_matches[row, expected])
        assert counts[row] == len(expected)
    assert counts[1] == 0  # The empty row cannot retain state from earlier queries.
    assert dense_matches[0, 0] == 1  # The two candidate edges share a physical peak.
