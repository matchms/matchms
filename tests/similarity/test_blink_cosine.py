# test_blinkcosine.py
import numpy as np
import pytest
from matchms.similarity.BlinkCosine import BlinkCosine
from ..builder_Spectrum import SpectrumBuilder


# ---------- helpers ----------

def _build(builder, mz, intens):
    return builder.with_mz(np.asarray(mz, dtype=float)).with_intensities(np.asarray(intens, dtype=float)).build()


def _expected_strict_cosine(bins1, vals1, bins2, vals2):
    """Cosine under R=0 (no blur), after L2 normalization; assumes unique bins."""
    # Intersect bins
    common, i1, i2 = np.intersect1d(bins1, bins2, assume_unique=True, return_indices=True)
    if common.size == 0:
        return 0.0
    return float(np.dot(vals1[i1], vals2[i2]))


def _prep_naive(s, bin_width=1.0, mz_power=0.0, intensity_power=1.0):
    """Replicates BlinkCosine _prep_spectrum steps used in tests (prefilter disabled)."""
    mz = s.peaks.mz
    inten = s.peaks.intensities
    if mz.size == 0:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=float)
    if mz_power != 0.0:
        inten = inten * np.power(mz, mz_power, dtype=float)
    if intensity_power != 1.0:
        inten = np.power(inten, intensity_power, dtype=float)
    bins = np.floor(mz / bin_width + 0.5).astype(np.int64)
    order = np.argsort(bins, kind="mergesort")
    bins = bins[order]
    inten = inten[order]
    uniq, idx, _cnts = np.unique(bins, return_index=True, return_counts=True)
    val_sum = np.add.reduceat(inten, idx)
    norm = np.linalg.norm(val_sum)
    if norm == 0.0:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=float)
    return uniq, (val_sum / norm)


# ---------- tests ----------

@pytest.mark.parametrize("mz1,int1,mz2,int2", [
    # partial overlap
    ([100, 200, 300], [1.0, 2.0, 3.0], [200, 400], [5.0, 1.0]),
    # exact same peaks
    ([50, 60], [2.0, 2.0], [50, 60], [3.0, 4.0]),
    # no overlap
    ([10, 20], [1.0, 1.0], [30, 40], [1.0, 1.0]),
])
@pytest.mark.parametrize("use_numba", [True, False])
def test_pair_strict_R0_expected(mz1, int1, mz2, int2, use_numba):
    """R=0 via tolerance=0, bin_width=1: score equals cosine of shared binned intensities."""
    builder = SpectrumBuilder()
    s1 = _build(builder, mz1, int1)
    s2 = _build(builder, mz2, int2)

    sim = BlinkCosine(tolerance=0.0, bin_width=1.0, prefilter=False, use_numba=use_numba)
    out = sim.pair(s1, s2)

    b1, v1 = _prep_naive(s1, bin_width=1.0)
    b2, v2 = _prep_naive(s2, bin_width=1.0)
    expected = _expected_strict_cosine(b1, v1, b2, v2)
    assert out["score"] == pytest.approx(expected, 1e-12)
    # matches equals number of exact-bin overlaps at R=0 (counts are 1 since we ensured unique bins)
    expected_matches = np.intersect1d(b1, b2, assume_unique=True).size
    assert out["matches"] == expected_matches


def test_pair_clip_to_one_and_matches():
    """Identical spectra -> score clipped to 1.0, matches = number of peaks (R=0)."""
    builder = SpectrumBuilder()
    s = _build(builder, [100, 200, 300], [0.1, 0.2, 1.0])

    sim = BlinkCosine(tolerance=0.0, bin_width=1.0, prefilter=False, clip_to_one=True)
    out = sim.pair(s, s)
    assert out["score"] == pytest.approx(1.0, 1e-12)
    assert out["matches"] == 3


def test_pair_empty_spectrum():
    """Empty vs non-empty yields zero score and zero matches."""
    builder = SpectrumBuilder()
    s_empty = _build(builder, [], [])
    s_full = _build(builder, [100, 200], [1.0, 1.0])

    sim = BlinkCosine(tolerance=0.0, bin_width=1.0, prefilter=False)
    out = sim.pair(s_empty, s_full)
    assert out["score"] == 0.0
    assert out["matches"] == 0


@pytest.mark.parametrize("use_numba", [True, False])
def test_matrix_equals_pair_scores_dense(use_numba):
    """Dense matrix scores must equal pair() scores for each (ref, qry)."""
    builder = SpectrumBuilder()
    refs = [
        _build(builder, [100, 200, 300], [0.1, 0.2, 1.0]),
        _build(builder, [110, 190, 290], [0.5, 0.2, 1.0]),
    ]
    qrys = [
        _build(builder, [100, 205, 305], [0.2, 0.3, 1.0]),
        _build(builder, [50, 60, 70], [0.2, 0.2, 0.2]),
        _build(builder, [110, 190, 290], [0.5, 0.2, 1.0]),
    ]

    # Small radius to allow near matches:
    sim = BlinkCosine(tolerance=5.0, bin_width=1.0, prefilter=False, use_numba=use_numba)
    M = sim.matrix(refs, qrys, array_type="numpy")

    assert M.shape == (len(refs), len(qrys))
    for i, r in enumerate(refs):
        for j, q in enumerate(qrys):
            s_pair = sim.pair(r, q)["score"]
            assert M[i, j] == pytest.approx(s_pair, 1e-12)


def test_matrix_is_symmetric_when_requested():
    """When references is queries and is_symmetric=True, enforce symmetry."""
    builder = SpectrumBuilder()
    s1 = _build(builder, [100, 200, 300], [0.1, 0.2, 1.0])
    s2 = _build(builder, [100, 210, 310], [0.1, 0.2, 1.0])
    spectra = [s1, s2]

    sim = BlinkCosine(tolerance=10.0, bin_width=1.0, prefilter=False)
    S = sim.matrix(spectra, spectra, array_type="numpy", is_symmetric=True)

    assert S.shape == (2, 2)
    # symmetry
    assert S[0, 1] == pytest.approx(S[1, 0], 1e-12)
    # diagonals clipped to <= 1
    assert S[0, 0] <= 1.0 and S[1, 1] <= 1.0
    # matches pair() on off-diagonal
    p01 = sim.pair(s1, s2)["score"]
    assert S[0, 1] == pytest.approx(p01, 1e-12)


def test_matrix_handles_empty_inputs():
    sim = BlinkCosine(tolerance=0.0, bin_width=1.0, prefilter=False)
    builder = SpectrumBuilder()
    s = _build(builder, [100], [1.0])

    # no refs
    M1 = sim.matrix([], [s])
    assert M1.shape == (0, 1)
    # no qrys
    M2 = sim.matrix([s], [])
    assert M2.shape == (1, 0)
    # all empty spectra
    empty = _build(builder, [], [])
    M3 = sim.matrix([empty], [empty])
    assert M3.shape == (1, 1)
    assert M3[0, 0] == 0.0


def test_matrix_invalid_array_type_raises():
    builder = SpectrumBuilder()
    s = _build(builder, [100], [1.0])
    sim = BlinkCosine(prefilter=False)
    with pytest.raises(ValueError, match="array_type must be 'numpy' or 'sparse'"):
        sim.matrix([s], [s], array_type="not-valid")


def test_matrix_sparse_and_threshold_matches_pair():
    """Sparse output must include exactly the entries with score >= sparse_score_min."""
    builder = SpectrumBuilder()
    refs = [
        _build(builder, [100, 200], [1.0, 1.0]),
        _build(builder, [150, 250], [1.0, 1.0]),
    ]
    qrys = [
        _build(builder, [100, 201], [1.0, 1.0]),   # near first ref
        _build(builder, [350], [1.0]),             # no match anywhere
    ]

    sim = BlinkCosine(tolerance=2.0, bin_width=1.0, prefilter=False, sparse_score_min=0.2)
    S_sparse = sim.matrix(refs, qrys, array_type="sparse")
    # to COO arrays
    rows, cols, data = S_sparse.row, S_sparse.col, S_sparse.data

    # compute all pair scores to know ground truth
    scores = np.array([[sim.pair(r, q)["score"] for q in qrys] for r in refs])
    mask = scores >= sim.sparse_score_min
    expected_nnz = mask.sum()

    assert data.size == expected_nnz
    # Each stored entry should meet the threshold, and values must match pair() within tolerance
    for r, c, v in zip(rows, cols, data):
        assert scores[r, c] >= sim.sparse_score_min
        assert v == pytest.approx(scores[r, c], 1e-12)


def test_matrix_dense_vs_sparse_values_identical_without_threshold():
    """With threshold 0, the numeric values in dense and sparse must agree (ignoring explicit zeros)."""
    builder = SpectrumBuilder()
    refs = [
        _build(builder, [100, 200], [1.0, 1.0]),
        _build(builder, [300], [2.0]),
    ]
    qrys = [
        _build(builder, [100, 199], [1.0, 1.0]),
        _build(builder, [300, 301], [1.0, 1.0]),
    ]

    sim = BlinkCosine(tolerance=2.0, bin_width=1.0, prefilter=False, sparse_score_min=0.0)
    D = sim.matrix(refs, qrys, array_type="numpy")
    S = sim.matrix(refs, qrys, array_type="sparse").tocoo()

    # Build dense from sparse for comparison
    dense_from_sparse = np.zeros_like(D)
    dense_from_sparse[S.row, S.col] = S.data

    assert D.shape == dense_from_sparse.shape
    assert np.allclose(D, dense_from_sparse, atol=1e-12)


@pytest.mark.parametrize("use_numba_pair", [True, False])
def test_pair_numba_vs_fallback_identical(use_numba_pair):
    """pair() results must be identical whether numba path is used or not."""
    builder = SpectrumBuilder()
    s1 = _build(builder, [100, 200, 300, 400], [0.3, 1.0, 0.5, 0.2])
    s2 = _build(builder, [99, 201, 305, 500], [0.2, 0.9, 0.4, 0.1])

    # Larger R to exercise windowed sum beyond exact matches
    sim_numba = BlinkCosine(tolerance=5.0, bin_width=1.0, prefilter=False, use_numba=True)
    sim_fallback = BlinkCosine(tolerance=5.0, bin_width=1.0, prefilter=False, use_numba=False)

    out_a = (sim_numba if use_numba_pair else sim_fallback).pair(s1, s2)
    out_b = (sim_fallback if use_numba_pair else sim_numba).pair(s1, s2)

    assert out_a["score"] == pytest.approx(out_b["score"], 1e-12)
    assert out_a["matches"] == out_b["matches"]
