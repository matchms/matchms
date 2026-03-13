import numpy as np
import pytest
from matchms.Scores import Scores, ScoresMask
from matchms.similarity import CosineGreedy
from matchms.similarity.BlinkCosine import BlinkCosine
from ..builder_Spectrum import SpectrumBuilder


# ---------- helpers ----------

def _build(builder, mz, intens):
    return (
        builder.with_mz(np.asarray(mz, dtype=float))
        .with_intensities(np.asarray(intens, dtype=float))
        .build()
    )


def _expected_strict_cosine(bins1, vals1, bins2, vals2):
    """Cosine under R=0 (no blur), after L2 normalization; assumes unique bins."""
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
    ([100, 200, 300], [1.0, 2.0, 3.0], [200, 400], [5.0, 1.0]),
    ([50, 60], [2.0, 2.0], [50, 60], [3.0, 4.0]),
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

    assert out == pytest.approx(expected, 1e-6)


def test_pair_clip_to_one():
    """Identical spectra -> score clipped to 1.0."""
    builder = SpectrumBuilder()
    s = _build(builder, [100, 200, 300], [0.1, 0.2, 1.0])

    sim = BlinkCosine(tolerance=0.0, bin_width=1.0, prefilter=False, clip_to_one=True)
    out = sim.pair(s, s)

    assert out == pytest.approx(1.0, 1e-6)


def test_pair_empty_spectrum():
    """Empty vs non-empty yields zero score."""
    builder = SpectrumBuilder()
    s_empty = _build(builder, [], [])
    s_full = _build(builder, [100, 200], [1.0, 1.0])

    sim = BlinkCosine(tolerance=0.0, bin_width=1.0, prefilter=False)
    out = sim.pair(s_empty, s_full)

    assert out == 0.0


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

    sim = BlinkCosine(tolerance=5.0, bin_width=1.0, prefilter=False, use_numba=use_numba)
    scores = sim.matrix(refs, qrys)

    assert isinstance(scores, Scores)
    assert scores.is_sparse is False
    assert scores.is_scalar is True
    assert scores.score_fields == ("score",)

    M = scores.to_array()
    assert M.shape == (len(refs), len(qrys))

    for i, r in enumerate(refs):
        for j, q in enumerate(qrys):
            s_pair = sim.pair(r, q)
            assert M[i, j] == pytest.approx(s_pair, 1e-6)


def test_matrix_self_comparison_is_symmetric():
    """When spectra_2 is None, matrix should return a symmetric dense score matrix."""
    builder = SpectrumBuilder()
    s1 = _build(builder, [100, 200, 300], [0.1, 0.2, 1.0])
    s2 = _build(builder, [100, 210, 310], [0.1, 0.2, 1.0])
    spectra = [s1, s2]

    sim = BlinkCosine(tolerance=10.0, bin_width=1.0, prefilter=False)
    scores = sim.matrix(spectra)

    assert isinstance(scores, Scores)
    S = scores.to_array()
    assert S.shape == (2, 2)
    assert S[0, 1] == pytest.approx(S[1, 0], 1e-6)
    assert S[0, 0] <= 1.0 and S[1, 1] <= 1.0

    p01 = sim.pair(s1, s2)
    assert S[0, 1] == pytest.approx(p01, 1e-6)


def test_matrix_handles_empty_inputs():
    sim = BlinkCosine(tolerance=0.0, bin_width=1.0, prefilter=False)
    builder = SpectrumBuilder()
    s = _build(builder, [100], [1.0])

    scores_1 = sim.matrix([], [s])
    assert isinstance(scores_1, Scores)
    assert scores_1.shape == (0, 1)

    scores_2 = sim.matrix([s], [])
    assert isinstance(scores_2, Scores)
    assert scores_2.shape == (1, 0)

    empty = _build(builder, [], [])
    scores_3 = sim.matrix([empty], [empty])
    assert scores_3.shape == (1, 1)
    assert scores_3.to_array()[0, 0] == 0.0


def test_matrix_default_score_fields_returns_score_only():
    """Default matrix() output should be scalar score-only Scores."""
    builder = SpectrumBuilder()
    refs = [_build(builder, [100, 200], [1.0, 1.0])]
    qrys = [_build(builder, [100, 201], [1.0, 1.0])]

    sim = BlinkCosine(tolerance=2.0, bin_width=1.0, prefilter=False)
    scores = sim.matrix(refs, qrys)

    assert isinstance(scores, Scores)
    assert scores.score_fields == ("score",)
    assert scores.is_scalar is True


def test_matrix_score_field_selection_still_works():
    """Scalar score matrices should still allow selecting ['score'] explicitly."""
    builder = SpectrumBuilder()
    refs = [_build(builder, [100, 200], [1.0, 1.0])]
    qrys = [_build(builder, [100, 201], [1.0, 1.0])]

    sim = BlinkCosine(tolerance=2.0, bin_width=1.0, prefilter=False)
    scores = sim.matrix(refs, qrys)
    score_only = scores["score"]

    assert isinstance(score_only, Scores)
    assert score_only.is_scalar is True
    assert score_only.score_fields == ("score",)
    np.testing.assert_array_equal(score_only.to_array(), scores.to_array())


def test_scalar_scores_score_alias_behaves_like_scores_itself():
    """For scalar Scores, scores['score'] and scores should behave equivalently."""
    builder = SpectrumBuilder()
    refs = [
        _build(builder, [100, 200], [1.0, 1.0]),
        _build(builder, [300], [1.0]),
    ]
    qrys = [
        _build(builder, [100, 201], [1.0, 1.0]),
        _build(builder, [350], [1.0]),
    ]

    sim = BlinkCosine(tolerance=2.0, bin_width=1.0, prefilter=False)
    scores = sim.matrix(refs, qrys)
    score_view = scores["score"]

    np.testing.assert_array_equal(scores.to_array(), score_view.to_array())
    assert scores[0, 0] == score_view[0, 0]

    mask_direct = scores > 0.2
    mask_via_alias = score_view > 0.2

    assert isinstance(mask_direct, ScoresMask)
    assert isinstance(mask_via_alias, ScoresMask)
    np.testing.assert_array_equal(mask_direct.to_dense(), mask_via_alias.to_dense())


def test_sparse_matrix_not_implemented():
    builder = SpectrumBuilder()
    s = _build(builder, [100, 200], [1.0, 1.0])

    sim = BlinkCosine(tolerance=2.0, bin_width=1.0, prefilter=False)

    with pytest.raises(NotImplementedError, match="sparse_matrix"):
        sim.sparse_matrix([s], [s], progress_bar=False)


@pytest.mark.parametrize("use_numba_pair", [True, False])
def test_pair_numba_vs_fallback_identical(use_numba_pair):
    """pair() results must be identical whether numba path is used or not."""
    builder = SpectrumBuilder()
    s1 = _build(builder, [100, 200, 300, 400], [0.3, 1.0, 0.5, 0.2])
    s2 = _build(builder, [99, 201, 305, 500], [0.2, 0.9, 0.4, 0.1])

    sim_numba = BlinkCosine(tolerance=5.0, bin_width=1.0, prefilter=False, use_numba=True)
    sim_fallback = BlinkCosine(tolerance=5.0, bin_width=1.0, prefilter=False, use_numba=False)

    out_a = (sim_numba if use_numba_pair else sim_fallback).pair(s1, s2)
    out_b = (sim_fallback if use_numba_pair else sim_numba).pair(s1, s2)

    assert out_a == pytest.approx(out_b, 1e-6)


def test_blinkcosine_upper_bound_cosinegreedy():
    """BLINK similarity should always be >= CosineGreedy for the same tolerance."""
    builder = SpectrumBuilder()

    spectrum_1 = builder.with_mz(np.array([100, 200, 300, 400], dtype=float)).with_intensities(
        np.array([0.5, 0.2, 1.0, 0.3], dtype=float)
    ).build()

    spectrum_2 = builder.with_mz(np.array([98, 199, 305, 410], dtype=float)).with_intensities(
        np.array([0.4, 0.3, 0.8, 0.2], dtype=float)
    ).build()

    tolerance = 5.0
    cg = CosineGreedy(tolerance=tolerance)
    bc = BlinkCosine(tolerance=tolerance, bin_width=1.0, prefilter=False)

    score_cg = cg.pair(spectrum_1, spectrum_2)["score"]
    score_bc = bc.pair(spectrum_1, spectrum_2)

    assert score_bc >= score_cg - 1e-6
