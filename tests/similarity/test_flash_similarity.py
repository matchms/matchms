import numpy as np
import pytest
from matchms.Scores import Scores
from matchms.similarity import CosineGreedy, ModifiedCosineGreedy
from matchms.similarity.FlashSimilarity import FlashCosine, FlashEntropy
from ..builder_Spectrum import SpectrumBuilder


# ----------------------------
# Helpers
# ----------------------------

def build_spectrum(mz, intens, precursor_mz=None):
    """Build a Spectrum via SpectrumBuilder, setting precursor_mz when available."""
    b = SpectrumBuilder().with_mz(np.asarray(mz, dtype="float")).with_intensities(
        np.asarray(intens, dtype="float")
    )
    if hasattr(b, "with_precursor_mz") and precursor_mz is not None:
        b = b.with_precursor_mz(float(precursor_mz))
    elif precursor_mz is not None and hasattr(b, "with_metadata"):
        b = b.with_metadata({"precursor_mz": float(precursor_mz)})
    return b.build()


# ----------------------------
# Spectral-entropy path (public API level)
# ----------------------------

@pytest.mark.parametrize("use_ppm,tol", [(False, 0.1), (True, 100.0)])
def test_entropy_pair_fragment_commutative_and_positive(use_ppm, tol):
    s1 = build_spectrum([100, 200, 300], [0.2, 1.0, 0.4], precursor_mz=500.0)
    s2 = build_spectrum([100, 200, 305], [0.1, 0.5, 0.2], precursor_mz=500.0)

    fse = FlashEntropy(
        tolerance=tol,
        use_ppm=use_ppm,
        matching_mode="fragment",
        remove_precursor=False,
        noise_cutoff=0.0,
        normalize_to_half=False,
        merge_within=0.0,
        dtype=np.float32,
    )
    score12 = float(fse.pair(s1, s2))
    score21 = float(fse.pair(s2, s1))
    assert score12 == pytest.approx(score21, 1e-6)
    assert score12 > 0.0


def test_entropy_pair_returns_zero_when_empty_after_cleanup():
    # Precursor window removes everything
    s1 = build_spectrum([199.0, 199.5], [1.0, 0.5], precursor_mz=200.0)
    s2 = build_spectrum([199.2, 199.7], [1.0, 0.5], precursor_mz=200.0)
    fse = FlashEntropy(
        tolerance=0.02,
        matching_mode="fragment",
        remove_precursor=True,
        precursor_window=1.6,
        noise_cutoff=0.0,
        normalize_to_half=True,
        merge_within=0.0,
    )
    assert float(fse.pair(s1, s2)) == 0.0


def test_entropy_identity_gate_da_and_ppm():
    s1 = build_spectrum([100, 200], [1.0, 1.0], precursor_mz=500.0)
    s2 = build_spectrum([100, 200], [1.0, 1.0], precursor_mz=500.3)

    base = FlashEntropy(
        tolerance=0.02, matching_mode="fragment", remove_precursor=False, noise_cutoff=0.0
    )
    base_score = float(base.pair(s1, s2))
    assert base_score > 0.0

    # Strict Da gate
    gate_da_tight = FlashEntropy(
        tolerance=0.02,
        matching_mode="fragment",
        identity_precursor_tolerance=0.2,
        identity_use_ppm=False,
        remove_precursor=False,
        noise_cutoff=0.0,
    )
    assert float(gate_da_tight.pair(s1, s2)) == 0.0

    gate_da_loose = FlashEntropy(
        tolerance=0.02,
        matching_mode="fragment",
        identity_precursor_tolerance=0.5,
        identity_use_ppm=False,
        remove_precursor=False,
        noise_cutoff=0.0,
    )
    assert float(gate_da_loose.pair(s1, s2)) == pytest.approx(base_score, abs=1e-7)

    # PPM gate
    gate_ppm_tight = FlashEntropy(
        tolerance=0.02,
        matching_mode="fragment",
        identity_precursor_tolerance=300.0,
        identity_use_ppm=True,
        remove_precursor=False,
        noise_cutoff=0.0,
    )
    assert float(gate_ppm_tight.pair(s1, s2)) == 0.0

    gate_ppm_loose = FlashEntropy(
        tolerance=0.02,
        matching_mode="fragment",
        identity_precursor_tolerance=800.0,
        identity_use_ppm=True,
        remove_precursor=False,
        noise_cutoff=0.0,
    )
    assert float(gate_ppm_loose.pair(s1, s2)) == pytest.approx(base_score, abs=1e-7)


def test_entropy_neutral_loss_vs_hybrid_prefers_fragments():
    # NL hits duplicate fragment hits -> hybrid should equal fragment-only
    q = build_spectrum([100, 200], [1.0, 1.0], precursor_mz=500.0)
    r = build_spectrum([100, 200], [1.0, 1.0], precursor_mz=500.0)

    kwargs = dict(
        tolerance=0.1,
        use_ppm=False,
        remove_precursor=False,
        noise_cutoff=0.0,
        normalize_to_half=False,
        merge_within=0.0,
    )

    frag = FlashEntropy(matching_mode="fragment", **kwargs)
    nl = FlashEntropy(matching_mode="neutral_loss", **kwargs)
    hyb = FlashEntropy(matching_mode="hybrid", **kwargs)

    s_frag = float(frag.pair(r, q))
    s_nl = float(nl.pair(r, q))
    s_hyb = float(hyb.pair(r, q))

    assert s_frag > 0.0
    assert s_hyb == pytest.approx(s_frag, abs=1e-7)
    assert s_nl >= s_hyb


def test_entropy_matrix_dense_matches_pair():
    refs = [
        build_spectrum([100, 200], [1.0, 0.5], precursor_mz=500.0),
        build_spectrum([110, 300], [0.3, 1.0], precursor_mz=600.0),
    ]
    qs = [
        build_spectrum([100, 205], [1.0, 0.5], precursor_mz=500.0),
        build_spectrum([110, 300], [1.0, 0.3], precursor_mz=600.0),
    ]
    fse = FlashEntropy(
        tolerance=0.1,
        use_ppm=False,
        matching_mode="fragment",
        remove_precursor=False,
        noise_cutoff=0.0,
        normalize_to_half=False,
        merge_within=0.0,
    )
    scores = fse.matrix(refs, qs, n_jobs=0, progress_bar=False)
    assert isinstance(scores, Scores)
    assert scores.is_sparse is False
    assert scores.is_scalar is True

    M = scores.to_array()
    assert M.shape == (2, 2)
    for i, r in enumerate(refs):
        for j, q in enumerate(qs):
            expected = float(fse.pair(r, q))
            assert float(M[i, j]) == pytest.approx(expected, abs=1e-6)


def test_entropy_matrix_self_comparison_returns_scores():
    spectra = [
        build_spectrum([100, 200], [1.0, 0.5], precursor_mz=500.0),
        build_spectrum([110, 300], [0.3, 1.0], precursor_mz=600.0),
    ]
    fse = FlashEntropy(
        tolerance=0.1,
        use_ppm=False,
        matching_mode="fragment",
        remove_precursor=False,
        noise_cutoff=0.0,
        normalize_to_half=False,
        merge_within=0.0,
    )

    scores = fse.matrix(spectra, n_jobs=0, progress_bar=False)
    assert isinstance(scores, Scores)
    assert scores.shape == (2, 2)

    M = scores.to_array()
    assert M.shape == (2, 2)
    assert np.allclose(M, M.T, atol=1e-6)


def test_entropy_fragment_score_is_bounded_with_overlapping_windows():
    # Overlapping tolerance windows used to cause many-to-many over-counting.
    s1 = build_spectrum([100.000, 100.010], [1.0, 1.0], precursor_mz=250.0)
    s2 = build_spectrum([100.005, 100.015], [1.0, 1.0], precursor_mz=250.0)

    fse = FlashEntropy(
        matching_mode="fragment",
        tolerance=0.02,
        use_ppm=False,
        remove_precursor=False,
        noise_cutoff=0.0,
        normalize_to_half=True,
        merge_within=0.0,
        dtype=np.float64,
    )

    score = float(fse.pair(s1, s2))
    assert score <= 1.0 + 1e-7


def test_entropy_fragment_ignores_non_positive_peaks_in_pairwise_matching():
    ref_with_zero = build_spectrum([100.0, 100.1], [0.0, 1.0], precursor_mz=250.0)
    ref_without_zero = build_spectrum([100.1], [1.0], precursor_mz=250.0)
    query = build_spectrum([100.05], [1.0], precursor_mz=250.0)

    fse = FlashEntropy(
        matching_mode="fragment",
        tolerance=0.1,
        use_ppm=False,
        remove_precursor=False,
        noise_cutoff=0.0,
        normalize_to_half=False,
        merge_within=0.0,
        dtype=np.float64,
    )

    expected = float(fse.pair(ref_without_zero, query))
    assert expected > 0.0
    assert float(fse.pair(ref_with_zero, query)) == pytest.approx(expected, abs=1e-7)

    matrix_scores = fse.matrix([ref_with_zero, ref_without_zero], [query], n_jobs=0, progress_bar=False)
    M = matrix_scores.to_array()
    assert float(M[0, 0]) == pytest.approx(expected, abs=1e-7)
    assert float(M[1, 0]) == pytest.approx(expected, abs=1e-7)


def test_entropy_fragment_matrix_matches_pair_with_sparse_candidate_columns():
    spectrum_1 = build_spectrum([100.0, 200.0], [1.0, 0.8], precursor_mz=450.0)
    spectra_2 = [
        build_spectrum([100.005, 200.003], [1.0, 0.7], precursor_mz=450.0),
        build_spectrum([100.01, 350.0], [0.9, 0.5], precursor_mz=450.0),
        build_spectrum([199.99], [0.8], precursor_mz=450.0),
    ]
    for shift in range(15):
        base = 500.0 + 10.0 * shift
        spectra_2.append(build_spectrum([base, base + 0.3], [1.0, 0.5], precursor_mz=450.0))

    fse = FlashEntropy(
        matching_mode="fragment",
        tolerance=0.02,
        use_ppm=False,
        remove_precursor=False,
        noise_cutoff=0.0,
        normalize_to_half=False,
        merge_within=0.0,
        dtype=np.float64,
    )

    matrix_scores = fse.matrix([spectrum_1], spectra_2, n_jobs=0, progress_bar=False)
    M = matrix_scores.to_array()
    assert M.shape == (1, len(spectra_2))
    assert np.count_nonzero(M[0] > 0.0) == 3

    for j, query in enumerate(spectra_2):
        expected = float(fse.pair(spectrum_1, query))
        assert float(M[0, j]) == pytest.approx(expected, abs=1e-7)


# ----------------------------
# Cosine / Modified Cosine path + baseline parity
# ----------------------------

def _mc_flash(tolerance, intensity_power=1.0):
    return FlashCosine(
        matching_mode="hybrid",  # hybrid = "modified cosine"
        tolerance=tolerance,
        intensity_power=intensity_power,
        remove_precursor=False,
        noise_cutoff=0.0,
        normalize_to_half=True,
        merge_within=0.0,
        dtype=np.float64,
    )


@pytest.mark.parametrize(
    "mz_a,int_a,pmz_a,mz_b,int_b,pmz_b,tol",
    [
        # 1) Clean shift, should match
        pytest.param(
            [100.0, 200.0, 300.0],
            [0.8, 1.0, 0.6],
            500.0,
            [110.0, 210.002, 300.005],
            [0.8, 1.0, 0.6],
            510.0,
            0.01,
            id="pure_shift_exploited",
        ),
        # 2) Mixed: one shared unshifted fragment plus shifted set
        # Expect both algorithms to pick the same best non-overlapping set.
        pytest.param(
            [100.0, 150.0, 200.0, 300.0],
            [0.8, 0.2, 1.0, 0.6],
            500.0,
            [110.0, 150.0, 210.0005, 310.0],
            [0.8, 0.2, 1.0, 0.6],
            510.0,
            0.01,
            id="mixed_direct_and_shifted",
        ),
        # 3) Ambiguity: two query peaks within tolerance of the same shifted match
        # This stresses greedy choice ordering and tie-breaking
        pytest.param(
            [100.0, 200.0],
            [1.0, 0.5],
            500.0,
            [110.0, 110.007, 210.0],
            [1.0, 1.0, 0.5],
            510.0,
            0.01,
            id="duplicate_candidates_nearby",
        ),
        # 4) Competition: a direct fragment match competes with a shifted/NL match
        # Constructed so dot products are close and greedy selection matters.
        pytest.param(
            [100.0, 200.0, 250.0],
            [1.0, 0.9, 0.2],
            500.0,
            [110.0, 200.0, 260.0],
            [1.0, 0.9, 0.2],
            510.0,
            0.01,
            id="direct_competes_with_shifted",
        ),
        # 5) Edge tolerance: shifted peaks sit near the boundary
        # To catch subtle differences in symmetric ppm/Da handling or window trimming.
        pytest.param(
            [100.0, 200.0, 300.0],
            [0.8, 1.0, 0.6],
            500.0,
            [110.0099, 210.0099, 310.0099],
            [0.8, 1.0, 0.6],
            510.0,
            0.01,
            id="near_tolerance_boundary",
        ),
    ],
)
def test_flash_hybrid_cosine_matches_modified_cosine_greedy(mz_a, int_a, pmz_a, mz_b, int_b, pmz_b, tol):
    a = build_spectrum(mz_a, int_a, precursor_mz=pmz_a)
    b = build_spectrum(mz_b, int_b, precursor_mz=pmz_b)

    for intensity_power in [1.0, 0.5]:  # Test with and without intensity weighting!
        flash = _mc_flash(tol, intensity_power=intensity_power)
        baseline = ModifiedCosineGreedy(tolerance=tol, intensity_power=intensity_power)

        s_flash = float(flash.pair(a, b)["score"])
        s_base = float(baseline.pair(a, b)["score"])
        matches_flash = flash.pair(a, b)["matches"]
        matches_base = baseline.pair(a, b)["matches"]

        assert s_flash == pytest.approx(s_base, rel=1e-12, abs=1e-12)
        assert matches_flash == matches_base


def test_cosine_pair_matches_cosinegreedy_default_tolerance_001():
    # Build a few pairs with clear fragment matches inside 0.01 Da
    pairs = [
        (
            build_spectrum([100.000, 150.000, 200.000, 300.000], [0.6, 1.0, 0.8, 0.4], 500.0),
            build_spectrum([100.005, 150.004, 200.007, 300.002], [0.6, 0.9, 0.8, 0.4], 500.0),
        ),
        (
            build_spectrum([80.0, 120.0, 250.0], [0.9, 0.7, 1.0], 420.0),
            build_spectrum([80.006, 120.004, 250.009], [0.9, 0.7, 1.0], 420.0),
        ),
    ]

    flash = FlashCosine(
        matching_mode="fragment",
        tolerance=0.01,
        remove_precursor=False,
        noise_cutoff=0.0,
        normalize_to_half=True,
        merge_within=0.0,
        dtype=np.float64,
    )
    baseline = CosineGreedy(tolerance=0.01)

    for a, b in pairs:
        s_flash = flash.pair(a, b)["score"]
        s_base = baseline.pair(a, b)["score"]
        assert s_flash == pytest.approx(s_base, rel=1e-12, abs=1e-12)

        matches_flash = flash.pair(a, b)["matches"]
        matches_base = baseline.pair(a, b)["matches"]
        assert matches_flash == matches_base


def test_cosine_matrix_dense_matches_pair():
    refs = [
        build_spectrum([100, 150, 300], [0.6, 1.0, 0.4], precursor_mz=500.0),
        build_spectrum([110, 250, 400], [0.5, 0.9, 0.7], precursor_mz=600.0),
    ]
    qs = [
        build_spectrum([100.007, 150.002, 300.000], [0.6, 1.0, 0.4], precursor_mz=500.0),
        build_spectrum([110.004, 250.009, 400.006], [0.5, 0.9, 0.7], precursor_mz=600.0),
    ]
    flash = FlashCosine(
        matching_mode="fragment",
        tolerance=0.01,
        remove_precursor=False,
        noise_cutoff=0.0,
        normalize_to_half=True,
        merge_within=0.0,
        dtype=np.float64,
    )
    scores = flash.matrix(refs, qs, n_jobs=0, progress_bar=False)
    assert isinstance(scores, Scores)
    M = scores.to_array("score")
    assert M.shape == (2, 2)
    for i, r in enumerate(refs):
        for j, q in enumerate(qs):
            expected = float(flash.pair(r, q)["score"])
            assert float(M[i, j]) == pytest.approx(expected, abs=1e-12)


def test_cosine_dtype_and_commutativity():
    a = build_spectrum([100, 150, 300], [0.5, 1.0, 0.4], precursor_mz=600.0)
    b = build_spectrum([100, 155, 295], [0.5, 0.8, 0.6], precursor_mz=600.0)
    f32 = FlashCosine(dtype=np.float32, remove_precursor=False, noise_cutoff=0.0)
    f64 = FlashCosine(dtype=np.float64, remove_precursor=False, noise_cutoff=0.0)

    s_ab_32 = f32.pair(a, b)["score"]
    s_ba_32 = f32.pair(b, a)["score"]
    s_ab_64 = f64.pair(a, b)["score"]
    s_ba_64 = f64.pair(b, a)["score"]

    assert s_ab_32.dtype == np.float32 and s_ba_32.dtype == np.float32
    assert s_ab_64.dtype == np.float64 and s_ba_64.dtype == np.float64
    assert float(s_ab_32) == pytest.approx(float(s_ba_32), 1e-6)
    assert float(s_ab_64) == pytest.approx(float(s_ba_64), 1e-12)
