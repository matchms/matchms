import numpy as np
import pytest
from matchms.scores import Scores
from matchms.similarity import Entropy, EntropyGreedy
from matchms.similarity.flash_similarity import FlashEntropy
from ..builder_spectrum import SpectrumBuilder


# ----------------------------
# Helpers
# ----------------------------


def build_spectrum(mz, intens, precursor_mz=None):
    """Build a Spectrum via SpectrumBuilder, setting precursor_mz when available."""
    builder = SpectrumBuilder().with_mz(np.asarray(mz, dtype="float")).with_intensities(
        np.asarray(intens, dtype="float")
    )
    if hasattr(builder, "with_precursor_mz") and precursor_mz is not None:
        builder = builder.with_precursor_mz(float(precursor_mz))
    elif precursor_mz is not None and hasattr(builder, "with_metadata"):
        builder = builder.with_metadata({"precursor_mz": float(precursor_mz)})
    return builder.build()


def entropy_increment(intensity_1, intensity_2):
    """Entropy-similarity contribution of one matched weighted peak pair."""
    def xlog2(x):
        return 0.0 if x <= 0.0 else x * np.log2(x)

    return (
        xlog2(intensity_1 + intensity_2)
        - xlog2(intensity_1)
        - xlog2(intensity_2)
    )


# ----------------------------
# EntropyGreedy: fundamental scoring behavior
# ----------------------------


def test_entropy_greedy_identical_spectra_score_one():
    spectrum = build_spectrum([100.0, 150.0, 200.0], [0.7, 0.2, 0.1])

    score = EntropyGreedy(tolerance=0.02).pair(spectrum, spectrum)

    assert float(score) == pytest.approx(1.0, abs=1e-12)


def test_entropy_greedy_nonmatching_spectra_score_zero():
    spectrum_1 = build_spectrum([100.0, 150.0], [0.8, 0.2])
    spectrum_2 = build_spectrum([110.0, 160.0], [0.8, 0.2])

    score = EntropyGreedy(tolerance=0.02).pair(spectrum_1, spectrum_2)

    assert float(score) == 0.0


def test_entropy_greedy_one_of_two_equal_peaks_matching_has_known_score():
    """Independent analytical check, without using FlashEntropy as oracle.

    With two equal peaks per spectrum, entropy weighting keeps both peaks equal.
    Normalization to sum 0.5 therefore gives intensity 0.25 per peak. If exactly
    one peak matches, its contribution is exactly 0.5.
    """
    spectrum_1 = build_spectrum([100.0, 200.0], [1.0, 1.0])
    spectrum_2 = build_spectrum([100.0, 300.0], [5.0, 5.0])

    score = EntropyGreedy(tolerance=0.02, noise_cutoff=0.0).pair(
        spectrum_1, spectrum_2
    )

    assert float(score) == pytest.approx(0.5, abs=1e-12)


def test_entropy_greedy_is_invariant_to_intensity_scaling():
    spectrum_1 = build_spectrum([100.0, 150.0, 200.0], [1.0, 2.0, 4.0])
    spectrum_2 = build_spectrum([100.0, 150.0, 200.0], [10.0, 20.0, 40.0])

    score = EntropyGreedy(tolerance=0.02, noise_cutoff=0.0).pair(
        spectrum_1, spectrum_2
    )

    assert float(score) == pytest.approx(1.0, abs=1e-12)


def test_entropy_greedy_is_commutative():
    spectrum_1 = build_spectrum([100.00, 150.00, 200.00], [10.0, 2.0, 1.0])
    spectrum_2 = build_spectrum([100.01, 150.03, 220.00], [5.0, 3.0, 1.0])
    similarity = EntropyGreedy(tolerance=0.02)

    score_12 = similarity.pair(spectrum_1, spectrum_2)
    score_21 = similarity.pair(spectrum_2, spectrum_1)

    assert float(score_12) == pytest.approx(float(score_21), abs=1e-12)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_entropy_greedy_pair_respects_dtype(dtype):
    spectrum_1 = build_spectrum([100.0, 200.0], [1.0, 0.5])
    spectrum_2 = build_spectrum([100.01, 250.0], [1.0, 0.5])

    score = EntropyGreedy(tolerance=0.02, dtype=dtype).pair(spectrum_1, spectrum_2)

    assert score.dtype == np.dtype(dtype)


def test_entropy_greedy_empty_spectrum_returns_zero():
    empty = build_spectrum([], [])
    nonempty = build_spectrum([100.0], [1.0])

    similarity = EntropyGreedy(noise_cutoff=0.0)

    assert float(similarity.pair(empty, nonempty)) == 0.0
    assert float(similarity.pair(nonempty, empty)) == 0.0


def test_entropy_greedy_all_zero_intensities_return_zero():
    spectrum_1 = build_spectrum([100.0, 200.0], [0.0, 0.0])
    spectrum_2 = build_spectrum([100.0, 200.0], [1.0, 1.0])

    score = EntropyGreedy(noise_cutoff=0.0).pair(spectrum_1, spectrum_2)

    assert float(score) == 0.0


def test_entropy_greedy_ignores_zero_intensity_peaks_during_matching():
    reference_with_zero = build_spectrum([100.0, 100.1], [0.0, 1.0])
    reference_without_zero = build_spectrum([100.1], [1.0])
    query = build_spectrum([100.05], [1.0])

    similarity = EntropyGreedy(
        tolerance=0.1,
        noise_cutoff=0.0,
        dtype=np.float64,
    )

    expected = float(similarity.pair(reference_without_zero, query))
    actual = float(similarity.pair(reference_with_zero, query))

    assert expected == pytest.approx(1.0, abs=1e-12)
    assert actual == pytest.approx(expected, abs=1e-12)


def test_entropy_greedy_overlapping_windows_are_one_to_one_and_bounded():
    """One query peak may not contribute multiple times to the score."""
    spectrum_1 = build_spectrum([100.000, 100.010], [1.0, 1.0])
    spectrum_2 = build_spectrum([100.005], [1.0])

    score = EntropyGreedy(
        tolerance=0.02,
        noise_cutoff=0.0,
        dtype=np.float64,
    ).pair(spectrum_1, spectrum_2)

    # After entropy weighting + normalization, spectrum_1 contains two peaks at
    # 0.25 and spectrum_2 one peak at 0.5. Only one of the two possible matches
    # may be consumed.
    expected = entropy_increment(0.25, 0.5)
    assert float(score) == pytest.approx(expected, abs=1e-12)
    assert 0.0 <= float(score) <= 1.0


# ----------------------------
# EntropyGreedy: tolerance handling
# ----------------------------


@pytest.mark.parametrize(
    "delta,expected",
    [
        pytest.param(0.019999, 1.0, id="inside"),
        pytest.param(0.020000, 1.0, id="exact_boundary"),
        pytest.param(0.020001, 0.0, id="outside"),
    ],
)
def test_entropy_greedy_da_tolerance_boundary(delta, expected):
    spectrum_1 = build_spectrum([100.0], [1.0])
    spectrum_2 = build_spectrum([100.0 + delta], [1.0])

    score = EntropyGreedy(
        tolerance=0.02,
        use_ppm=False,
        noise_cutoff=0.0,
    ).pair(spectrum_1, spectrum_2)

    assert float(score) == pytest.approx(expected, abs=1e-12)


@pytest.mark.parametrize(
    "query_mz,expected",
    [
        pytest.param(1000.009, 1.0, id="inside"),
        pytest.param(1000.011, 0.0, id="outside"),
    ],
)
def test_entropy_greedy_symmetric_ppm_tolerance(query_mz, expected):
    spectrum_1 = build_spectrum([1000.0], [1.0])
    spectrum_2 = build_spectrum([query_mz], [1.0])

    score = EntropyGreedy(
        tolerance=10.0,
        use_ppm=True,
        noise_cutoff=0.0,
    ).pair(spectrum_1, spectrum_2)

    assert float(score) == pytest.approx(expected, abs=1e-12)


@pytest.mark.parametrize(
    "kwargs",
    [
        pytest.param({"tolerance": -0.01}, id="negative_tolerance"),
        pytest.param({"merge_within": -0.01}, id="negative_merge_window"),
    ],
)
def test_entropy_greedy_rejects_negative_windows(kwargs):
    with pytest.raises(ValueError):
        EntropyGreedy(**kwargs)


# ----------------------------
# EntropyGreedy: preprocessing behavior
# ----------------------------


def test_entropy_greedy_noise_cutoff_removes_small_unmatched_peak():
    reference = build_spectrum([100.0, 200.0], [1.0, 0.005])
    query = build_spectrum([100.0], [1.0])

    with_filter = EntropyGreedy(
        tolerance=0.02,
        noise_cutoff=0.01,
    ).pair(reference, query)
    without_filter = EntropyGreedy(
        tolerance=0.02,
        noise_cutoff=0.0,
    ).pair(reference, query)

    assert float(with_filter) == pytest.approx(1.0, abs=1e-12)
    assert 0.0 < float(without_filter) < 1.0


def test_entropy_greedy_remove_precursor_can_remove_all_peaks():
    spectrum_1 = build_spectrum([199.0, 199.5], [1.0, 0.5], precursor_mz=200.0)
    spectrum_2 = build_spectrum([199.2, 199.7], [1.0, 0.5], precursor_mz=200.0)

    score = EntropyGreedy(
        tolerance=0.02,
        remove_precursor=True,
        offset_to_precursor=-1.6,
        noise_cutoff=0.0,
    ).pair(spectrum_1, spectrum_2)

    assert float(score) == 0.0


def test_entropy_greedy_remove_precursor_keeps_cutoff_boundary():
    spectrum_1 = build_spectrum(
        [100.0, 198.4, 199.0],
        [1.0, 1.0, 20.0],
        precursor_mz=200.0,
    )
    spectrum_2 = build_spectrum(
        [100.0, 198.4, 199.2],
        [1.0, 1.0, 30.0],
        precursor_mz=200.0,
    )

    score = EntropyGreedy(
        tolerance=0.02,
        remove_precursor=True,
        offset_to_precursor=-1.6,
        noise_cutoff=0.0,
    ).pair(spectrum_1, spectrum_2)

    assert float(score) == pytest.approx(1.0, abs=1e-12)


def test_entropy_greedy_remove_precursor_without_precursor_metadata_is_noop():
    spectrum_1 = build_spectrum([100.0, 199.0], [1.0, 2.0])
    spectrum_2 = build_spectrum([100.0, 199.0], [1.0, 2.0])

    score_remove = EntropyGreedy(
        remove_precursor=True,
        noise_cutoff=0.0,
    ).pair(spectrum_1, spectrum_2)
    score_keep = EntropyGreedy(
        remove_precursor=False,
        noise_cutoff=0.0,
    ).pair(spectrum_1, spectrum_2)

    assert float(score_remove) == pytest.approx(float(score_keep), abs=1e-12)
    assert float(score_remove) == pytest.approx(1.0, abs=1e-12)


def test_entropy_greedy_merge_within_can_restore_peak_match():
    spectrum_1 = build_spectrum([100.000, 100.010], [1.0, 1.0])
    spectrum_2 = build_spectrum([100.004, 100.006], [1.0, 1.0])

    without_merge = EntropyGreedy(
        tolerance=0.001,
        merge_within=0.0,
        noise_cutoff=0.0,
    ).pair(spectrum_1, spectrum_2)
    with_merge = EntropyGreedy(
        tolerance=0.001,
        merge_within=0.02,
        noise_cutoff=0.0,
    ).pair(spectrum_1, spectrum_2)

    assert float(without_merge) == 0.0
    assert float(with_merge) == pytest.approx(1.0, abs=1e-12)


# ----------------------------
# EntropyGreedy vs FlashEntropy parity
# ----------------------------


@pytest.mark.parametrize(
    "mz_a,int_a,mz_b,int_b,tol,use_ppm,noise_cutoff,merge_within",
    [
        pytest.param(
            [100.0, 200.0, 300.0],
            [0.8, 1.0, 0.6],
            [100.005, 200.005, 350.0],
            [0.7, 1.0, 0.4],
            0.01,
            False,
            0.0,
            0.0,
            id="ordinary_da_matches",
        ),
        pytest.param(
            [100.000, 100.010, 200.0],
            [1.0, 0.5, 0.2],
            [100.005, 100.015, 250.0],
            [0.8, 0.7, 0.3],
            0.02,
            False,
            0.0,
            0.0,
            id="overlapping_tolerance_windows",
        ),
        pytest.param(
            [100.0, 200.0],
            [1.0, 0.009],
            [100.0, 300.0],
            [1.0, 0.009],
            0.02,
            False,
            0.01,
            0.0,
            id="noise_filtering",
        ),
        pytest.param(
            [1000.000, 1200.000],
            [1.0, 0.5],
            [1000.009, 1200.011],
            [0.8, 0.6],
            10.0,
            True,
            0.0,
            0.0,
            id="ppm_matching",
        ),
        pytest.param(
            [100.000, 100.010, 200.0],
            [1.0, 1.0, 0.5],
            [100.004, 100.006, 200.0],
            [1.0, 1.0, 0.5],
            0.001,
            False,
            0.0,
            0.02,
            id="peak_merging",
        ),
        pytest.param(
            [100.0, 150.0],
            [1.0, 1.0],
            [300.0, 350.0],
            [1.0, 1.0],
            0.02,
            False,
            0.0,
            0.0,
            id="no_matches",
        ),
    ],
)
def test_entropy_greedy_matches_flash_entropy_fragment_pair(
    mz_a,
    int_a,
    mz_b,
    int_b,
    tol,
    use_ppm,
    noise_cutoff,
    merge_within,
):
    spectrum_a = build_spectrum(mz_a, int_a, precursor_mz=500.0)
    spectrum_b = build_spectrum(mz_b, int_b, precursor_mz=500.0)

    kwargs = {
        "tolerance": tol,
        "use_ppm": use_ppm,
        "remove_precursor": False,
        "noise_cutoff": noise_cutoff,
        "merge_within": merge_within,
        "dtype": np.float64,
    }

    baseline = EntropyGreedy(**kwargs)
    flash = FlashEntropy(
        matching_mode="fragment",
        normalize_to_half=True,
        **kwargs,
    )

    baseline_score = float(baseline.pair(spectrum_a, spectrum_b))
    flash_score = float(flash.pair(spectrum_a, spectrum_b))

    assert baseline_score == pytest.approx(flash_score, rel=1e-12, abs=1e-12)


def test_entropy_greedy_matches_flash_entropy_after_precursor_cleanup():
    spectrum_1 = build_spectrum(
        [100.0, 198.8, 199.5],
        [1.0, 10.0, 100.0],
        precursor_mz=200.0,
    )
    spectrum_2 = build_spectrum(
        [100.0, 198.9, 199.4],
        [1.0, 10.0, 100.0],
        precursor_mz=200.0,
    )

    kwargs = {
        "tolerance": 0.02,
        "remove_precursor": True,
        "offset_to_precursor": -1.5,
        "noise_cutoff": 0.0,
        "dtype": np.float64,
    }

    baseline = EntropyGreedy(**kwargs).pair(spectrum_1, spectrum_2)
    flash = FlashEntropy(
        matching_mode="fragment",
        normalize_to_half=True,
        **kwargs,
    ).pair(spectrum_1, spectrum_2)

    assert float(baseline) == pytest.approx(float(flash), abs=1e-12)


# ----------------------------
# Entropy: public wrapper / go-to implementation
# ----------------------------


def test_entropy_pair_matches_entropy_greedy_with_same_configuration():
    spectrum_1 = build_spectrum(
        [100.0, 100.01, 200.0, 499.0],
        [1.0, 0.5, 0.2, 0.01],
        precursor_mz=500.0,
    )
    spectrum_2 = build_spectrum(
        [100.004, 100.007, 200.01, 499.2],
        [0.8, 0.7, 0.3, 0.02],
        precursor_mz=500.0,
    )
    kwargs = {
        "tolerance": 0.02,
        "use_ppm": False,
        "remove_precursor": True,
        "offset_to_precursor": -0.5,
        "noise_cutoff": 0.0,
        "merge_within": 0.015,
        "dtype": np.float64,
    }

    score = Entropy(**kwargs).pair(spectrum_1, spectrum_2)
    expected = EntropyGreedy(**kwargs).pair(spectrum_1, spectrum_2)

    assert float(score) == pytest.approx(float(expected), abs=1e-12)


def test_entropy_matrix_matches_flash_entropy_with_same_configuration():
    references = [
        build_spectrum([100.0, 200.0], [1.0, 0.5], precursor_mz=500.0),
        build_spectrum([110.0, 300.0], [0.3, 1.0], precursor_mz=600.0),
    ]
    queries = [
        build_spectrum([100.005, 205.0], [1.0, 0.5], precursor_mz=500.0),
        build_spectrum([110.005, 300.0], [1.0, 0.3], precursor_mz=600.0),
    ]
    kwargs = {
        "tolerance": 0.01,
        "use_ppm": False,
        "remove_precursor": False,
        "offset_to_precursor": -1.6,
        "noise_cutoff": 0.0,
        "merge_within": 0.0,
        "dtype": np.float64,
    }

    scores = Entropy(**kwargs).matrix(
        references,
        queries,
        progress_bar=False,
        n_jobs=0,
    )
    expected = FlashEntropy(
        matching_mode="fragment",
        normalize_to_half=True,
        **kwargs,
    ).matrix(
        references,
        queries,
        progress_bar=False,
        n_jobs=0,
    )

    assert np.allclose(scores.to_array(), expected.to_array(), atol=1e-12, rtol=1e-12)


def test_entropy_matrix_dense_matches_pair_for_rectangular_input():
    references = [
        build_spectrum([100.0, 200.0], [1.0, 0.5], precursor_mz=500.0),
        build_spectrum([110.0, 300.0], [0.3, 1.0], precursor_mz=600.0),
    ]
    queries = [
        build_spectrum([100.005, 205.0], [1.0, 0.5], precursor_mz=500.0),
        build_spectrum([110.005, 300.0], [1.0, 0.3], precursor_mz=600.0),
        build_spectrum([700.0], [1.0], precursor_mz=800.0),
    ]
    similarity = Entropy(
        tolerance=0.01,
        noise_cutoff=0.0,
        dtype=np.float64,
    )

    scores = similarity.matrix(
        references,
        queries,
        n_jobs=0,
        progress_bar=False,
    )

    assert isinstance(scores, Scores)
    assert scores.is_sparse is False
    assert scores.is_scalar is True

    matrix = scores.to_array()
    assert matrix.shape == (2, 3)

    for i, reference in enumerate(references):
        for j, query in enumerate(queries):
            expected = float(similarity.pair(reference, query))
            assert float(matrix[i, j]) == pytest.approx(expected, abs=1e-12)


def test_entropy_matrix_self_comparison_is_symmetric_with_unit_diagonal():
    spectra = [
        build_spectrum([100.0, 200.0], [1.0, 0.5]),
        build_spectrum([110.0, 300.0], [0.3, 1.0]),
        build_spectrum([100.01, 250.0], [0.7, 0.2]),
    ]
    similarity = Entropy(
        tolerance=0.02,
        noise_cutoff=0.0,
        dtype=np.float64,
    )

    scores = similarity.matrix(spectra, n_jobs=0, progress_bar=False)
    matrix = scores.to_array()

    assert isinstance(scores, Scores)
    assert matrix.shape == (3, 3)
    assert np.allclose(matrix, matrix.T, atol=1e-12, rtol=0.0)
    assert np.allclose(np.diag(matrix), 1.0, atol=1e-12, rtol=0.0)


def test_entropy_matrix_matches_pair_with_many_non_candidate_columns():
    reference = build_spectrum([100.0, 200.0], [1.0, 0.8], precursor_mz=450.0)
    queries = [
        build_spectrum([100.005, 200.003], [1.0, 0.7], precursor_mz=450.0),
        build_spectrum([100.01, 350.0], [0.9, 0.5], precursor_mz=450.0),
        build_spectrum([199.99], [0.8], precursor_mz=450.0),
    ]
    for shift in range(15):
        base = 500.0 + 10.0 * shift
        queries.append(
            build_spectrum([base, base + 0.3], [1.0, 0.5], precursor_mz=450.0)
        )

    similarity = Entropy(
        tolerance=0.02,
        noise_cutoff=0.0,
        dtype=np.float64,
    )

    scores = similarity.matrix(
        [reference],
        queries,
        n_jobs=0,
        progress_bar=False,
    )
    matrix = scores.to_array()

    assert matrix.shape == (1, len(queries))
    assert np.count_nonzero(matrix[0] > 0.0) == 3

    for j, query in enumerate(queries):
        expected = float(similarity.pair(reference, query))
        assert float(matrix[0, j]) == pytest.approx(expected, abs=1e-12)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_entropy_pair_and_matrix_respect_dtype(dtype):
    spectrum_1 = build_spectrum([100.0, 200.0], [1.0, 0.5])
    spectrum_2 = build_spectrum([100.01, 250.0], [0.8, 0.4])
    similarity = Entropy(
        tolerance=0.02,
        noise_cutoff=0.0,
        dtype=dtype,
    )

    pair_score = similarity.pair(spectrum_1, spectrum_2)
    matrix_score = similarity.matrix(
        [spectrum_1],
        [spectrum_2],
        n_jobs=0,
        progress_bar=False,
    ).to_array()

    assert pair_score.dtype == np.dtype(dtype)
    assert matrix_score.dtype == np.dtype(dtype)
    tolerance = 1e-6 if dtype is np.float32 else 1e-12
    assert float(matrix_score[0, 0]) == pytest.approx(float(pair_score), abs=tolerance)


def test_entropy_matrix_supports_explicit_score_field():
    spectrum = build_spectrum([100.0, 200.0], [1.0, 0.5])

    scores = Entropy().matrix(
        [spectrum],
        score_fields=("score",),
        n_jobs=0,
        progress_bar=False,
    )

    assert scores.score_fields == ("score",)
    assert scores.shape == (1, 1)


def test_entropy_matrix_rejects_unknown_score_field():
    spectrum = build_spectrum([100.0, 200.0], [1.0, 0.5])

    with pytest.raises(ValueError, match="Unknown score field"):
        Entropy().matrix(
            [spectrum],
            score_fields=("matches",),
            n_jobs=0,
            progress_bar=False,
        )
