import numpy as np
import pytest
from matchms import SpectraCollection
from matchms.scores import Scores
from matchms.similarity.flash_similarity import CosineFlash, EntropyFlash
from ..builder_spectrum import SpectrumBuilder


# -------------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------------


def _build_spectrum(mz, intensities, precursor_mz=None):
    """Create a Spectrum with optional precursor m/z metadata."""
    builder = (
        SpectrumBuilder()
        .with_mz(np.asarray(mz, dtype=float))
        .with_intensities(np.asarray(intensities, dtype=float))
    )

    if precursor_mz is not None:
        if hasattr(builder, "with_precursor_mz"):
            builder = builder.with_precursor_mz(float(precursor_mz))
        elif hasattr(builder, "with_metadata"):
            builder = builder.with_metadata(
                {"precursor_mz": float(precursor_mz)}
            )

    return builder.build()


def _build_collection(spectra):
    """Create a SpectraCollection with deterministic m/z precision."""
    return SpectraCollection(
        spectra,
        mz_precision=1e-6,
    )


def _example_entropy_inputs():
    references = [
        _build_spectrum(
            [100.0, 200.0],
            [1.0, 0.5],
            precursor_mz=500.0,
        ),
        _build_spectrum(
            [110.0, 300.0],
            [0.3, 1.0],
            precursor_mz=600.0,
        ),
        _build_spectrum(
            [125.0, 250.0, 400.0],
            [0.4, 1.0, 0.2],
            precursor_mz=550.0,
        ),
    ]
    queries = [
        _build_spectrum(
            [100.005, 210.0],
            [1.0, 0.5],
            precursor_mz=510.0,
        ),
        _build_spectrum(
            [110.0, 300.0],
            [1.0, 0.3],
            precursor_mz=600.0,
        ),
    ]
    return references, queries


def _example_cosine_inputs():
    references = [
        _build_spectrum(
            [100.0, 150.0, 300.0],
            [0.6, 1.0, 0.4],
            precursor_mz=500.0,
        ),
        _build_spectrum(
            [110.0, 250.0, 400.0],
            [0.5, 0.9, 0.7],
            precursor_mz=600.0,
        ),
    ]
    queries = [
        _build_spectrum(
            [100.007, 150.002, 300.0],
            [0.6, 1.0, 0.4],
            precursor_mz=500.0,
        ),
        _build_spectrum(
            [120.0, 260.0, 410.0],
            [0.5, 0.9, 0.7],
            precursor_mz=610.0,
        ),
    ]
    return references, queries


# -------------------------------------------------------------------------
# EntropyFlash: input handling and matrix API
# -------------------------------------------------------------------------


@pytest.mark.parametrize(
    "matching_mode",
    ["fragment", "neutral_loss", "hybrid"],
)
def test_flash_entropy_collection_input_matches_list_input(matching_mode):
    """List and SpectraCollection inputs should produce identical scores."""
    references, queries = _example_entropy_inputs()

    similarity = EntropyFlash(
        matching_mode=matching_mode,
        tolerance=0.01,
        remove_precursor=False,
        noise_cutoff=0.0,
        dtype=np.float64,
    )

    scores_from_lists = similarity.matrix(
        references,
        queries,
        n_jobs=1,
        progress_bar=False,
    )
    scores_from_collections = similarity.matrix(
        _build_collection(references),
        _build_collection(queries),
        n_jobs=1,
        progress_bar=False,
    )

    assert isinstance(scores_from_lists, Scores)
    assert scores_from_lists.score_fields == ("score",)
    assert scores_from_lists.shape == (3, 2)

    np.testing.assert_allclose(
        scores_from_collections.to_array(),
        scores_from_lists.to_array(),
        atol=1e-12,
        rtol=0.0,
    )


def test_flash_entropy_self_comparison_is_symmetric():
    """A self-comparison should be symmetric with unit diagonal."""
    spectra = [
        _build_spectrum(
            [100.0, 200.0],
            [1.0, 0.5],
            precursor_mz=500.0,
        ),
        _build_spectrum(
            [110.0, 300.0],
            [0.3, 1.0],
            precursor_mz=600.0,
        ),
        _build_spectrum(
            [100.0, 250.0],
            [0.7, 0.9],
            precursor_mz=550.0,
        ),
    ]

    similarity = EntropyFlash(
        matching_mode="fragment",
        tolerance=0.01,
        remove_precursor=False,
        noise_cutoff=0.0,
        dtype=np.float64,
    )

    scores = similarity.matrix(
        _build_collection(spectra),
        n_jobs=1,
        progress_bar=False,
    ).to_array()

    np.testing.assert_allclose(
        scores,
        scores.T,
        atol=1e-12,
        rtol=0.0,
    )
    np.testing.assert_allclose(
        np.diag(scores),
        np.ones(len(spectra)),
        atol=1e-12,
        rtol=0.0,
    )


def test_flash_entropy_pair_matches_matrix_element():
    """pair() and matrix() should use the same scoring semantics."""
    reference = _build_spectrum(
        [100.0, 200.0],
        [1.0, 0.5],
        precursor_mz=500.0,
    )
    query = _build_spectrum(
        [100.005, 210.0],
        [1.0, 0.5],
        precursor_mz=510.0,
    )

    similarity = EntropyFlash(
        matching_mode="hybrid",
        tolerance=0.01,
        remove_precursor=False,
        noise_cutoff=0.0,
        dtype=np.float64,
    )

    pair_score = float(similarity.pair(reference, query))
    matrix_score = similarity.matrix(
        [reference],
        [query],
        n_jobs=1,
        progress_bar=False,
    ).to_array()[0, 0]

    assert matrix_score == pytest.approx(pair_score, abs=1e-12)


def test_flash_entropy_rectangular_matrix_preserves_requested_orientation():
    """Rows must correspond to spectra_1 and columns to spectra_2."""
    references, queries = _example_entropy_inputs()

    similarity = EntropyFlash(
        matching_mode="fragment",
        tolerance=0.01,
        remove_precursor=False,
        noise_cutoff=0.0,
        dtype=np.float64,
    )

    scores = similarity.matrix(
        references,
        queries,
        n_jobs=1,
        progress_bar=False,
    ).to_array()

    assert scores.shape == (len(references), len(queries))

    for reference_index, reference in enumerate(references):
        for query_index, query in enumerate(queries):
            expected = float(similarity.pair(reference, query))
            assert scores[reference_index, query_index] == pytest.approx(
                expected,
                abs=1e-12,
            )


# -------------------------------------------------------------------------
# EntropyFlash: matching modes and precursor handling
# -------------------------------------------------------------------------


def test_flash_entropy_neutral_loss_requires_precursor_mz():
    """Neutral-loss matching should score zero without precursor metadata."""
    reference = _build_spectrum([100.0], [1.0])
    query = _build_spectrum([100.0], [1.0])

    similarity = EntropyFlash(
        matching_mode="neutral_loss",
        tolerance=0.01,
        remove_precursor=False,
        noise_cutoff=0.0,
        dtype=np.float64,
    )

    score = similarity.matrix(
        [reference],
        [query],
        n_jobs=1,
        progress_bar=False,
    ).to_array()[0, 0]

    assert score == 0.0


def test_flash_entropy_hybrid_combines_distinct_fragment_and_loss_matches():
    """Hybrid mode should combine non-overlapping fragment and loss matches."""
    reference = _build_spectrum(
        [100.0, 200.0],
        [1.0, 1.0],
        precursor_mz=500.0,
    )
    query = _build_spectrum(
        [100.0, 210.0],
        [1.0, 1.0],
        precursor_mz=510.0,
    )

    similarity = EntropyFlash(
        matching_mode="hybrid",
        tolerance=0.01,
        remove_precursor=False,
        noise_cutoff=0.0,
        dtype=np.float64,
    )

    score = similarity.matrix(
        [reference],
        [query],
        n_jobs=1,
        progress_bar=False,
    ).to_array()[0, 0]

    assert score == pytest.approx(1.0, abs=1e-12)


def test_flash_entropy_identity_precursor_gate_excludes_distant_precursor():
    """Identity gating should remove scores outside the precursor window."""
    reference = _build_spectrum(
        [100.0, 200.0],
        [1.0, 1.0],
        precursor_mz=500.0,
    )
    query = _build_spectrum(
        [100.0, 200.0],
        [1.0, 1.0],
        precursor_mz=500.3,
    )

    similarity = EntropyFlash(
        matching_mode="fragment",
        tolerance=0.01,
        identity_precursor_tolerance=0.2,
        identity_use_ppm=False,
        remove_precursor=False,
        noise_cutoff=0.0,
        dtype=np.float64,
    )

    score = similarity.matrix(
        [reference],
        [query],
        n_jobs=1,
        progress_bar=False,
    ).to_array()[0, 0]

    assert score == 0.0


# -------------------------------------------------------------------------
# CosineFlash: input handling and matrix API
# -------------------------------------------------------------------------


@pytest.mark.parametrize(
    "matching_mode",
    ["fragment", "hybrid"],
)
@pytest.mark.parametrize(
    "intensity_power",
    [1.0, 0.5],
)
def test_cosine_flash_collection_input_matches_list_input(
    matching_mode,
    intensity_power,
):
    """Packed collection input should reproduce list-input score and match fields."""
    references, queries = _example_cosine_inputs()

    similarity = CosineFlash(
        matching_mode=matching_mode,
        tolerance=0.01,
        intensity_power=intensity_power,
        remove_precursor=False,
        noise_cutoff=0.0,
        dtype=np.float64,
    )

    scores_from_lists = similarity.matrix(
        references,
        queries,
        n_jobs=1,
        progress_bar=False,
    )
    scores_from_collections = similarity.matrix(
        _build_collection(references),
        _build_collection(queries),
        n_jobs=1,
        progress_bar=False,
    )

    assert isinstance(scores_from_lists, Scores)
    assert scores_from_lists.score_fields == ("score", "matches")
    assert scores_from_lists.shape == (2, 2)

    np.testing.assert_allclose(
        scores_from_collections.to_array("score"),
        scores_from_lists.to_array("score"),
        atol=1e-12,
        rtol=0.0,
    )
    np.testing.assert_array_equal(
        scores_from_collections.to_array("matches"),
        scores_from_lists.to_array("matches"),
    )


@pytest.mark.parametrize("intensity_power", [1.0, 0.5])
def test_cosine_flash_pair_matches_matrix_element(intensity_power):
    """pair() should return the same score and match count as matrix()."""
    reference = _build_spectrum(
        [100.0, 150.0, 300.0],
        [0.6, 1.0, 0.4],
        precursor_mz=500.0,
    )
    query = _build_spectrum(
        [100.005, 150.005, 300.0],
        [0.6, 0.9, 0.4],
        precursor_mz=500.0,
    )

    similarity = CosineFlash(
        matching_mode="fragment",
        tolerance=0.01,
        intensity_power=intensity_power,
        remove_precursor=False,
        noise_cutoff=0.0,
        dtype=np.float64,
    )

    pair_score = similarity.pair(reference, query)
    matrix_scores = similarity.matrix(
        [reference],
        [query],
        n_jobs=1,
        progress_bar=False,
    )

    assert float(pair_score["score"]) == pytest.approx(
        matrix_scores.to_array("score")[0, 0],
        abs=1e-12,
    )
    assert int(pair_score["matches"]) == int(
        matrix_scores.to_array("matches")[0, 0]
    )


def test_cosine_flash_score_field_selection():
    """matrix() should calculate only the explicitly requested score fields."""
    spectrum = _build_spectrum(
        [100.0, 200.0],
        [1.0, 0.5],
        precursor_mz=500.0,
    )

    similarity = CosineFlash(
        tolerance=0.01,
        remove_precursor=False,
        noise_cutoff=0.0,
    )

    score_only = similarity.matrix(
        [spectrum],
        score_fields=("score",),
        n_jobs=1,
        progress_bar=False,
    )
    matches_only = similarity.matrix(
        [spectrum],
        score_fields=("matches",),
        n_jobs=1,
        progress_bar=False,
    )

    assert score_only.score_fields == ("score",)
    assert matches_only.score_fields == ("matches",)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_cosine_flash_pair_respects_dtype(dtype):
    """The score field should use the configured floating-point dtype."""
    spectrum = _build_spectrum(
        [100.0],
        [1.0],
        precursor_mz=500.0,
    )

    score = CosineFlash(
        dtype=dtype,
        remove_precursor=False,
        noise_cutoff=0.0,
    ).pair(
        spectrum,
        spectrum,
    )

    assert score["score"].dtype == np.dtype(dtype)


def test_cosine_flash_rectangular_matrix_preserves_requested_orientation():
    """Rectangular output must follow spectra_1 x spectra_2 orientation."""
    references, queries = _example_cosine_inputs()

    similarity = CosineFlash(
        matching_mode="fragment",
        tolerance=0.01,
        remove_precursor=False,
        noise_cutoff=0.0,
        dtype=np.float64,
    )

    scores = similarity.matrix(
        references,
        queries,
        n_jobs=1,
        progress_bar=False,
    )

    assert scores.shape == (len(references), len(queries))

    score_array = scores.to_array("score")
    match_array = scores.to_array("matches")

    for reference_index, reference in enumerate(references):
        for query_index, query in enumerate(queries):
            expected = similarity.pair(reference, query)

            assert score_array[reference_index, query_index] == pytest.approx(
                float(expected["score"]),
                abs=1e-12,
            )
            assert match_array[reference_index, query_index] == int(
                expected["matches"]
            )
