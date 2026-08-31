import numpy as np
import pytest
from matchms import SpectraCollection
from matchms.scores import Scores
from matchms.similarity.flash_similarity import (
    CosineFlash,
    FlashEntropy,
)
from matchms.similarity.flash_similarity_spectrum_list import (
    CosineFlash as CosineFlashSL,
)
from matchms.similarity.flash_similarity_spectrum_list import (
    FlashEntropy as FlashEntropySL,
)
from ..builder_spectrum import SpectrumBuilder


# -------------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------------

def build_spectrum(
    mz,
    intens,
    precursor_mz=None,
):
    builder = (
        SpectrumBuilder()
        .with_mz(
            np.asarray(
                mz,
                dtype=float,
            )
        )
        .with_intensities(
            np.asarray(
                intens,
                dtype=float,
            )
        )
    )

    if precursor_mz is not None:
        if hasattr(
            builder,
            "with_precursor_mz",
        ):
            builder = builder.with_precursor_mz(
                float(precursor_mz)
            )
        elif hasattr(
            builder,
            "with_metadata",
        ):
            builder = builder.with_metadata(
                {
                    "precursor_mz":
                        float(precursor_mz)
                }
            )

    return builder.build()


def build_collection(spectra):
    return SpectraCollection(
        spectra,
        mz_precision=1e-6,
    )


# -------------------------------------------------------------------------
# Input handling
# -------------------------------------------------------------------------

def test_entropy_sc_matrix_accepts_spectrum_lists():
    spectra = [
        build_spectrum(
            [100.0, 200.0],
            [1.0, 0.5],
            500.0,
        ),
        build_spectrum(
            [110.0, 300.0],
            [0.3, 1.0],
            600.0,
        ),
    ]

    similarity = FlashEntropySL(
        tolerance=0.01,
        remove_precursor=False,
        noise_cutoff=0.0,
        dtype=np.float64,
    )

    scores = similarity.matrix(
        spectra,
        n_jobs=0,
        progress_bar=False,
    )

    assert isinstance(
        scores,
        Scores,
    )
    assert scores.shape == (2, 2)


def test_entropy_sc_matrix_accepts_spectra_collection():
    spectra = [
        build_spectrum(
            [100.0, 200.0],
            [1.0, 0.5],
            500.0,
        ),
        build_spectrum(
            [110.0, 300.0],
            [0.3, 1.0],
            600.0,
        ),
    ]
    collection = build_collection(
        spectra,
    )

    similarity = FlashEntropySL(
        tolerance=0.01,
        remove_precursor=False,
        noise_cutoff=0.0,
        dtype=np.float64,
    )

    scores = similarity.matrix(
        collection,
        n_jobs=0,
        progress_bar=False,
    )

    assert scores.shape == (2, 2)


# -------------------------------------------------------------------------
# Flash entropy
# -------------------------------------------------------------------------

@pytest.mark.parametrize(
    "matching_mode",
    [
        "fragment",
        "neutral_loss",
        "hybrid",
    ],
)
def test_entropy_sc_matches_existing_flash_matrix(
    matching_mode,
):
    references = [
        build_spectrum(
            [100.0, 200.0],
            [1.0, 0.5],
            500.0,
        ),
        build_spectrum(
            [110.0, 300.0],
            [0.3, 1.0],
            600.0,
        ),
        build_spectrum(
            [125.0, 250.0, 400.0],
            [0.4, 1.0, 0.2],
            550.0,
        ),
    ]

    queries = [
        build_spectrum(
            [100.005, 210.0],
            [1.0, 0.5],
            510.0,
        ),
        build_spectrum(
            [110.0, 300.0],
            [1.0, 0.3],
            600.0,
        ),
    ]

    kwargs = {
        "matching_mode": matching_mode,
        "tolerance": 0.01,
        "use_ppm": False,
        "remove_precursor": False,
        "noise_cutoff": 0.0,
        "normalize_to_half": True,
        "merge_within": 0.0,
        "dtype": np.float64,
    }

    old = FlashEntropy(
        **kwargs,
    )
    sc = FlashEntropySL(
        **kwargs,
    )

    expected = old.matrix(
        references,
        queries,
        n_jobs=0,
        progress_bar=False,
    ).to_array()

    actual = sc.matrix(
        references,
        queries,
        n_jobs=0,
        progress_bar=False,
    ).to_array()

    assert actual.shape == expected.shape
    assert np.allclose(
        actual,
        expected,
        atol=1e-12,
    )


@pytest.mark.parametrize(
    "matching_mode",
    [
        "fragment",
        "neutral_loss",
        "hybrid",
    ],
)
def test_entropy_sc_collection_input_matches_list_input(
    matching_mode,
):
    references = [
        build_spectrum(
            [100.0, 200.0],
            [1.0, 0.5],
            500.0,
        ),
        build_spectrum(
            [110.0, 300.0],
            [0.3, 1.0],
            600.0,
        ),
    ]

    queries = [
        build_spectrum(
            [100.005, 210.0],
            [1.0, 0.5],
            510.0,
        ),
        build_spectrum(
            [110.0, 300.0],
            [1.0, 0.3],
            600.0,
        ),
    ]

    similarity = FlashEntropySL(
        matching_mode=matching_mode,
        tolerance=0.01,
        remove_precursor=False,
        noise_cutoff=0.0,
        dtype=np.float64,
    )

    from_lists = similarity.matrix(
        references,
        queries,
        n_jobs=0,
        progress_bar=False,
    ).to_array()

    from_collections = similarity.matrix(
        build_collection(references),
        build_collection(queries),
        n_jobs=0,
        progress_bar=False,
    ).to_array()

    assert np.allclose(
        from_lists,
        from_collections,
        atol=1e-12,
    )


def test_entropy_sc_self_comparison_is_symmetric():
    spectra = [
        build_spectrum(
            [100.0, 200.0],
            [1.0, 0.5],
            500.0,
        ),
        build_spectrum(
            [110.0, 300.0],
            [0.3, 1.0],
            600.0,
        ),
        build_spectrum(
            [100.0, 250.0],
            [0.7, 0.9],
            550.0,
        ),
    ]

    similarity = FlashEntropySL(
        matching_mode="fragment",
        tolerance=0.01,
        remove_precursor=False,
        noise_cutoff=0.0,
        dtype=np.float64,
    )

    matrix = similarity.matrix(
        build_collection(spectra),
        n_jobs=0,
        progress_bar=False,
    ).to_array()

    assert np.allclose(
        matrix,
        matrix.T,
        atol=1e-12,
    )


def test_entropy_sc_pair_matches_matrix_element():
    reference = build_spectrum(
        [100.0, 200.0],
        [1.0, 0.5],
        500.0,
    )
    query = build_spectrum(
        [100.005, 210.0],
        [1.0, 0.5],
        510.0,
    )

    similarity = FlashEntropySL(
        matching_mode="hybrid",
        tolerance=0.01,
        remove_precursor=False,
        noise_cutoff=0.0,
        dtype=np.float64,
    )

    pair_score = float(
        similarity.pair(
            reference,
            query,
        )
    )

    matrix_score = float(
        similarity.matrix(
            [reference],
            [query],
            n_jobs=0,
            progress_bar=False,
        ).to_array()[0, 0]
    )

    assert pair_score == pytest.approx(
        matrix_score,
        abs=1e-12,
    )


def test_entropy_sc_neutral_loss_requires_precursor():
    reference = build_spectrum(
        [100.0],
        [1.0],
    )
    query = build_spectrum(
        [100.0],
        [1.0],
    )

    similarity = FlashEntropySL(
        matching_mode="neutral_loss",
        tolerance=0.01,
        remove_precursor=False,
        noise_cutoff=0.0,
        dtype=np.float64,
    )

    score = similarity.matrix(
        [reference],
        [query],
        n_jobs=0,
        progress_bar=False,
    ).to_array()[0, 0]

    assert score == 0.0


def test_entropy_sc_hybrid_combines_distinct_fragment_and_loss_matches():
    reference = build_spectrum(
        [100.0, 200.0],
        [1.0, 1.0],
        500.0,
    )
    query = build_spectrum(
        [100.0, 210.0],
        [1.0, 1.0],
        510.0,
    )

    similarity = FlashEntropySL(
        matching_mode="hybrid",
        tolerance=0.01,
        remove_precursor=False,
        noise_cutoff=0.0,
        dtype=np.float64,
    )

    score = similarity.matrix(
        [reference],
        [query],
        n_jobs=0,
        progress_bar=False,
    ).to_array()[0, 0]

    assert score == pytest.approx(
        1.0,
        abs=1e-12,
    )


def test_entropy_sc_identity_precursor_gate():
    reference = build_spectrum(
        [100.0, 200.0],
        [1.0, 1.0],
        500.0,
    )
    query = build_spectrum(
        [100.0, 200.0],
        [1.0, 1.0],
        500.3,
    )

    similarity = FlashEntropySL(
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
        n_jobs=0,
        progress_bar=False,
    ).to_array()[0, 0]

    assert score == 0.0


# -------------------------------------------------------------------------
# Cosine Flash SC
# -------------------------------------------------------------------------

@pytest.mark.parametrize(
    "matching_mode",
    [
        "fragment",
        "hybrid",
    ],
)
def test_cosine_sc_matches_existing_flash_matrix(
    matching_mode,
):
    references = [
        build_spectrum(
            [100.0, 150.0, 300.0],
            [0.6, 1.0, 0.4],
            500.0,
        ),
        build_spectrum(
            [110.0, 250.0, 400.0],
            [0.5, 0.9, 0.7],
            600.0,
        ),
    ]

    queries = [
        build_spectrum(
            [100.007, 150.002, 300.0],
            [0.6, 1.0, 0.4],
            500.0,
        ),
        build_spectrum(
            [120.0, 260.0, 410.0],
            [0.5, 0.9, 0.7],
            610.0,
        ),
    ]

    kwargs = {
        "matching_mode": matching_mode,
        "tolerance": 0.01,
        "intensity_power": 1.0,
        "remove_precursor": False,
        "noise_cutoff": 0.0,
        "normalize_to_half": True,
        "merge_within": 0.0,
        "dtype": np.float64,
    }

    old = CosineFlash(
        **kwargs,
    )
    sc = CosineFlashSL(
        **kwargs,
    )

    expected = old.matrix(
        references,
        queries,
        n_jobs=0,
        progress_bar=False,
    )
    actual = sc.matrix(
        references,
        queries,
        n_jobs=0,
        progress_bar=False,
    )

    assert np.allclose(
        actual.to_array("score"),
        expected.to_array("score"),
        atol=1e-12,
    )

    assert np.array_equal(
        actual.to_array("matches"),
        expected.to_array("matches"),
    )


@pytest.mark.parametrize(
    "intensity_power",
    [1.0, 0.5],
)
def test_cosine_sc_intensity_power_matches_existing_flash(
    intensity_power,
):
    references = [
        build_spectrum(
            [100.0, 150.0, 300.0],
            [0.6, 1.0, 0.4],
            500.0,
        ),
    ]

    queries = [
        build_spectrum(
            [100.005, 150.005, 300.0],
            [0.3, 1.0, 0.7],
            500.0,
        ),
    ]

    kwargs = {
        "matching_mode": "fragment",
        "tolerance": 0.01,
        "intensity_power": intensity_power,
        "remove_precursor": False,
        "noise_cutoff": 0.0,
        "normalize_to_half": True,
        "merge_within": 0.0,
        "dtype": np.float64,
    }

    old = CosineFlash(
        **kwargs,
    ).matrix(
        references,
        queries,
        n_jobs=0,
        progress_bar=False,
    )

    sc = CosineFlashSL(
        **kwargs,
    ).matrix(
        references,
        queries,
        n_jobs=0,
        progress_bar=False,
    )

    assert np.allclose(
        sc.to_array("score"),
        old.to_array("score"),
        atol=1e-12,
    )

    assert np.array_equal(
        sc.to_array("matches"),
        old.to_array("matches"),
    )


def test_cosine_sc_pair_matches_matrix():
    reference = build_spectrum(
        [100.0, 150.0, 300.0],
        [0.6, 1.0, 0.4],
        500.0,
    )
    query = build_spectrum(
        [100.005, 150.005, 300.0],
        [0.6, 0.9, 0.4],
        500.0,
    )

    similarity = CosineFlashSL(
        matching_mode="fragment",
        tolerance=0.01,
        remove_precursor=False,
        noise_cutoff=0.0,
        dtype=np.float64,
    )

    pair = similarity.pair(
        reference,
        query,
    )

    matrix = similarity.matrix(
        [reference],
        [query],
        n_jobs=0,
        progress_bar=False,
    )

    assert float(
        pair["score"]
    ) == pytest.approx(
        matrix.to_array("score")[0, 0],
        abs=1e-12,
    )

    assert int(
        pair["matches"]
    ) == int(
        matrix.to_array("matches")[0, 0]
    )


def test_cosine_sc_score_field_selection():
    spectra = [
        build_spectrum(
            [100.0, 200.0],
            [1.0, 0.5],
            500.0,
        ),
    ]

    similarity = CosineFlashSL(
        tolerance=0.01,
        remove_precursor=False,
        noise_cutoff=0.0,
    )

    score_only = similarity.matrix(
        spectra,
        score_fields=("score",),
        n_jobs=0,
        progress_bar=False,
    )

    matches_only = similarity.matrix(
        spectra,
        score_fields=("matches",),
        n_jobs=0,
        progress_bar=False,
    )

    assert score_only.score_fields == (
        "score",
    )
    assert matches_only.score_fields == (
        "matches",
    )


def test_cosine_sc_dtype():
    reference = build_spectrum(
        [100.0],
        [1.0],
        500.0,
    )
    query = build_spectrum(
        [100.0],
        [1.0],
        500.0,
    )

    score_32 = CosineFlashSL(
        dtype=np.float32,
        remove_precursor=False,
        noise_cutoff=0.0,
    ).pair(
        reference,
        query,
    )

    score_64 = CosineFlashSL(
        dtype=np.float64,
        remove_precursor=False,
        noise_cutoff=0.0,
    ).pair(
        reference,
        query,
    )

    assert score_32["score"].dtype == np.float32
    assert score_64["score"].dtype == np.float64


def test_optimize_matrix_orientation_puts_smaller_collection_first():
    spectra_large = [
        build_spectrum([100.0 + i], [1.0], precursor_mz=500.0)
        for i in range(5)
    ]
    spectra_small = spectra_large[:2]

    similarity = CosineFlash(
        remove_precursor=False,
        noise_cutoff=0.0,
    )

    refs, queries, is_symmetric = similarity._prepare_matrix_inputs(
        spectra_large,
        spectra_small,
    )

    refs, queries, transpose_output = similarity._optimize_matrix_orientation(
        refs,
        queries,
        is_symmetric,
    )

    assert refs.n_specs == 2
    assert queries.n_specs == 5


def test_cosine_matrix_swap_preserves_requested_orientation():
    spectra_large = [
        build_spectrum(
            [100.0 + i, 200.0 + i],
            [1.0, 0.5],
            precursor_mz=500.0 + i,
        )
        for i in range(5)
    ]
    spectra_small = spectra_large[:2]

    similarity = CosineFlash(
        tolerance=0.01,
        remove_precursor=False,
        noise_cutoff=0.0,
        dtype=np.float64,
    )

    large_small = similarity.matrix(
        spectra_large,
        spectra_small,
        n_jobs=0,
        progress_bar=False,
    )

    small_large = similarity.matrix(
        spectra_small,
        spectra_large,
        n_jobs=0,
        progress_bar=False,
    )

    assert large_small.shape == (5, 2)
    assert small_large.shape == (2, 5)

    assert np.array_equal(
        large_small.to_array("score"),
        small_large.to_array("score").T,
    )

    assert np.array_equal(
        large_small.to_array("matches"),
        small_large.to_array("matches").T,
    )


def test_entropy_matrix_swap_preserves_requested_orientation():
    spectra_large = [
        build_spectrum(
            [100.0 + i, 200.0 + i],
            [1.0, 0.5],
            precursor_mz=500.0 + i,
        )
        for i in range(5)
    ]
    spectra_small = spectra_large[:2]

    similarity = FlashEntropy(
        tolerance=0.01,
        remove_precursor=False,
        noise_cutoff=0.0,
        dtype=np.float64,
    )

    large_small = similarity.matrix(
        spectra_large,
        spectra_small,
        n_jobs=0,
        progress_bar=False,
    )

    small_large = similarity.matrix(
        spectra_small,
        spectra_large,
        n_jobs=0,
        progress_bar=False,
    )

    assert large_small.shape == (5, 2)
    assert small_large.shape == (2, 5)

    assert np.array_equal(
        large_small.to_array("score"),
        small_large.to_array("score").T,
    )