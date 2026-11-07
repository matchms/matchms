import numpy as np
import pytest
from matchms import SpectraCollection
from matchms.scores import Scores
from matchms.similarity.entropy_search import EntropySearch
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


# -------------------------------------------------------------------------
# EntropySearch: top-N search
# -------------------------------------------------------------------------


def _random_separated_spectrum(rng, n_peaks, precursor_mz=1000.0):
    """Spectrum whose peaks are at least 0.5 Da apart, as EntropySearch requires."""
    mz = rng.choice(np.arange(50, 900), size=n_peaks, replace=False) + rng.uniform(0.0, 0.4, n_peaks)
    intensities = rng.uniform(0.05, 1.0, n_peaks)
    return _build_spectrum(np.sort(mz), intensities, precursor_mz=precursor_mz)


def _top_n_example(seed=0, n_library=60, n_queries=12):
    rng = np.random.default_rng(seed)
    library = [_random_separated_spectrum(rng, int(rng.integers(5, 25))) for _ in range(n_library)]
    queries = []
    for i in rng.choice(n_library, size=n_queries, replace=False):
        base = library[i]
        mz = np.sort(base.peaks.mz + rng.normal(0.0, 0.003, base.peaks.mz.size))
        scale = rng.uniform(0.5, 1.5, mz.size)
        queries.append(_build_spectrum(mz, base.peaks.intensities * scale, precursor_mz=1000.0))
    return library, queries


def _expected_top_n(dense_row, top_n):
    """Columns of the top_n positive scores: descending score, ties by lower column."""
    order = np.lexsort((np.arange(dense_row.size), -dense_row))
    return [int(c) for c in order[:top_n] if dense_row[c] > 0]


@pytest.mark.parametrize("top_n", [1, 3, 7])
@pytest.mark.parametrize("n_jobs", [1, 2])
def test_entropy_search_top_n_matches_dense_search(top_n, n_jobs):
    library, queries = _top_n_example()
    similarity = EntropySearch(tolerance=0.01)
    index = similarity.build_index(library)

    dense = similarity.search(queries, index, progress_bar=False, n_jobs=n_jobs).to_array()
    hits = similarity.search(queries, index, progress_bar=False, n_jobs=n_jobs, top_n=top_n)

    assert isinstance(hits, Scores)
    assert hits.is_sparse
    assert hits.shape == dense.shape
    assert hits.score_fields == ("score",)

    coo = hits.to_coo()
    for row in range(dense.shape[0]):
        expected = _expected_top_n(dense[row], top_n)
        got = coo.col[coo.row == row]
        assert list(got) == expected  # same hits, stored in rank order
        assert np.array_equal(coo.data[coo.row == row], dense[row, expected])
        assert len(expected) <= top_n


def test_entropy_search_top_n_larger_than_library_keeps_all_positive_scores():
    library, queries = _top_n_example(seed=1)
    similarity = EntropySearch(tolerance=0.01)
    index = similarity.build_index(library)

    dense = similarity.search(queries, index, progress_bar=False).to_array()
    hits = similarity.search(queries, index, progress_bar=False, top_n=10 * len(library))

    assert hits.is_sparse
    assert hits.to_coo().nnz == int((dense > 0).sum())
    np.testing.assert_array_equal(hits.to_array(), dense)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_entropy_search_top_n_is_independent_of_block_size_and_threads(monkeypatch, dtype):
    from matchms.similarity import entropy_search

    library, queries = _top_n_example(seed=2, n_library=40, n_queries=9)
    similarity = EntropySearch(tolerance=0.01, dtype=dtype)
    index = similarity.build_index(library)

    reference = similarity.search(queries, index, progress_bar=False, top_n=4).to_coo()
    assert reference.data.dtype == dtype

    # One query per block, scored by several worker threads.
    monkeypatch.setattr(entropy_search, "_TOP_N_BLOCK_BYTES", 1)
    blocked = similarity.search(queries, index, progress_bar=False, n_jobs=3, top_n=4).to_coo()

    np.testing.assert_array_equal(blocked.row, reference.row)
    np.testing.assert_array_equal(blocked.col, reference.col)
    np.testing.assert_array_equal(blocked.data, reference.data)


def test_entropy_search_top_n_ties_prefer_lower_library_index():
    rng = np.random.default_rng(3)
    a = _random_separated_spectrum(rng, 10)
    b = _random_separated_spectrum(rng, 10)
    c = _random_separated_spectrum(rng, 10)
    similarity = EntropySearch(tolerance=0.01)
    index = similarity.build_index([a, b, a, a, c])

    top1 = similarity.search([a], index, progress_bar=False, top_n=1).to_coo()
    top2 = similarity.search([a], index, progress_bar=False, top_n=2).to_coo()

    assert list(top1.col) == [0]
    assert list(top2.col) == [0, 2]
    assert top2.data[0] == top2.data[1] == pytest.approx(1.0, abs=1e-6)


def test_entropy_search_top_n_empty_inputs():
    library, _ = _top_n_example(seed=4, n_library=5, n_queries=2)
    similarity = EntropySearch(tolerance=0.01)
    index = similarity.build_index(library)

    no_queries = similarity.search([], index, progress_bar=False, top_n=3)
    assert no_queries.is_sparse
    assert no_queries.shape == (0, 5)
    assert no_queries.to_coo().nnz == 0

    no_peaks = similarity.search([_build_spectrum([], [])], index, progress_bar=False, top_n=3)
    assert no_peaks.shape == (1, 5)
    assert no_peaks.to_coo().nnz == 0


@pytest.mark.parametrize("top_n", [0, -1, 2.5, True])
def test_entropy_search_top_n_rejects_invalid_values(top_n):
    library, queries = _top_n_example(seed=5, n_library=5, n_queries=2)
    similarity = EntropySearch(tolerance=0.01)
    index = similarity.build_index(library)

    with pytest.raises(ValueError, match="top_n"):
        similarity.search(queries, index, progress_bar=False, top_n=top_n)


def test_entropy_search_top_n_sparse_scores_support_filtering():
    library, queries = _top_n_example(seed=6)
    similarity = EntropySearch(tolerance=0.01)
    index = similarity.build_index(library)

    hits = similarity.search(queries, index, progress_bar=False, top_n=5)
    strong = hits[hits > 0.5]

    assert strong.is_sparse
    assert strong.shape == hits.shape
    assert np.all(strong.to_coo().data > 0.5)
