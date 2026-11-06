import numpy as np
import pytest
from numpy.linalg import norm as lnorm
from matchms import Scores, SpectraCollection, Spectrum
from matchms.similarity import BinnedEmbeddingSimilarity


def _spectrum(mz, intensities):
    return Spectrum(
        mz=np.asarray(mz, dtype="float"),
        intensities=np.asarray(intensities, dtype="float"),
    )


def _expected_binned_vector(similarity, mz, intensities):
    expected = np.zeros(similarity.n_bins, dtype=np.float64)
    mz = np.asarray(mz, dtype="float")
    intensities = np.asarray(intensities, dtype="float") ** similarity.intensity_power

    valid_mask = (mz >= 0) & (mz <= similarity.max_mz)
    if not np.any(valid_mask):
        return expected

    bin_indices = np.floor(mz[valid_mask] / similarity.bin_width).astype(np.int64)
    bin_indices = np.clip(bin_indices, 0, similarity.n_bins - 1)
    np.add.at(expected, bin_indices, intensities[valid_mask])

    max_intensity = expected.max()
    if max_intensity > 0:
        expected /= max_intensity
    return expected


def _cosine_score(vec_1, vec_2):
    denominator = lnorm(vec_1) * lnorm(vec_2)
    if denominator == 0:
        return 0.0
    return np.dot(vec_1, vec_2) / denominator


@pytest.fixture
def spectra_collection():
    return SpectraCollection([
        _spectrum([100, 200, 300], [0.1, 0.2, 1.0]),
        _spectrum([110, 190, 290], [0.5, 0.2, 1.0]),
    ])


def test_binned_embedding_similarity_without_parameters():
    """Test binned embedding similarity with default parameters."""
    spectrum_1 = _spectrum([100, 200, 300, 500, 510], [0.1, 0.2, 1.0, 0.3, 0.4])
    spectrum_2 = _spectrum([100, 200, 290, 490, 510], [0.1, 0.2, 1.0, 0.3, 0.4])

    similarity = BinnedEmbeddingSimilarity()
    score = similarity.pair(spectrum_1, spectrum_2)

    expected_vec1 = _expected_binned_vector(similarity, [100, 200, 300, 500, 510], [0.1, 0.2, 1.0, 0.3, 0.4])
    expected_vec2 = _expected_binned_vector(similarity, [100, 200, 290, 490, 510], [0.1, 0.2, 1.0, 0.3, 0.4])
    expected_score = _cosine_score(expected_vec1, expected_vec2)

    assert score == pytest.approx(expected_score, rel=1e-6)


def test_binned_embedding_similarity_matrix_uses_spectra_collection(spectra_collection):
    """Test binned embedding similarity matrix computation for SpectraCollection."""
    similarity = BinnedEmbeddingSimilarity()

    scores = similarity.matrix(spectra_collection, spectra_collection, progress_bar=False)

    expected_vec1 = _expected_binned_vector(similarity, [100, 200, 300], [0.1, 0.2, 1.0])
    expected_vec2 = _expected_binned_vector(similarity, [110, 190, 290], [0.5, 0.2, 1.0])
    expected_score = _cosine_score(expected_vec1, expected_vec2)

    assert isinstance(scores, Scores)
    assert scores.shape == (2, 2)
    assert scores[0, 1] == pytest.approx(expected_score, rel=1e-6)
    assert scores[0, 0] == pytest.approx(1.0, rel=1e-6)
    assert scores[1, 1] == pytest.approx(1.0, rel=1e-6)
    assert scores[0, 1] == pytest.approx(scores[1, 0], rel=1e-6)


def test_binned_embedding_similarity_matrix_self_comparison_uses_spectra_collection(spectra_collection):
    similarity = BinnedEmbeddingSimilarity()

    scores = similarity.matrix(spectra_collection, progress_bar=False)

    assert isinstance(scores, Scores)
    assert scores.shape == (2, 2)
    assert np.allclose(scores.to_array(), scores.to_array().T)


def test_binned_embedding_similarity_parameters_with_spectra_collection():
    """Test binned embedding similarity with different parameters."""
    spectra = SpectraCollection([
        _spectrum([100, 200, 300], [0.1, 0.2, 1.0])
    ])

    similarity_1 = BinnedEmbeddingSimilarity(max_mz=500, bin_width=1)
    similarity_2 = BinnedEmbeddingSimilarity(max_mz=500, bin_width=10)

    embedding_1 = similarity_1.compute_embeddings(spectra)
    embedding_2 = similarity_2.compute_embeddings(spectra)

    assert embedding_1.shape == (1, similarity_1.n_bins)
    assert embedding_1.shape[1] == 501
    assert embedding_2.shape == (1, similarity_2.n_bins)
    assert embedding_2.shape[1] == 51

    expected_vec = np.zeros(similarity_2.n_bins)
    expected_vec[[10, 20, 30]] = [0.1, 0.2, 1.0]
    expected_vec /= np.max(expected_vec)

    assert np.allclose(embedding_2[0], expected_vec)


def test_binned_embedding_similarity_empty_and_zero_spectra_do_not_create_nans():
    spectra = SpectraCollection([
        _spectrum([], []),
        _spectrum([2000, 3000], [1.0, 2.0]),
        _spectrum([100, 200], [0.0, 0.0]),
    ])
    similarity = BinnedEmbeddingSimilarity()

    embeddings = similarity.compute_embeddings(spectra)
    scores = similarity.matrix(spectra, progress_bar=False)

    assert embeddings.shape == (3, similarity.n_bins)
    assert np.all(np.isfinite(embeddings))
    assert np.allclose(embeddings, 0.0)
    assert np.all(np.isfinite(scores.to_array()))
    assert np.allclose(scores.to_array(), 0.0)


def test_binned_embedding_similarity_empty_iterable_shape():
    similarity = BinnedEmbeddingSimilarity()

    embeddings = similarity.compute_embeddings([])

    assert embeddings.shape == (0, similarity.n_bins)


def test_binned_embedding_similarity_euclidean():
    """Test binned embedding similarity with euclidean distance."""
    spectrum_1 = _spectrum([100, 200, 300], [0.1, 0.2, 1.0])
    spectrum_2 = _spectrum([110, 210, 310], [0.1, 0.2, 1.0])

    similarity = BinnedEmbeddingSimilarity(similarity="euclidean")
    score = similarity.pair(spectrum_1, spectrum_2)

    assert score < 0


def test_binned_embedding_similarity_invalid_similarity():
    """Test binned embedding similarity with invalid similarity measure."""
    with pytest.raises(ValueError):
        BinnedEmbeddingSimilarity(similarity="invalid")


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"max_mz": 0}, "max_mz must be > 0"),
        ({"bin_width": 0}, "bin_width must be > 0"),
    ],
)
def test_binned_embedding_similarity_invalid_parameters(kwargs, message):
    with pytest.raises(ValueError, match=message):
        BinnedEmbeddingSimilarity(**kwargs)


def test_binned_embedding_similarity_ann():
    """Test approximate nearest neighbor search functionality."""
    pytest.importorskip("pynndescent")

    reference_spectra = SpectraCollection([
        _spectrum(
            [100 + i, 200 + i, 300 + i],
            [0.1 + i * 0.01, 0.2 + i * 0.01, 1.0 - i * 0.01],
        )
        for i in range(10)
    ])
    query_spectra = SpectraCollection([
        _spectrum([100, 200, 300], [0.11, 0.21, 0.99])
    ])

    similarity = BinnedEmbeddingSimilarity()
    similarity.build_ann_index(reference_spectra, k=5)

    neighbors, similarities = similarity.get_anns(query_spectra, k=3)
    assert neighbors.shape == (1, 3)
    assert similarities.shape == (1, 3)
    assert neighbors[0, 0] == 0

    neighbors, similarities = similarity.get_index_anns()
    assert neighbors.shape == (10, 5)
    assert similarities.shape == (10, 5)

    exact_neighbors, exact_similarities = similarity.get_anns(query_spectra, k=1)
    assert exact_neighbors.shape == (1, 1)
    assert exact_similarities.shape == (1, 1)
    assert exact_neighbors[0, 0] == 0


def test_binned_embedding_similarity_ann_save_load(tmp_path):
    """Test saving and loading ANN index."""
    pytest.importorskip("pynndescent")

    reference_spectra = SpectraCollection([
        _spectrum(
            [100 + i, 200 + i, 300 + i],
            [0.1 + i * 0.01, 0.2 + i * 0.01, 1.0 - i * 0.01],
        )
        for i in range(10)
    ])

    similarity = BinnedEmbeddingSimilarity()
    similarity.build_ann_index(reference_spectra, k=5)

    index_path = tmp_path / "test_index.pkl"
    similarity.save_ann_index(index_path)

    similarity_2 = BinnedEmbeddingSimilarity()
    similarity_2.load_ann_index(index_path)

    query_spectra = SpectraCollection([reference_spectra[0]])
    neighbors, similarities = similarity_2.get_anns(query_spectra, k=1)
    assert neighbors.shape == (1, 1)
    assert similarities.shape == (1, 1)
    assert neighbors[0, 0] == 0


def test_binned_embedding_similarity_ann_errors():
    """Test error handling in ANN functionality."""
    pytest.importorskip("pynndescent")

    reference_spectra = SpectraCollection([
        _spectrum(
            [100 + i, 200 + i, 300 + i],
            [0.1 + i * 0.01, 0.2 + i * 0.01, 1.0 - i * 0.01],
        )
        for i in range(10)
    ])

    similarity = BinnedEmbeddingSimilarity()

    with pytest.raises(ValueError, match="No index built yet"):
        similarity.get_anns(SpectraCollection([reference_spectra[0]]))

    similarity.build_ann_index(reference_spectra, k=1)

    with pytest.raises(ValueError, match="k .* is larger than"):
        similarity.get_anns(SpectraCollection([reference_spectra[0]]), k=2)
