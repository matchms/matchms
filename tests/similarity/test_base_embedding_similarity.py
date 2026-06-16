from collections.abc import Iterable
import numpy as np
import pytest
from matchms import Scores, SpectraCollection
from matchms.similarity.BaseEmbeddingSimilarity import BaseEmbeddingSimilarity
from matchms.typing import SpectrumType
from tests.builder_Spectrum import SpectrumBuilder


class MockEmbeddingSimilarity(BaseEmbeddingSimilarity):
    def __init__(self, similarity: str = "cosine"):
        super().__init__(similarity=similarity)

    def compute_embeddings(self, spectra: Iterable[SpectrumType]) -> np.ndarray:
        spectra = list(spectra)
        if len(spectra) == 0:
            return np.zeros((0, 3), dtype=np.float64)
        base_embedding = np.array([0.1, 0.2, 0.3], dtype=np.float64)
        return np.tile(base_embedding, (len(spectra), 1))


@pytest.fixture
def spectra_collection():
    a = "CCC(C)C(C(=O)O)NC(=O)CCl"
    b = "C(C(=O)O)(NC(=O)O)S"

    builder = SpectrumBuilder()
    spectrum_1 = (
        builder.with_mz(np.array([100, 150, 200.]))
        .with_intensities(np.array([0.7, 0.2, 0.1]))
        .with_metadata({"id": "spectrum1", "precursor_mz": 210, "parent_mass": 210, "smiles": a})
        .build()
    )
    spectrum_2 = (
        builder.with_mz(np.array([100, 140, 190.]))
        .with_intensities(np.array([0.4, 0.2, 0.1]))
        .with_metadata({"id": "spectrum2", "precursor_mz": 200, "parent_mass": 200, "smiles": a})
        .build()
    )
    spectrum_3 = (
        builder.with_mz(np.array([110, 140, 195.]))
        .with_intensities(np.array([0.6, 0.2, 0.1]))
        .with_metadata({"id": "spectrum3", "precursor_mz": 205, "parent_mass": 205, "smiles": b})
        .build()
    )
    spectrum_4 = (
        builder.with_mz(np.array([100, 150, 200.]))
        .with_intensities(np.array([0.6, 0.1, 0.6]))
        .with_metadata({"id": "spectrum4", "precursor_mz": 210, "parent_mass": 210, "smiles": b})
        .build()
    )

    return SpectraCollection([spectrum_1, spectrum_2, spectrum_3, spectrum_4])


def test_compute_embeddings_not_implemented():
    class DummyEmbeddingSimilarity(BaseEmbeddingSimilarity):
        def compute_embeddings(self, spectra):
            return super().compute_embeddings(spectra)

    base_similarity = DummyEmbeddingSimilarity()
    with pytest.raises(NotImplementedError, match="Subclasses must implement this method."):
        base_similarity.compute_embeddings([])


def test_no_input_specified_error():
    base_similarity = MockEmbeddingSimilarity()

    with pytest.raises(ValueError, match="Either spectra or npy_path must be provided."):
        base_similarity.get_embeddings(spectra=None, npy_path=None)


def test_get_embeddings_accepts_spectra_collection(spectra_collection):
    similarity = MockEmbeddingSimilarity()

    embeddings = similarity.get_embeddings(spectra_collection)

    assert embeddings.shape == (len(spectra_collection), 3)
    assert np.allclose(embeddings[0], [0.1, 0.2, 0.3])


def test_matrix_returns_scores_for_spectra_collection(spectra_collection):
    similarity = MockEmbeddingSimilarity()

    scores = similarity.matrix(spectra_collection, spectra_collection, progress_bar=False)

    assert isinstance(scores, Scores)
    assert scores.shape == (len(spectra_collection), len(spectra_collection))
    assert scores.score_fields == ("score",)
    assert np.allclose(scores.to_array(), 1.0)


def test_matrix_self_comparison_uses_spectra_collection(spectra_collection):
    similarity = MockEmbeddingSimilarity()

    scores = similarity.matrix(spectra_collection, progress_bar=False)

    assert isinstance(scores, Scores)
    assert scores.shape == (len(spectra_collection), len(spectra_collection))
    assert np.allclose(scores.to_array(), scores.to_array().T)


def test_pair_uses_standard_matrix_contract(spectra_collection):
    similarity = MockEmbeddingSimilarity()

    score = similarity.pair(spectra_collection[0], spectra_collection[1])

    assert score == pytest.approx(1.0)


def test_compute_similarity_matrix_from_embeddings_returns_raw_array():
    similarity = MockEmbeddingSimilarity()
    embeddings = np.array([[1.0, 0.0], [0.0, 1.0]])

    matrix = similarity.compute_similarity_matrix_from_embeddings(embeddings)

    assert isinstance(matrix, np.ndarray)
    assert matrix.shape == (2, 2)
    assert np.allclose(matrix, np.eye(2))


def test_build_ann_index_missing_backend(spectra_collection):
    similarity = MockEmbeddingSimilarity()

    with pytest.raises(ValueError, match="Only pynndescent is supported for now. Got missing."):
        similarity.build_ann_index(spectra_collection, index_backend="missing")


def test_get_anns_incorrect_query_dim_error(spectra_collection):
    pytest.importorskip("pynndescent")
    similarity = MockEmbeddingSimilarity()

    similarity.build_ann_index(spectra_collection, k=2)
    with pytest.raises(ValueError, match="Expected 2D embeddings array, got 1D array."):
        similarity.get_anns(query_spectra=np.array([100, 200, 300]), k=1)
