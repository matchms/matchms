from typing import Iterable
import numpy as np
import pytest
from matchms.similarity.BaseEmbeddingSimilarity import BaseEmbeddingSimilarity
from matchms.typing import SpectrumType
from tests.builder_Spectrum import SpectrumBuilder


class MockEmbeddingSimilarity(BaseEmbeddingSimilarity):
    def __init__(self, similarity: str = "cosine"):
        super().__init__(similarity=similarity)

    def compute_embeddings(self, spectra: Iterable[SpectrumType]) -> np.ndarray:
        return np.array([[0.1, 0.2, 0.3]])


@pytest.fixture
def spectra():
    builder = SpectrumBuilder()
    spectrum_1 = builder.with_mz(np.array([100, 150, 200.])) \
        .with_intensities(np.array([0.7, 0.2, 0.1])) \
        .with_metadata({'id': 'spectrum1', "precursor_mz": 210, "parent_mass": 210, "smiles": "CCC(C)C(C(=O)O)NC(=O)CCl"}) \
        .build()
    spectrum_2 = builder.with_mz(np.array([100, 140, 190.])) \
        .with_intensities(np.array([0.4, 0.2, 0.1])) \
        .with_metadata({'id': 'spectrum2', "precursor_mz": 200, "parent_mass": 200, "smiles": "CCC(C)C(C(=O)O)NC(=O)CCl"}) \
        .build()
    spectrum_3 = builder.with_mz(np.array([110, 140, 195.])) \
        .with_intensities(
        np.array([0.6, 0.2, 0.1])) \
        .with_metadata({'id': 'spectrum3', "precursor_mz": 205, "parent_mass": 205, "smiles": "C(C(=O)O)(NC(=O)O)S"}) \
        .build()
    spectrum_4 = builder.with_mz(np.array([100, 150, 200.])) \
        .with_intensities(
        np.array([0.6, 0.1, 0.6])) \
        .with_metadata({'id': 'spectrum4', "precursor_mz": 210, "parent_mass": 210, "smiles": "C(C(=O)O)(NC(=O)O)S"}) \
        .build()

    yield [spectrum_1, spectrum_2, spectrum_3, spectrum_4]


def test_compute_embeddings_not_implemented():
    with pytest.raises(NotImplementedError, match="Subclasses must implement this method."):
        base_similarity = BaseEmbeddingSimilarity()
        base_similarity.compute_embeddings([])


def test_no_input_specified_error():
    base_similarity = BaseEmbeddingSimilarity()

    with pytest.raises(ValueError, match="Either spectra or npy_path must be provided."):
        base_similarity.get_embeddings(spectra=None, npy_path=None)


def test_matrix_asymmetric_false_error(spectra):
    base_similarity = MockEmbeddingSimilarity()

    queries = spectra
    references = spectra[1:3]

    with pytest.raises(ValueError, match="Any embedding base similarity matrix is supposed to be dense and symmetric."):
        base_similarity.matrix(references, queries, is_symmetric=False)


def test_build_ann_index_missing_backend(spectra):
    similarity = MockEmbeddingSimilarity()

    with pytest.raises(ValueError, match="Only pynndescent is supported for now. Got missing."):
        similarity.build_ann_index(spectra, index_backend="missing")


def test_get_anns_incorrect_query_dim_error(spectra):
    similarity = MockEmbeddingSimilarity()

    with pytest.raises(ValueError, match="Expected 2D embeddings array, got 1D array."):
        similarity.build_ann_index(spectra)
        similarity.get_anns(query_spectra=np.array([100, 200, 300]))
