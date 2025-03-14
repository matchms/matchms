import numpy as np
import pytest
from matchms import Spectrum
from matchms.similarity import BinnedEmbeddingSimilarity


def test_binned_embedding_similarity_without_parameters():
    """Test binned embedding similarity with default parameters."""
    spectrum_1 = Spectrum(mz=np.array([100, 200, 300, 500, 510], dtype="float"),
                         intensities=np.array([0.1, 0.2, 1.0, 0.3, 0.4], dtype="float"))

    spectrum_2 = Spectrum(mz=np.array([100, 200, 290, 490, 510], dtype="float"),
                         intensities=np.array([0.1, 0.2, 1.0, 0.3, 0.4], dtype="float"))

    similarity = BinnedEmbeddingSimilarity()
    score = similarity.pair(spectrum_1, spectrum_2)

    # Calculate expected binned vectors
    expected_vec1 = np.zeros(1005)
    expected_vec2 = np.zeros(1005)
    
    # Fill bins for spectrum 1
    expected_vec1[[100, 200, 300, 500, 510]] = [0.1, 0.2, 1.0, 0.3, 0.4]
    expected_vec1 /= np.max(expected_vec1)
    
    # Fill bins for spectrum 2 
    expected_vec2[[100, 200, 290, 490, 510]] = [0.1, 0.2, 1.0, 0.3, 0.4]
    expected_vec2 /= np.max(expected_vec2)

    # Calculate expected cosine similarity
    expected_score = np.dot(expected_vec1, expected_vec2) / (np.linalg.norm(expected_vec1) * np.linalg.norm(expected_vec2))

    assert score == pytest.approx(expected_score, rel=1e-6), "Expected different cosine score"


def test_binned_embedding_similarity_matrix():
    """Test binned embedding similarity matrix computation."""
    spectrum_1 = Spectrum(mz=np.array([100, 200, 300], dtype="float"),
                         intensities=np.array([0.1, 0.2, 1.0], dtype="float"))
    spectrum_2 = Spectrum(mz=np.array([110, 190, 290], dtype="float"),
                         intensities=np.array([0.5, 0.2, 1.0], dtype="float"))

    spectra = [spectrum_1, spectrum_2]
    similarity = BinnedEmbeddingSimilarity()
    scores = similarity.matrix(spectra, spectra)

    # Calculate expected binned vectors
    expected_vec1 = np.zeros(1005)
    expected_vec2 = np.zeros(1005)
    
    expected_vec1[[100, 200, 300]] = [0.1, 0.2, 1.0]
    expected_vec1 /= np.max(expected_vec1)
    
    expected_vec2[[110, 190, 290]] = [0.5, 0.2, 1.0]
    expected_vec2 /= np.max(expected_vec2)

    # Calculate expected cosine similarity
    expected_score = np.dot(expected_vec1, expected_vec2) / (np.linalg.norm(expected_vec1) * np.linalg.norm(expected_vec2))

    assert scores[0, 1] == pytest.approx(expected_score, rel=1e-6), "Expected different cosine score"
    assert np.allclose(scores[0, 0], scores[1, 1], atol=1e-6), "Expected perfect self-similarity"
    assert np.allclose(scores[0, 1], scores[1, 0], atol=1e-6), "Expected symmetric matrix"


def test_binned_embedding_similarity_parameters():
    """Test binned embedding similarity with different parameters."""
    spectrum = Spectrum(mz=np.array([100, 200, 300], dtype="float"),
                       intensities=np.array([0.1, 0.2, 1.0], dtype="float"))

    similarity_1 = BinnedEmbeddingSimilarity(max_mz=500, bin_width=1)
    similarity_2 = BinnedEmbeddingSimilarity(max_mz=500, bin_width=10)

    embedding_1 = similarity_1.compute_embeddings([spectrum])
    embedding_2 = similarity_2.compute_embeddings([spectrum])

    assert embedding_1.shape[1] == 500, "Expected 500 bins"
    assert embedding_2.shape[1] == 50, "Expected 50 bins"

    # Check if binning is correct for bin_width=10
    expected_vec = np.zeros(50)
    expected_vec[[10, 20, 30]] = [0.1, 0.2, 1.0]  # 100/10=10, 200/10=20, 300/10=30
    expected_vec /= np.max(expected_vec)

    assert np.allclose(embedding_2[0], expected_vec), "Expected different binned values"


def test_binned_embedding_similarity_euclidean():
    """Test binned embedding similarity with euclidean distance."""
    spectrum_1 = Spectrum(mz=np.array([100, 200, 300], dtype="float"),
                         intensities=np.array([0.1, 0.2, 1.0], dtype="float"))
    spectrum_2 = Spectrum(mz=np.array([110, 210, 310], dtype="float"),
                         intensities=np.array([0.1, 0.2, 1.0], dtype="float"))

    similarity = BinnedEmbeddingSimilarity(similarity="euclidean")
    score = similarity.pair(spectrum_1, spectrum_2)

    # For shifted peaks, expect non-zero distance
    assert score < 0, "Expected negative score for euclidean similarity"


def test_binned_embedding_similarity_invalid_similarity():
    """Test binned embedding similarity with invalid similarity measure."""
    with pytest.raises(ValueError):
        BinnedEmbeddingSimilarity(similarity="invalid")


def test_binned_embedding_similarity_ann():
    """Test approximate nearest neighbor search functionality."""
    # Create 10 test spectra with slightly varying peaks
    reference_spectra = []
    for i in range(10):
        mz = np.array([100 + i, 200 + i, 300 + i], dtype="float")
        intensities = np.array([0.1 + i*0.01, 0.2 + i*0.01, 1.0 - i*0.01], dtype="float") 
        spectrum = Spectrum(mz=mz, intensities=intensities)
        reference_spectra.append(spectrum)

    # Create query spectrum similar to first reference spectrum
    query_spectrum = Spectrum(
        mz=np.array([100, 200, 300], dtype="float"),
        intensities=np.array([0.11, 0.21, 0.99], dtype="float")
    )

    similarity = BinnedEmbeddingSimilarity()
    
    # Build index
    similarity.build_ann_index(reference_spectra, k=5)
    
    # Test get_anns
    neighbors, similarities = similarity.get_anns([query_spectrum], k=3)
    assert neighbors.shape == (1, 3), "Expected shape (1,3) for neighbors"
    assert similarities.shape == (1, 3), "Expected shape (1,3) for similarities"
    
    # Query spectrum should be most similar to first reference spectrum
    assert neighbors[0,0] == 0, "Expected first spectrum to be closest neighbor"
    
    # Test get_index_anns
    neighbors, similarities = similarity.get_index_anns()
    assert neighbors.shape == (10, 5), "Expected shape (10,5) for index neighbors"
    assert similarities.shape == (10, 5), "Expected shape (10,5) for index similarities"

    # Test against exact nearest neighbor
    exact_neighbors, exact_similarities = similarity.get_anns([query_spectrum], k=1)
    assert exact_neighbors.shape == (1, 1), "Expected shape (1,1) for exact neighbors"
    assert exact_similarities.shape == (1, 1), "Expected shape (1,1) for exact similarities"
    assert exact_neighbors[0,0] == 0, "Expected first spectrum to be exact nearest neighbor"


def test_binned_embedding_similarity_ann_save_load(tmp_path):
    """Test saving and loading ANN index."""
    # Create 10 test spectra
    reference_spectra = []
    for i in range(10):
        mz = np.array([100 + i, 200 + i, 300 + i], dtype="float")
        intensities = np.array([0.1 + i*0.01, 0.2 + i*0.01, 1.0 - i*0.01], dtype="float")
        spectrum = Spectrum(mz=mz, intensities=intensities)
        reference_spectra.append(spectrum)
    
    similarity = BinnedEmbeddingSimilarity()
    similarity.build_ann_index(reference_spectra, k=5)
    
    # Save index
    index_path = tmp_path / "test_index.pkl"
    similarity.save_ann_index(index_path)
    
    # Load index in new instance
    similarity_2 = BinnedEmbeddingSimilarity()
    similarity_2.load_ann_index(index_path)
    
    # Test loaded index with first spectrum
    neighbors, similarities = similarity_2.get_anns([reference_spectra[0]], k=1)
    assert neighbors.shape == (1, 1), "Expected shape (1,1) for neighbors"
    assert similarities.shape == (1, 1), "Expected shape (1,1) for similarities"
    assert neighbors[0,0] == 0, "Expected self as nearest neighbor"


def test_binned_embedding_similarity_ann_errors():
    """Test error handling in ANN functionality."""
    # Create 10 test spectra
    reference_spectra = []
    for i in range(10):
        mz = np.array([100 + i, 200 + i, 300 + i], dtype="float")
        intensities = np.array([0.1 + i*0.01, 0.2 + i*0.01, 1.0 - i*0.01], dtype="float")
        spectrum = Spectrum(mz=mz, intensities=intensities)
        reference_spectra.append(spectrum)
    
    similarity = BinnedEmbeddingSimilarity()
    
    # Test error when getting ANNs without building index
    with pytest.raises(ValueError, match="No index built yet"):
        similarity.get_anns([reference_spectra[0]])
    
    # Build index
    similarity.build_ann_index(reference_spectra, k=1)
    
    # Test error when k is larger than index k
    with pytest.raises(ValueError, match="k .* is larger than"):
        similarity.get_anns([reference_spectra[0]], k=2)
