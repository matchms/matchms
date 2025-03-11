import numpy as np
from typing import List, Iterable
from abc import abstractmethod
from sklearn.metrics.pairwise import cosine_similarity
from matchms.similarity.BaseSimilarity import BaseSimilarity
from matchms.typing import SpectrumType

try:
    import pynndescent
except ImportError:
    pynndescent = None


class BaseEmbeddingSimilarity(BaseSimilarity):

    # TODO: add pytests
    # TODO: add pynndescent to pyproject.toml, as optional dependency?
    # TODO: compute_embeddings save/load from disk?
    # TODO: not only string `similarity`, but also a function?
    # TODO: at least cos and eucl similarity
    # TODO: not only PyNNDescent

    def __init__(self, similarity: str = "cosine"):  
        self.similarity = similarity
        self.index = None

        if self.similarity == "cosine":
            self.pairwise_similarity_fn = cosine_similarity
        else:
            raise ValueError(f"Only cosine and euclidean similarity are supported for now. Got {self.similarity}.")

    @abstractmethod
    def compute_embeddings(self, spectra: Iterable[SpectrumType]) -> np.ndarray:
        """Compute embeddings for a list of spectra.
        
        Args:
            spectra: List of spectra to compute embeddings for.
            
        Returns:
            np.ndarray: Embeddings for the spectra. Shape: (n_spectra, n_embedding_features).
        """

    def pair(self, reference: SpectrumType, query: SpectrumType) -> float:
        return self.matrix([reference], [query])[0, 0]
            
    def matrix(self, references: List[SpectrumType], queries: List[SpectrumType]) -> np.ndarray:

        # Compute embeddings
        embs_ref = self.compute_embeddings(references)
        embs_query = self.compute_embeddings(queries)

        # Compute pairwise similarity matrix                
        return self.pairwise_similarity_fn(embs_ref, embs_query)

    def generate_ann_index(self, reference_spectra: Iterable[SpectrumType], k: int = 50):

        if not pynndescent:
            raise ImportError("pynndescent is not installed. Please install it with `pip install pynndescent`.")

        # Compute reference embeddings
        embs_ref = self.compute_embeddings(reference_spectra)

        # Build ANN index
        index = pynndescent.NNDescent(embs_ref, metric=self.similarity, n_neighbors=k)
        self.index = index

        return index

    def get_anns(self, query_spectra: Iterable[SpectrumType], k: int = 50):

        if self.index is None:
            raise ValueError(
                "No index generated yet. Please call `generate_ann_index` on your reference spectra first."
            )

        # Compute query embeddings
        embs_query = self.compute_embeddings(query_spectra)

        # Get ANN indices
        return self.index.query(embs_query, k=k)

    def get_index_anns(self):
        return self.index.neighbor_graph
