import numpy as np
from typing import List, Iterable, Union, Optional
from pathlib import Path
from abc import abstractmethod
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from matchms.similarity.BaseSimilarity import BaseSimilarity
from matchms.typing import SpectrumType

try:
    import pynndescent
except ImportError:
    pynndescent = None


class BaseEmbeddingSimilarity(BaseSimilarity):

    # TODO: docstrings
    # TODO: add pytests
    # TODO: not only string `metric`, but also a function?
    # TODO: not only PyNNDescent

    def __init__(self, similarity: str = "cosine"):  
        self.similarity = similarity
        self.index = None
        self.index_backend = None

        if self.similarity == "cosine":
            self.pairwise_similarity_fn = cosine_similarity
        elif self.similarity == "euclidean":
            self.pairwise_similarity_fn = lambda x, y: self._distances_to_similarities(euclidean_distances(x, y))
        else:
            raise ValueError(f"Only cosine and euclidean similarities are supported for now. Got {self.similarity}.")

    @abstractmethod
    def compute_embeddings(self, spectra: Iterable[SpectrumType]) -> np.ndarray:
        """Compute embeddings for a list of spectra.
        
        Args:
            spectra: List of spectra to compute embeddings for.
            
        Returns:
            np.ndarray: Embeddings for the spectra. Shape: (n_spectra, n_embedding_features).
        """

    def get_embeddings(
            self, spectra: Optional[Iterable[SpectrumType]] = None, npy_path: Optional[Union[str, Path]] = None
        ) -> np.ndarray:

        if spectra is None and npy_path is None:
            raise ValueError("Either spectra or npy_path must be provided.")

        if npy_path is not None:
            if Path(npy_path).exists():
                # If file path is provided and exists, load embeddings
                embs = self.load_embeddings(npy_path)
            else:
                # If file path is provided and does not exist, compute embeddings and store them
                embs = self.compute_embeddings(spectra)
                self.store_embeddings(npy_path, embs)
        else:
            # If no file path is provided, compute embeddings
            embs = self.compute_embeddings(spectra)
        return embs

    def pair(self, reference: SpectrumType, query: SpectrumType) -> float:
        return self.matrix([reference], [query])[0, 0]
            
    def matrix(
        self, references: List[SpectrumType], queries: List[SpectrumType], array_type: str = "numpy",
        is_symmetric: bool = True
    ) -> np.ndarray:
        if array_type != "numpy" or not is_symmetric:
            raise ValueError("Any embedding base similarity matrix is supposed to be dense and symmetric.")

        # Compute embeddings
        embs_ref = self.compute_embeddings(references)
        embs_query = self.compute_embeddings(queries)

        # Compute pairwise similarities matrix                
        return self.pairwise_similarity_fn(embs_ref, embs_query)

    def build_ann_index(
            self,
            reference_spectra: Optional[Iterable[SpectrumType]] = None,
            embeddings_path: Optional[Union[str, Path]] = None,
            k: int = 50,
            index_backend: str = "pynndescent"
        ):
        """
        Build an ANN index for the reference spectra.

        Args:
            reference_spectra: List of reference spectra to build the ANN index for.
            embeddings_path: If embeddings are already computed, provide the path to the numpy file containing them
                             instead of `reference_spectra`.
            k: Number of nearest neighbors to use for the ANN index.

        Returns:
            ANN index object.
        """

        # Compute reference embeddings
        embs_ref = self.get_embeddings(reference_spectra, embeddings_path)

        if index_backend == "pynndescent":
            if not pynndescent:
                raise ImportError("pynndescent is not installed. Please install it with `pip install pynndescent`.")
            self.index_backend = index_backend

            # Build ANN index
            index = pynndescent.NNDescent(embs_ref, metric=self.similarity, n_neighbors=k)
        else:
            raise ValueError(f"Only pynndescent is supported for now. Got {index_backend}.")

        # Keep index in memory
        self.index = index
        return self.index

    def get_anns(self, query_spectra: Iterable[SpectrumType], k: int = 50):
        if self.index is None:
            raise ValueError(
                "No index built yet. Please call `build_ann_index` on your reference spectra first."
            )

        # Compute query embeddings
        embs_query = self.compute_embeddings(query_spectra)

        # Get ANN indices
        if self.index_backend == "pynndescent":
            neighbors, distances = self.index.query(embs_query, k=k)
            similarities = self._distances_to_similarities(distances)
        else:
            raise ValueError(f"Only pynndescent is supported for now. Got {self.index_backend}.")
        return neighbors, similarities

    def get_index_anns(self):
        if self.index_backend == "pynndescent":
            neighbors, distances = self.index.neighbor_graph
            similarities = self._distances_to_similarities(distances)
            return neighbors, similarities
        else:
            raise ValueError(f"Only pynndescent is supported for now. Got {self.index_backend}.")

    def _distances_to_similarities(self, distances):
        if self.similarity == "cosine":
            return 1 - distances
        elif self.similarity == "euclidean":
            return -distances
        else:
            raise ValueError(f"Only cosine and euclidean similarities are supported for now. Got {self.similarity}.")

    @staticmethod
    def load_embeddings(npy_path: Union[str, Path]):
        """Load embeddings from a numpy file.
        
        Args:
            npy_path: Path to the numpy file.
        
        Returns:
            np.ndarray: Embeddings for the spectra. Shape: (n_spectra, n_embedding_features).
        """
        embs = np.load(npy_path)
        if embs.ndim != 2:
            raise ValueError(f"Expected 2D embeddings array, got {embs.ndim}D array.")
        return embs

    @staticmethod
    def store_embeddings(npy_path: Union[str, Path], embs: np.ndarray):
        """Store embeddings in a numpy file.
        
        Args:
            npy_path: Path to the numpy file.
            embs: Embeddings array to store.
        """
        np.save(npy_path, embs)
