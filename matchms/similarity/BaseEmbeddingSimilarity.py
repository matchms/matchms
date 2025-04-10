import pickle
from abc import abstractmethod
from pathlib import Path
from typing import Any, Iterable, List, Optional, Tuple, Union
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from matchms.similarity.BaseSimilarity import BaseSimilarity
from matchms.typing import SpectrumType


try:
    import pynndescent
except ImportError:
    pynndescent = None


class BaseEmbeddingSimilarity(BaseSimilarity):
    """Base class for similarity measures that work with embeddings.

    This class provides functionality for computing similarities between spectra based on their
    embeddings (vector representations). It supports cosine and euclidean similarity metrics,
    and includes approximate nearest neighbor (ANN) search capabilities.

    Parameters
    ----------
    similarity : str
        The similarity measure to use for comparing embeddings. Default is "cosine".
        Options are "cosine" or "euclidean".

    Attributes
    ----------
    index : object
        The ANN index object; if built.
    index_backend : str
        The backend used for ANN indexing (currently only "pynndescent" supported); if index is built.
    index_kwargs : dict
        Additional arguments passed to the ANN index constructor; if index is built.
    index_k : int
        Number of nearest neighbors used in the ANN index; if index is built.
    """

    def __init__(self, similarity: str = "cosine"):
        self.similarity = similarity
        self.index = None
        self.index_backend = None
        self.index_kwargs = None
        self.index_k = None

        if self.similarity == "cosine":
            self.pairwise_similarity_fn = cosine_similarity
        elif self.similarity == "euclidean":
            self.pairwise_similarity_fn = lambda x, y: self._distances_to_similarities(euclidean_distances(x, y))
        else:
            raise ValueError(f"Only cosine and euclidean similarities are supported for now. Got {self.similarity}.")

    @abstractmethod
    def compute_embeddings(self, spectra: Iterable[SpectrumType]) -> np.ndarray:
        """Compute embeddings for a list of spectra.

        Parameters
        ----------
        spectra:
            List of spectra to compute embeddings for.

        Returns
        -------
        np.ndarray
            Embeddings for the spectra. Shape: (n_spectra, n_embedding_features).
        """
        raise NotImplementedError("Subclasses must implement this method.")

    def get_embeddings(
            self,
            spectra: Optional[Iterable[SpectrumType]] = None,
            npy_path: Optional[Union[str, Path]] = None) -> np.ndarray:
        """Get embeddings either by computing them or loading from disk.

        Parameters
        ----------
        spectra:
            List of spectra to compute embeddings for.
        npy_path:
            Path to load/save embeddings from/to. If provided, embeddings are loaded from disk if it exists,
            otherwise they are computed and saved on disk to the provided path.

        Returns
        -------
        np.ndarray
            Embeddings array.

        Raises
        ------
        ValueError
            If neither spectra nor npy_path is provided.
        """
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
        """Compute similarity between a pair of spectra.

        Parameters
        ----------
        reference : SpectrumType
            Reference spectrum.
        query : SpectrumType
            Query spectrum.

        Returns
        -------
        float
            Similarity score between the spectra.
        """
        return self.matrix([reference], [query])[0, 0]

    def matrix(
            self,
            references: List[SpectrumType],
            queries: List[SpectrumType],
            array_type: str = "numpy",
            is_symmetric: bool = True) -> np.ndarray:
        """Compute similarity matrix between reference and query spectra.

        Parameters
        ----------
        references:
            List of reference spectra.
        queries:
            List of query spectra.
        array_type:
            Type of array to return. Must be "numpy".
        is_symmetric:
            Whether the matrix is symmetric. Must be True.

        Returns
        -------
        np.ndarray
            Similarity matrix.

        Raises
        ------
        ValueError
            If array_type is not "numpy" or is_symmetric is False.
        """
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
            k: int = 100,
            index_backend: str = "pynndescent",
            **index_kwargs) -> Any:
        """Build an ANN index for the reference spectra.

        Parameters
        ----------
        reference_spectra : Optional[Iterable[SpectrumType]]
            List of reference spectra to build the ANN index for.
        embeddings_path : Optional[Union[str, Path]]
            If embeddings are already computed, provide the path to the numpy file.
        k : int, optional
            Number of nearest neighbors to use for the ANN index.
        index_backend : str, optional
            Backend to use for ANN index. Currently only "pynndescent" is supported.
        **index_kwargs
            Additional keyword arguments passed to the index constructor.

        Returns
        -------
        Any
            The constructed ANN index.

        Raises
        ------
        ImportError
            If pynndescent is not installed.
        ValueError
            If an unsupported index_backend is specified.
        """
        # Compute reference embeddings
        embs_ref = self.get_embeddings(reference_spectra, embeddings_path)

        if index_backend == "pynndescent":
            if not pynndescent:
                raise ImportError("pynndescent is not installed. Please install it with `pip install pynndescent`.")
            self.index_backend = index_backend
            self.index_k = k
            self.index_kwargs = index_kwargs

            # Build ANN index
            index = pynndescent.NNDescent(embs_ref, metric=self.similarity, n_neighbors=k, **index_kwargs)
        else:
            raise ValueError(f"Only pynndescent is supported for now. Got {index_backend}.")

        # Keep index in memory
        self.index = index
        return self.index

    def get_anns(
            self,
            query_spectra: Union[Iterable[SpectrumType], np.ndarray],
            k: int = 100) -> Tuple[np.ndarray, np.ndarray]:
        """Get approximate nearest neighbors for query spectra.

        Parameters
        ----------
        query_spectra : Union[Iterable[SpectrumType], np.ndarray]
            Query spectra or their embeddings.
        k : int, optional
            Number of nearest neighbors to return.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            Neighbor indices and similarity scores.

        Raises
        ------
        ValueError
            If no index is built or k is larger than index k.
        """
        if self.index is None:
            raise ValueError(
                "No index built yet. Please call `build_ann_index` on your reference spectra or `load_ann_index` if it "
                "was previously built and stored using `save_ann_index`."
            )

        if k > self.index_k:
            raise ValueError(f"k ({k}) is larger than the k used to build the index ({self.index_k}).")

        if isinstance(query_spectra, np.ndarray):
            embs_query = query_spectra
            if embs_query.ndim != 2:
                raise ValueError(f"Expected 2D embeddings array, got {embs_query.ndim}D array.")
        else:
            # Compute query embeddings
            embs_query = self.compute_embeddings(query_spectra)

        # Get ANN indices
        if self.index_backend == "pynndescent":
            neighbors, distances = self.index.query(embs_query, k=k)
            similarities = self._distances_to_similarities(distances)
        else:
            raise ValueError(f"Only pynndescent is supported for now. Got {self.index_backend}.")
        return neighbors, similarities

    def get_index_anns(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get nearest neighbors for all points in the index.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            Neighbor indices and similarity scores.

        Raises
        ------
        ValueError
            If unsupported index backend is used.
        """
        if self.index_backend == "pynndescent":
            neighbors, distances = self.index.neighbor_graph
            similarities = self._distances_to_similarities(distances)
            return neighbors, similarities
        raise ValueError(f"Only pynndescent is supported for now. Got {self.index_backend}.")

    def _distances_to_similarities(self, distances: np.ndarray) -> np.ndarray:
        """Convert distances to similarities based on similarity metric.

        Parameters
        ----------
        distances : np.ndarray
            Distance matrix.

        Returns
        -------
        np.ndarray
            Similarity matrix.

        Raises
        ------
        ValueError
            If unsupported similarity metric is used.
        """
        if self.similarity == "cosine":
            return 1 - distances
        if self.similarity == "euclidean":
            return -distances
        raise ValueError(f"Only cosine and euclidean similarities are supported for now. Got {self.similarity}.")

    @staticmethod
    def load_embeddings(npy_path: Union[str, Path]) -> np.ndarray:
        """Load embeddings from a numpy file.

        Parameters
        ----------
        npy_path : Union[str, Path]
            Path to the numpy file.

        Returns
        -------
        np.ndarray
            Embeddings array.

        Raises
        ------
        ValueError
            If loaded array is not 2D.
        """
        embs = np.load(npy_path)
        if embs.ndim != 2:
            raise ValueError(f"Expected 2D embeddings array, got {embs.ndim}D array.")
        return embs

    @staticmethod
    def store_embeddings(npy_path: Union[str, Path], embs: np.ndarray) -> None:
        """Store embeddings in a numpy file.

        Parameters
        ----------
        npy_path : Union[str, Path]
            Path to save the embeddings to.
        embs : np.ndarray
            Embeddings array to store.
        """
        np.save(npy_path, embs)

    def save_ann_index(self, path: Union[str, Path]) -> None:
        """Save the ANN index to disk.

        Parameters
        ----------
        path : Union[str, Path]
            Path to save the index to.

        Raises
        ------
        ValueError
            If no index exists to save.
        """
        if self.index is None:
            raise ValueError("No index to save. Please build an index first using build_ann_index().")

        save_dict = {
            'index': self.index,
            'backend': self.index_backend,
            'similarity': self.similarity,
            'index_kwargs': self.index_kwargs,
            'index_k': self.index_k
        }

        with open(path, 'wb') as f:
            pickle.dump(save_dict, f)

    def load_ann_index(self, path: Union[str, Path]) -> Any:
        """Load an ANN index from disk.

        Parameters
        ----------
        path : Union[str, Path]
            Path to load the index from.

        Returns
        -------
        Any
            The loaded ANN index.

        Raises
        ------
        ValueError
            If loaded index similarity metric doesn't match current metric.
        """
        with open(path, 'rb') as f:
            load_dict = pickle.load(f)

        if load_dict['similarity'] != self.similarity:
            raise ValueError(
                f"Loaded index similarity metric ({load_dict['similarity']}) does not match "
                f"current similarity metric ({self.similarity})"
            )

        self.index = load_dict['index']
        self.index_backend = load_dict['backend']
        self.index_kwargs = load_dict['index_kwargs']
        self.index_k = load_dict['index_k']

        return self.index
