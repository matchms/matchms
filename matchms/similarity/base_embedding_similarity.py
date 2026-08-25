import pickle
from abc import abstractmethod
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from matchms.scores import Scores
from matchms.similarity.base_similarity import BaseSimilarity
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
    is_commutative = True
    score_datatype = np.float64
    score_fields = ("score",)

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
            spectra: Iterable[SpectrumType] | None = None,
            npy_path: str | Path | None = None) -> np.ndarray:
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
                embeddings = self.load_embeddings(npy_path)
            else:
                if spectra is None:
                    raise ValueError("Spectra must be provided when storing newly computed embeddings.")
                embeddings = self.compute_embeddings(spectra)
                self.store_embeddings(npy_path, embeddings)
        else:
            embeddings = self.compute_embeddings(spectra)

        return self._validate_embeddings(embeddings)

    def pair(self, spectrum_1: SpectrumType, spectrum_2: SpectrumType) -> float:
        """Compute similarity between a pair of spectra.

        Parameters
        ----------
        spectrum_1 : SpectrumType
            Reference spectrum.
        spectrum_2 : SpectrumType
            Query spectrum.
        """
        score = self.matrix([spectrum_1], [spectrum_2], progress_bar=False).to_array()[0, 0]
        return np.asarray(score, dtype=self.score_datatype)

    def matrix(
        self,
        spectra_1: Sequence[SpectrumType],
        spectra_2: Sequence[SpectrumType] | None = None,
        score_fields: Sequence[str] | None = None,
        progress_bar: bool = True,
    ) -> Scores:
        """Compute similarity matrix between spectra_1 and spectra_2.

        Parameters
        ----------
        spectra_1
            First collection of spectra.
        spectra_2
            Second collection of spectra. If ``None``, compare ``spectra_1``
            against itself.
        score_fields
            Requested score fields. Embedding similarities expose only
            ``("score",)``.
        progress_bar
            Included for API compatibility. Embeddings are computed in batch and
            this implementation currently does not display a progress bar.

        Returns
        -------
        np.ndarray
            Similarity matrix.

        Raises
        ------
        ValueError
            If array_type is not "numpy" or is_symmetric is False.
        """
        del progress_bar  # Not used in this implementation, but included for API compatibility.

        selected_fields = self._resolve_score_fields(score_fields)
        if selected_fields != ("score",):
            raise NotImplementedError(
                f"{self.__class__.__name__}.matrix() supports only score_fields=('score',)."
            )

        spectra_2, is_symmetric = self._prepare_inputs(spectra_1, spectra_2)

        embeddings_1 = self._validate_embeddings(self.compute_embeddings(spectra_1))
        if is_symmetric:
            embeddings_2 = embeddings_1
        else:
            embeddings_2 = self._validate_embeddings(self.compute_embeddings(spectra_2))

        similarity_matrix = self.pairwise_similarity_fn(embeddings_1, embeddings_2)
        similarity_matrix = np.asarray(similarity_matrix, dtype=self.score_datatype)

        if similarity_matrix.shape != (len(spectra_1), len(spectra_2)):
            raise ValueError(
                "Embedding similarity matrix has unexpected shape "
                f"{similarity_matrix.shape}; expected {(len(spectra_1), len(spectra_2))}."
            )

        return Scores({"score": similarity_matrix})

    def compute_similarity_matrix_from_embeddings(
        self,
        embeddings_1: np.ndarray,
        embeddings_2: np.ndarray | None = None,
    ) -> np.ndarray:
        """Compute a raw NumPy similarity matrix from precomputed embeddings.

        This helper keeps the old raw-array use case available without changing
        the public :meth:`matrix` contract inherited from ``BaseSimilarity``.
        """
        embeddings_1 = self._validate_embeddings(embeddings_1)
        embeddings_2 = embeddings_1 if embeddings_2 is None else self._validate_embeddings(embeddings_2)
        return np.asarray(self.pairwise_similarity_fn(embeddings_1, embeddings_2), dtype=self.score_datatype)

    def build_ann_index(
            self,
            reference_spectra: Iterable[SpectrumType] | None = None,
            embeddings_path: str | Path | None = None,
            k: int = 100,
            index_backend: str = "pynndescent",
            **index_kwargs) -> Any:
        """Build an ANN index for the input spectra.

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
        embeddings_reference = self.get_embeddings(reference_spectra, embeddings_path)

        if index_backend == "pynndescent":
            if not pynndescent:
                raise ImportError("pynndescent is not installed. Please install it with `pip install pynndescent`.")
            self.index_backend = index_backend
            self.index_k = k
            self.index_kwargs = index_kwargs

            index = pynndescent.NNDescent(
                embeddings_reference,
                metric=self.similarity,
                n_neighbors=k,
                **index_kwargs,
            )
        else:
            raise ValueError(f"Only pynndescent is supported for now. Got {index_backend}.")

        self.index = index
        return self.index

    def get_anns(
            self,
            query_spectra: Iterable[SpectrumType] | np.ndarray,
            k: int = 100) -> tuple[np.ndarray, np.ndarray]:
        """Get approximate nearest neighbors for input spectra.

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
            embeddings_query = self._validate_embeddings(query_spectra)
        else:
            embeddings_query = self._validate_embeddings(self.compute_embeddings(query_spectra))

        if self.index_backend == "pynndescent":
            neighbors, distances = self.index.query(embeddings_query, k=k)
            similarities = self._distances_to_similarities(distances)
        else:
            raise ValueError(f"Only pynndescent is supported for now. Got {self.index_backend}.")
        return neighbors, similarities

    def get_index_anns(self) -> tuple[np.ndarray, np.ndarray]:
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
    def _validate_embeddings(embeddings: np.ndarray) -> np.ndarray:
        """Validate and return embeddings as a two-dimensional float array."""
        embeddings = np.asarray(embeddings)
        if embeddings.ndim != 2:
            raise ValueError(f"Expected 2D embeddings array, got {embeddings.ndim}D array.")
        if not np.issubdtype(embeddings.dtype, np.number):
            raise ValueError("Embeddings must contain numeric values.")
        return embeddings

    @staticmethod
    def load_embeddings(npy_path: str | Path) -> np.ndarray:
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
        embeddings = np.load(npy_path)
        return BaseEmbeddingSimilarity._validate_embeddings(embeddings)

    @staticmethod
    def store_embeddings(npy_path: str | Path, embeddings: np.ndarray) -> None:
        """Store embeddings in a numpy file.

        Parameters
        ----------
        npy_path : Union[str, Path]
            Path to save the embeddings to.
        embeddings : np.ndarray
            Embeddings array to store.
        """
        embeddings = BaseEmbeddingSimilarity._validate_embeddings(embeddings)
        np.save(npy_path, embeddings)

    def save_ann_index(self, path: str | Path) -> None:
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
            "index": self.index,
            "backend": self.index_backend,
            "similarity": self.similarity,
            "index_kwargs": self.index_kwargs,
            "index_k": self.index_k,
        }

        with open(path, "wb") as f:
            pickle.dump(save_dict, f)

    def load_ann_index(self, path: str | Path) -> Any:
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
        with open(path, "rb") as f:
            load_dict = pickle.load(f)

        if load_dict["similarity"] != self.similarity:
            raise ValueError(
                f"Loaded index similarity metric ({load_dict['similarity']}) does not match "
                f"current similarity metric ({self.similarity})"
            )

        self.index = load_dict["index"]
        self.index_backend = load_dict["backend"]
        self.index_kwargs = load_dict["index_kwargs"]
        self.index_k = load_dict["index_k"]

        return self.index
