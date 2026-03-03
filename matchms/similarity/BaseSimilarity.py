from abc import abstractmethod
from typing import Optional, Sequence
import numpy as np
import numpy.typing as npt
from scipy.sparse import coo_matrix
from tqdm import tqdm
from matchms.similarity.COOIndex import COOIndex
from matchms.Spectrum import Spectrum


class BaseSimilarity:
    """
    Base class for similarity functions.

    To implement a custom similarity measure, subclass BaseSimilarity and implement
    the `pair` method, which calculates the similarity between two spectra.

    Attributes
    ----------
    is_commutative:
        Indicates whether the similarity function is commutative (i.e. similarity(A, B) == similarity(B, A)).
        Defaults to True.
    score_datatype:
        Data type for the score output, e.g. "float" or "int"
    """

    # Set key characteristics as class attributes
    is_commutative = True
    score_datatype = np.float64

    @abstractmethod
    def pair(self, reference: Spectrum, query: Spectrum) -> np.ndarray:
        """
        Compute the similarity score for a single pair of spectra.

        Parameters
        ----------
        reference
            A single reference spectrum.
        query
            A single query spectrum.

        Returns
            The similarity score as numpy array (with dtype given by self.score_datatype).
            For example: return np.asarray(score, dtype=self.score_datatype)
        """
        raise NotImplementedError

    def matrix(
        self,
        references: Sequence[Spectrum],
        queries: Sequence[Spectrum],
        is_symmetric: bool = False,
        mask_indices: Optional[COOIndex] = None,
    ) -> npt.NDArray:
        """
        Compute a dense similarity matrix for pairs of reference and query spectra.
        If a mask is given only the pairs of spectra given by the mask will be computed.

        Parameters
        ----------
        references:
            Collection of reference spectra.
        queries:
            Collection of query spectra.
        is_symmetric:
            If True, indicates that the similarity matrix is symmetric (i.e., references and queries are the same).
            Only the upper triangle is computed and then mirrored. Defaults to False.
        mask_indices:
            A COOIndex instance specifying which pairs to compute.
            If provided, only the specified index pairs will be computed (others remain zero).
            This is helpful when a previous score already filters out many pairs, reducing computation time.
        """
        if mask_indices is None:
            return self._matrix_without_mask(references, queries, is_symmetric=is_symmetric)
        return self._matrix_with_mask(references, queries, mask_indices=mask_indices, is_symmetric=is_symmetric)

    def sparse_array(
        self,
        references: Sequence[Spectrum],
        queries: Sequence[Spectrum],
        mask_indices: Optional[COOIndex] = None,
        is_symmetric=False,
    ) -> coo_matrix:
        """
        Compute a sparse array (in COO format) of similarity scores.

        Use this method if you expect heavy filtering (i.e. many scores are dropped) or
        if you want to compute scores only for a selected set of index pairs. By using `sparse_array()` this can reduce
        the memory footprint if many scores are dropped.

        Note:
          - If no mask is provided, it is recommended to use `matrix()` and then convert the dense matrix to COO format.
          - When a mask is provided, only the pairs specified in the mask are computed.

        Parameters
        ----------
        references:
            A collection of reference spectra.
        queries:
            A collection of query spectra.
        mask_indices:
            A COOIndex instance specifying the (row, column) pairs to compute. Defaults to None.
        is_symmetric:
            If True, assumes that the matrix is symmetric. Defaults to False.
        """
        if mask_indices:
            return self._sparse_array_with_mask(references, queries, mask_indices=mask_indices)
        return self._sparse_array_without_mask(references, queries, is_symmetric=is_symmetric)

    # --- Dense Matrix Computations ---

    def _matrix_without_mask(
        self,
        references: Sequence[Spectrum],
        queries: Sequence[Spectrum],
        is_symmetric: bool = False,
    ) -> np.ndarray:
        """
        Compute a dense similarity matrix for all pairs of reference and query spectra.

        Parameters
        ----------
        references:
            Collection of reference spectra.
        queries:
            Collection of query spectra.
        is_symmetric:
            Indicates if the similarity matrix is symmetric (e.g., for all-vs-all comparisons).
            When True, only the upper triangle of the matrix is computed and then mirrored,
            which can reduce computation time.
        """
        sim_matrix = np.zeros((len(references), len(queries)), dtype=self.score_datatype)
        if is_symmetric:
            if len(references) != len(queries):
                raise ValueError(
                    f"Found unequal number of spectra {len(references)} and {len(queries)} while `is_symmetric` is True."
                )

            # Compute pairwise similarities for symmetric case
            for i_ref, reference in enumerate(tqdm(references, "Calculating similarities")):
                for i_query, query in enumerate(queries[i_ref:], start=i_ref):  # Compute only upper triangle
                    score = self.pair(reference, query)
                    sim_matrix[i_ref, i_query] = score
                    sim_matrix[i_query, i_ref] = score
        else:
            # Compute pairwise similarities
            for i, reference in enumerate(tqdm(references, "Calculating similarities")):
                for j, query in enumerate(queries):
                    score = self.pair(reference, query)
                    sim_matrix[i, j] = score
        return sim_matrix

    def _matrix_with_mask(
        self,
        references: Sequence[Spectrum],
        queries: Sequence[Spectrum],
        mask_indices: COOIndex,
        is_symmetric: bool = False,
    ) -> np.ndarray:
        """
        Compute a dense similarity matrix using a provided mask.

        Only the (row, column) pairs specified in mask_indices are computed. All pairs not in mask_indices are set to 0.
        Score filters are applied to each computed score.

        Parameters
        ----------
        references:
            Collection of reference spectra.
        queries:
            Collection of query spectra.
        mask_indices:
            Specifies which index pairs to compute.
        is_symmetric:
            If True, mirrors the computed score to the symmetric position. Defaults to False.
        """
        # todo implement is_symmetric by converting the mask to symmetric mask
        sim_matrix = np.zeros((len(references), len(queries)), dtype=self.score_datatype)
        for i_row, i_col in tqdm(mask_indices, desc="Calculating sparse similarities"):
            score = self.pair(references[int(i_row)], queries[int(i_col)])
            sim_matrix[i_row, i_col] = score
            if is_symmetric:
                sim_matrix[i_col, i_row] = score
        return sim_matrix

    # --- Sparse Matrix Computations ---

    def _sparse_array_with_mask(
        self,
        references: Sequence[Spectrum],
        queries: Sequence[Spectrum],
        mask_indices: COOIndex,
    ) -> coo_matrix:
        """Compute similarity scores for pairs of reference and query spectra as given by the indices
        idx_row (references) and idx_col (queries).

        Parameters
        ----------
        references
            List of reference objects
        queries
            List of query objects
        mask_indices
            The row column index pairs for which a score should be calculated.
        """
        scores = np.zeros((len(mask_indices)), dtype=self.score_datatype)
        for i, (i_row, i_col) in enumerate(tqdm(mask_indices, desc="Calculating sparse similarities")):
            scores[i] = self.pair(references[int(i_row)], queries[int(i_col)])
        return coo_matrix((scores, (mask_indices.idx_row, mask_indices.idx_col)), shape=(len(references), len(queries)))

    def _sparse_array_without_mask(
        self, references: Sequence[Spectrum], queries: Sequence[Spectrum], is_symmetric: bool
    ):
        # TODO: replace with matrix computation followed by a conversion to COO array.
        # (and a warning that this is not a good idea) do this once we settle on a COO Array format (e.g. using the sparse package)
        raise NotImplementedError("If no masking is needed, please use matrix() instead")

    def to_dict(self) -> dict:
        """Return a dictionary representation of a similarity function."""
        return {"__Similarity__": self.__class__.__name__, **self.__dict__}
