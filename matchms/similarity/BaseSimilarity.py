from abc import abstractmethod
from typing import Optional, Sequence, Tuple
import numpy as np
from tqdm import tqdm
from matchms.similarity.COOIndex import COOIndex
from matchms.similarity.COOMatrix import COOMatrix
from matchms.similarity.ScoreFilter import FilterScoreByValue
from matchms.typing import SpectrumType


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
        Data type for the score output, e.g. "float" or [("score", "float"), ("matches", "int")].
        If multiple data types are set, the main score should be set to "score" (used as default for filtering).
    """

    # Set key characteristics as class attributes
    is_commutative = True
    score_datatype = np.float64

    def __init__(self, score_filters: Optional[Tuple[FilterScoreByValue, ...]] = None):
        """
        Parameters
        ----------
        score_filters: tuple, optional
            A tuple of filter objects to apply to each similarity score.
            - If you do not wish to apply any filtering, pass an empty tuple and use the dense matrix method (`matrix()`).
            - For cases where you expect to filter out more than 90% of the computed scores, consider using
              the sparse array methods (see `sparse_array()`).
        """
        self.score_filters = score_filters if score_filters is not None else ()

    @abstractmethod
    def pair(self, reference: SpectrumType, query: SpectrumType) -> np.ndarray:
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
        references: Sequence[SpectrumType],
        queries: Sequence[SpectrumType],
        is_symmetric: bool = False,
        mask_indices: COOIndex = None,
    ) -> np.ndarray:
        """
        Compute a dense similarity matrix for all pairs of reference and query spectra.

        Use this method when filtering is either required (with a provided mask) or when working
        with moderately sized datasets. If no filtering is needed and you don't have a mask, simply
        pass an empty tuple for `score_filters` and use this method.

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
            This is helpful when a previous score already filters out many pairs, reducing computation time and memory footprint.
        """
        if mask_indices is None:
            return self._matrix_without_mask_with_filter(
                references, queries, is_symmetric=is_symmetric
            )
        return self._matrix_with_mask_with_filter(
            references, queries, mask_indices=mask_indices, is_symmetric=is_symmetric
        )

    def sparse_array(
        self,
        references: Sequence[SpectrumType],
        queries: Sequence[SpectrumType],
        mask_indices: COOIndex = None,
        is_symmetric=False,
    ) -> COOMatrix:
        """
        Compute a sparse array (in COO format) of similarity scores.

        Use this method if you expect heavy filtering (i.e. many scores are dropped) or
        if you want to compute scores only for a selected set of index pairs. By using `sparse_array()` this can reduce
        the memory footprint if many scores are dropped.

        Note:
          - If no filtering is required (empty score_filters) and no mask is provided,
            it is recommended to use `matrix()` and then convert the dense matrix to COO format.
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
        if len(self.score_filters) == 0 and mask_indices:
            return self._sparse_array_with_mask_without_filter(
                references, queries, mask_indices=mask_indices
            )
        if len(self.score_filters) != 0 and mask_indices is None:
            return self._sparse_array_without_mask_with_filter(
                references, queries, is_symmetric=is_symmetric
            )
        if len(self.score_filters) != 0 and mask_indices:
            return self._sparse_array_with_mask_with_filter(
                references, queries, mask_indices
            )

        # TODO: replace with matrix computation followed by a conversion to COO array.
        # (and a warning that this is not a good idea) do this once we settle on a COO Array format (e.g. using the sparse package)
        raise ValueError(
            "If no masking or score filters is needed, please use matrix() instead"
        )

    # --- Dense Matrix Computations ---

    def _matrix_without_mask_with_filter(
        self,
        references: Sequence[SpectrumType],
        queries: Sequence[SpectrumType],
        is_symmetric: bool = False,
    ) -> np.ndarray:
        """
        Compute a dense similarity matrix without a mask, then apply score filters (if any).

        This method is used internally by `matrix()` when no mask is provided.
        """
        sim_matrix = self._matrix_without_mask_without_filter(
            references, queries, is_symmetric=is_symmetric
        )

        for score_filter in self.score_filters:
            sim_matrix = score_filter.filter_matrix(sim_matrix)

        return sim_matrix

    def _matrix_without_mask_without_filter(
        self,
        references: Sequence[SpectrumType],
        queries: Sequence[SpectrumType],
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
        sim_matrix = np.zeros(
            (len(references), len(queries)), dtype=self.score_datatype
        )
        if is_symmetric:
            if len(references) != len(queries):
                raise ValueError(
                    f"Found unequal number of spectra {len(references)} and {len(queries)} while `is_symmetric` is True."
                )

            # Compute pairwise similarities
            for i_ref, reference in enumerate(
                tqdm(references, "Calculating similarities")
            ):
                for i_query, query in enumerate(
                    queries[i_ref:], start=i_ref
                ):  # Compute only upper triangle
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

    def _matrix_with_mask_with_filter(
        self,
        references: Sequence[SpectrumType],
        queries: Sequence[SpectrumType],
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
        sim_matrix = np.zeros(
            (len(references), len(queries)), dtype=self.score_datatype
        )
        for i_row, i_col in tqdm(mask_indices, desc="Calculating sparse similarities"):
            score = self.pair(references[i_row], queries[i_col])
            if np.all(
                [score_filter.keep_score(score) for score_filter in self.score_filters]
            ):
                # if not all filters pass the score is not added (so remains 0)
                sim_matrix[i_row, i_col] = score
                if is_symmetric:
                    sim_matrix[i_col, i_row] = score
        return sim_matrix

    # --- Sparse Matrix Computations ---

    def _sparse_array_without_mask_with_filter(
        self,
        references: Sequence[SpectrumType],
        queries: Sequence[SpectrumType],
        is_symmetric: bool = False,
    ) -> COOMatrix:
        """
        Compute a sparse similarity matrix (COO format) with filtering, without using a mask.

        This method is optimized for cases where filtering drops most scores.
        Important to note: Memory-wise, this method is only worth using if less than 1/12th of the scores are kept.
        Otherwise, it is generally better to use the `matrix()` method, followed by a filter.

        Parameters
        ----------
        references
            List of reference objects
        queries
            List of query objects
        is_symmetric
            If True, leverages commutativity to compute only half the matrix. Defaults to False.
        """
        n_rows = len(references)
        n_cols = len(queries)

        if is_symmetric and n_rows != n_cols:
            raise ValueError(
                f"Found unequal number of spectra {n_rows} and {n_cols} while `is_symmetric` is True."
            )

        idx_row = []
        idx_col = []
        scores = []
        # Wrap the outer loop with tqdm to track progress
        for i_ref, reference in enumerate(
            tqdm(references[:n_rows], desc="Calculating similarities")
        ):
            if is_symmetric and self.is_commutative:
                for i_query, query in enumerate(queries[i_ref:n_cols], start=i_ref):
                    score = self.pair(reference, query)
                    # Check if the score passes the filter before storing.
                    if np.all(
                        [
                            score_filter.keep_score(score)
                            for score_filter in self.score_filters
                        ]
                    ):
                        idx_row += [i_ref, i_query]
                        idx_col += [i_query, i_ref]
                        scores += [score, score]
            else:
                for i_query, query in enumerate(queries[:n_cols]):
                    score = self.pair(reference, query)
                    # Check if the score passes the filter before storing.
                    if np.all(
                        [
                            score_filter.keep_score(score)
                            for score_filter in self.score_filters
                        ]
                    ):
                        idx_row.append(i_ref)
                        idx_col.append(i_query)
                        scores.append(score)
        return COOMatrix(
            row_idx=idx_row,
            column_idx=idx_col,
            scores=scores,
            scores_dtype=self.score_datatype,
        )

    def _sparse_array_with_mask_without_filter(
        self,
        references: Sequence[SpectrumType],
        queries: Sequence[SpectrumType],
        mask_indices: COOIndex,
    ) -> COOMatrix:
        """Optional: Provide optimized method to calculate a sparse matrix of similarity scores.

        Compute similarity scores for pairs of reference and query spectra as given by the indices
        idx_row (references) and idx_col (queries). If no method is added here, the following naive
        implementation (i.e. a for-loop) is used.

        Parameters
        ----------
        references
            List of reference objects
        queries
            List of query objects
        mask_indices
            The row column index pairs for which a score should be calculated.
        """
        if len(self.score_filters) > 0:
            raise ValueError(
                "Don't run _sparse_array_with_mask_without_filter if score_filters are set. "
                "Instead run _sparse_array_with_mask_with_filter"
            )
        scores = np.zeros((len(mask_indices)), dtype=self.score_datatype)
        for i, (i_row, i_col) in enumerate(
            tqdm(mask_indices, desc="Calculating sparse similarities")
        ):
            scores[i] = self.pair(references[i_row], queries[i_col])
        return COOMatrix(
            row_idx=mask_indices.idx_row,
            column_idx=mask_indices.idx_col,
            scores=scores,
            scores_dtype=self.score_datatype,
        )

    def _sparse_array_with_mask_with_filter(
        self,
        references: Sequence[SpectrumType],
        queries: Sequence[SpectrumType],
        mask_indices: COOIndex,
    ) -> COOMatrix:
        """
        Compute a sparse similarity matrix (COO format) using a mask and applying score filters.

        Only the (row, column) pairs specified in mask_indices are computed, and only scores that pass
        all filters are stored.

        Please note: If the mask_indices contain more than 1/12th of all indices, this method could become less memory efficient
        than `_sparse_array_without_mask_with_filter`.

        Parameters
        ----------
        references
            List of reference objects
        queries
            List of query objects
        mask_indices
            Specifies the (row, column) pairs for which to compute scores.
        """
        scores = []
        idx_row = []
        idx_col = []
        for row, col in tqdm(mask_indices, desc="Calculating sparse similarities"):
            score = self.pair(references[row], queries[col])
            # Check if the score passes the filter before storing.
            if np.all(
                [score_filter.keep_score(score) for score_filter in self.score_filters]
            ):
                idx_row.append(row)
                idx_col.append(col)
                scores.append(score)
        return COOMatrix(
            row_idx=idx_row,
            column_idx=idx_col,
            scores=scores,
            scores_dtype=self.score_datatype,
        )

    def to_dict(self) -> dict:
        """Return a dictionary representation of a similarity function."""
        return {"__Similarity__": self.__class__.__name__, **self.__dict__}
