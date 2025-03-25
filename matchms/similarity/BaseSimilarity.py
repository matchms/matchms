from abc import abstractmethod
from typing import Iterable, List, Tuple
import numpy as np
from tqdm import tqdm
from matchms.similarity.COOIndex import COOIndex
from matchms.similarity.COOMatrix import COOMatrix
from matchms.similarity.ScoreFilter import FilterScoreByValue
from matchms.typing import SpectrumType


class BaseSimilarity:
    """Similarity function base class.
    When building a custom similarity measure, inherit from this class and implement
    the desired methods.

    Attributes
    ----------
    is_commutative:
       Whether similarity function is commutative, which means that the order of spectra
       does not matter (similarity(A, B) == similarity(B, A)). Default is True.
    score_datatype:
        Data type for the score output, e.g. "float" or [("score", "float"), ("matches", "int")].
        If multiple data types are set, the main score should be set to "score" (used as default for filtering).
    """
    # Set key characteristics as class attributes
    is_commutative = True
    score_datatype = np.float64

    def __init__(self, score_filters: Tuple[FilterScoreByValue]):
        """

        Attributes
        ----------
        score_filters:
            Tuple of filters that should be applied to each score before scoring. If you want to run without filtering,
            it is best to run matrix(). Filters can also be run after computing a matrix first, however, for strict
            filtering this implementation can be more memory efficient. Only use matrix_with_filter if you expect to
            filter out more than 90% of your data. Otherwise, it is more memory efficient to first run matrix, followed by
            filtering the matrix.
        """
        self.score_filters = score_filters

    @abstractmethod
    def pair(self, reference: SpectrumType, query: SpectrumType) -> np.ndarray:
        """Method to calculate the similarity for one input pair.

        Parameters
        ----------
        reference
            Single reference spectrum.
        query
            Single query spectrum.

        Returns
            The similarity score as numpy array (using self.score_datatype). For instance returning
            np.asarray(score, dtype=self.score_datatype)
        """
        raise NotImplementedError

    def matrix(self, references: np.ndarray[SpectrumType],
               queries: np.ndarray[SpectrumType],
               is_symmetric: bool = False,
               mask_indices: COOIndex = None,
               ) -> np.ndarray:
        """
        Compute a dense similarity matrix for all pairs of reference and query spectra.

        Parameters
        ----------
        references:
            Collection of reference spectra.
        queries:
            Collection of query spectra.
        mask_indices:
            Indices to calculate scores for the rest is set to 0.
        is_symmetric:
            Indicates if the similarity matrix is symmetric (e.g., for all-vs-all comparisons).
            When True, only the upper triangle of the matrix is computed and then mirrored,
            which can reduce computation time.
        """
        if mask_indices is None:
            return self._matrix_without_mask_with_filter(references, queries, is_symmetric=is_symmetric)
        return self._matrix_with_mask(references, queries,
                                      mask_indices=mask_indices, is_symmetric=is_symmetric)

    def sparse_array(self, references: List[SpectrumType], queries: List[SpectrumType],
                     mask_indices: COOIndex = None, is_symmetric = False) -> COOMatrix:
        if len(self.score_filters) == 0 and mask_indices is not None:
            return self._sparse_array_with_mask_without_filter(references, queries, mask_indices=mask_indices)
        if len(self.score_filters) != 0 and mask_indices is None:
            return self._sparse_array_with_filter_without_mask(references, queries, is_symmetric=is_symmetric)
        if len(self.score_filters) != 0 and mask_indices is not None:
            return self._sparse_array_with_filter(references, queries, mask_indices)

        # if score_filters is None and mask_indices is None:
        # todo replace with using matrix followed by a conversion to COO array. (and a warning that this is not a
        #  good idea) do this once we settle on a COO Array format (e.g. using the sparse package)
        raise ValueError("If no masking or score filters is needed, please use matrix() instead")

    def _matrix_without_mask_with_filter(self,
                          references: np.ndarray[SpectrumType], queries: np.ndarray[SpectrumType],
                          is_symmetric: bool = False):
        sim_matrix = self._matrix_without_mask_without_filter(references, queries, is_symmetric=is_symmetric)
        for score_filter in self.score_filters:
            sim_matrix = score_filter.filter_matrix(sim_matrix)
        return sim_matrix

    def _matrix_without_mask_without_filter(self,
                                            references: np.ndarray[SpectrumType], queries: np.ndarray[SpectrumType],
                                            is_symmetric: bool = False
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
                raise ValueError(f"Found unequal number of spectra {len(references)} and {len(queries)} while `is_symmetric` is True.")

            # Compute pairwise similarities
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

    def _matrix_with_mask(self,
                          references: np.ndarray[SpectrumType], queries: np.ndarray[SpectrumType],
                          mask_indices: COOIndex,
                          is_symmetric: bool = False
                          ) -> np.ndarray:
        sim_matrix = np.zeros((len(references), len(queries)), dtype=self.score_datatype)
        for i_row, i_col in tqdm(mask_indices, desc="Calculating sparse similarities"):
            score = self.pair(references[i_row], queries[i_col])
            if np.all(score_filter.keep_score(score) for score_filter in self.score_filters):
                # if not all filters pass the score is not added (so remains 0)
                sim_matrix[i_row, i_col] = score
                if is_symmetric:
                    sim_matrix[i_col, i_row] = score
        return sim_matrix


    def _sparse_array_with_filter_without_mask(self, references: Iterable[SpectrumType], queries: Iterable[SpectrumType],
                           is_symmetric: bool = False, ) -> COOMatrix:
        """Optional: Provide optimized method to calculate a sparse matrix with filtering applied directly.
        This is helpfull if the filter function is removing most scores. Important note, per score this takes about 12x
        the amount of memory. Therefore, doing filtering during compute is only worth it if you keep less than 1/12th of
        the scores. Otherwise, it is best to just use matrix, followed by a filter.

        Parameters
        ----------
        references
            List of reference objects
        queries
            List of query objects
        is_symmetric
            Set to True when *references* and *queries* are identical (as for instance for an all-vs-all
            comparison). By using the fact that score[i,j] = score[j,i] the calculation will be about
            2x faster.
        """
        n_rows = len(references)
        n_cols = len(queries)

        if is_symmetric and n_rows != n_cols:
            raise ValueError(f"Found unequal number of spectra {n_rows} and {n_cols} while `is_symmetric` is True.")

        idx_row = []
        idx_col = []
        scores = []
        # Wrap the outer loop with tqdm to track progress
        for i_ref, reference in enumerate(tqdm(references[:n_rows], desc="Calculating similarities")):
            if is_symmetric and self.is_commutative:
                for i_query, query in enumerate(queries[i_ref:n_cols], start=i_ref):
                    score = self.pair(reference, query)
                    # Check if the score passes the filter before storing.
                    if np.all(score_filter.keep_score(score) for score_filter in self.score_filters):
                        idx_row += [i_ref, i_query]
                        idx_col += [i_query, i_ref]
                        scores += [score, score]
            else:
                for i_query, query in enumerate(queries[:n_cols]):
                    score = self.pair(reference, query)
                    # Check if the score passes the filter before storing.
                    if np.all(score_filter.keep_score(score) for score_filter in self.score_filters):
                        idx_row.append(i_ref)
                        idx_col.append(i_query)
                        scores.append(score)
        return COOMatrix(row_idx=idx_row, column_idx=idx_col, scores=scores)


    def _sparse_array_with_mask_without_filter(self, references: List[SpectrumType], queries: List[SpectrumType],
                     mask_indices: COOIndex) -> COOMatrix:
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
            raise ValueError("Don't run _sparse_array_with_mask_without_filter if score_filters are set. "
                             "Instead run _sparse_array_with_filter")
        scores = np.zeros((len(mask_indices)), dtype=self.score_datatype)
        for i, (i_row, i_col) in enumerate(tqdm(mask_indices, desc="Calculating sparse similarities")):
            scores[i] = self.pair(references[i_row], queries[i_col])
        return COOMatrix(row_idx=mask_indices.idx_row, column_idx=mask_indices.idx_col, scores=scores)

    def _sparse_array_with_filter(self, references: List[SpectrumType], queries: List[SpectrumType],
                                 mask_indices: COOIndex) -> COOMatrix:
        """Uses a mask to compute only the required scores and filters scores that do not pass the filter.

        This method most of the time does not make sense. It is only worth it if you want to store less than 1/12th
        of the computed scores. This is not the case most of the time, so sparse_array is most of the time the most
        memory efficient.

        Parameters
        ----------
        references
            List of reference objects
        queries
            List of query objects
        mask_indices
            The row column index pairs for which a score should be calculated.
        """
        scores = []
        idx_row = []
        idx_col = []
        for row, col in tqdm(mask_indices, desc="Calculating sparse similarities"):
            score = self.pair(references[row], queries[col])
            # Check if the score passes the filter before storing.
            if np.all(score_filter.keep_score(score) for score_filter in self.score_filters):
                idx_row.append(row)
                idx_col.append(col)
                scores.append(score)
        return COOMatrix(row_idx=idx_row, column_idx=idx_col, scores=scores)


    def to_dict(self) -> dict:
        """Return a dictionary representation of a similarity function."""
        return {"__Similarity__": self.__class__.__name__,
                **self.__dict__}
