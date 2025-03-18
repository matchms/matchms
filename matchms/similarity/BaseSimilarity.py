from abc import abstractmethod
from typing import List
import numpy as np
from sparsestack import StackedSparseArray
from tqdm import tqdm
from matchms.typing import SpectrumType


class BaseSimilarity:
    """Similarity function base class.
    When building a custom similarity measure, inherit from this class and implement
    the desired methods.

    Attributes
    ----------
    is_commutative
       Whether similarity function is commutative, which means that the order of spectra
       does not matter (similarity(A, B) == similarity(B, A)). Default is True.
    """
    # Set key characteristics as class attributes
    is_commutative = True
    # Set output data type, e.g. "float" or [("score", "float"), ("matches", "int")]
    score_datatype = np.float64

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
            score as numpy array (using self.score_datatype). For instance returning
            np.asarray(score, dtype=self.score_datatype)
        """
        raise NotImplementedError

    def matrix(self,
                     references: List[SpectrumType], queries: List[SpectrumType],
                     is_symmetric: bool = False
                     ) -> np.ndarray:
        sim_matrix = np.zeros((len(references), len(queries)), dtype=self.score_datatype)
        if is_symmetric:
            if len(references) != len(queries):
                raise ValueError(f"Found unequal number of spectra {len(references)} and {len(queries)} while `is_symmetric` is True.")

            # Compute pairwise similarities
            for i_ref, reference in enumerate(tqdm(references, "Calculating similarities")):
                for i_query, query in enumerate(queries[i_ref:], start=i_ref):  # Compute only upper triangle
                    sim_matrix[i_ref, i_query] = self.pair(reference, query)
                    # todo once Scores can handle symmetric compute don't store
                    sim_matrix[i_query, i_ref] = sim_matrix[i_ref, i_query]
        else:
            # Compute pairwise similarities
            for i, reference in enumerate(tqdm(references, "Calculating similarities")):
                for j, query in enumerate(queries):
                    sim_matrix[i, j] = self.pair(reference, query)
        return sim_matrix

    def matrix_with_filter(self, references: List[SpectrumType], queries: List[SpectrumType],
                           is_symmetric: bool = False) -> StackedSparseArray:
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
        #pylint: disable=too-many-locals
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
                    if self.keep_score(score):
                        # todo never store duplicated scores, this should be handled by the scores object.
                        idx_row += [i_ref, i_query]
                        idx_col += [i_query, i_ref]
                        scores += [score, score]
            else:
                for i_query, query in enumerate(queries[:n_cols]):
                    score = self.pair(reference, query)
                    if self.keep_score(score):
                        idx_row.append(i_ref)
                        idx_col.append(i_query)
                        scores.append(score)

        idx_row = np.array(idx_row, dtype=np.int_)
        idx_col = np.array(idx_col, dtype=np.int_)
        scores_data = np.array(scores, dtype=self.score_datatype)
        # Todo replace with returning a sparse matrix (stacked is not needed.)
        scores_array = StackedSparseArray(n_rows, n_cols)
        scores_array.add_sparse_data(idx_row, idx_col, scores_data, "")
        return scores_array

    def sparse_array(self, references: List[SpectrumType], queries: List[SpectrumType],
                     idx_row, idx_col, is_symmetric: bool = False):
        """Optional: Provide optimized method to calculate an sparse matrix of similarity scores.

        Compute similarity scores for pairs of reference and query spectra as given by the indices
        idx_row (references) and idx_col (queries). If no method is added here, the following naive
        implementation (i.e. a for-loop) is used.

        Parameters
        ----------
        references
            List of reference objects
        queries
            List of query objects
        idx_row
            List/array of row indices
        idx_col
            List/array of column indices
        is_symmetric
            Set to True when *references* and *queries* are identical (as for instance for an all-vs-all
            comparison). By using the fact that score[i,j] = score[j,i] the calculation will be about
            2x faster.
        """
        # pylint: disable=too-many-arguments
        if is_symmetric is True:
            pass  # TODO: consider implementing faster method for symmetric cases

        assert idx_row.shape == idx_col.shape, "col and row indices must be of same shape"
        scores = np.zeros((len(idx_row)), dtype=self.score_datatype)  # TODO: switch to sparse matrix
        # Use tqdm to track progress through the indices
        for i, row in enumerate(tqdm(idx_row, desc="Calculating sparse similarities")):
            col = idx_col[i]
            scores[i] = self.pair(references[row], queries[col])
        return scores

    def keep_score(self, score: np.ndarray):
        """In the `.matrix` method scores will be collected in a sparse way.
        Overwrite this method here if values other than `False` or `0` should
        not be stored in the final collection.
        """
        if len(score.dtype) > 1:  # if structured array
            valuelike = True
            for dtype_name in score.dtype.names:
                valuelike = valuelike and (score[dtype_name] != 0)
            return valuelike
        return score != 0

    def to_dict(self) -> dict:
        """Return a dictionary representation of a similarity function."""
        return {"__Similarity__": self.__class__.__name__,
                **self.__dict__}


def get_indexes_dense_matrix(rows: int, columns: int, is_symmetric=False) -> Tuple[np.ndarray, np.ndarray]:
    """Computes COO indexes for a dense matrix"""
    if is_symmetric:
        if rows != columns:
            raise ValueError("rows and columns must have same size")
        idx_row = []
        idx_col = []
        for i_row in range(rows):
            for i_column in range(i_row, columns):  # Compute only upper triangle
                idx_row.append(i_row)
                idx_col.append(i_column)
        idx_row = np.array(idx_row, dtype=np.int_)
        idx_col = np.array(idx_col, dtype=np.int_)
    else:
        # creates a repeating range like [0,1,2,0,1,2]
        idx_row = np.tile(np.arange(rows), columns)
        # creates repeating numers in range like [0, 0, 1, 1, 2, 2]
        idx_col = np.repeat(np.arange(columns), rows)
    return idx_row, idx_col
