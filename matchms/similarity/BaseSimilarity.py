from abc import abstractmethod
from typing import List
import numpy as np
from sparsestack import StackedSparseArray
from matchms.typing import SpectrumType


class BaseSimilarity:
    """Similarity function base class.
    When building a custom similarity measure, inherit from this class and implement
    the desired methods.

    Attributes
    ----------
    is_commutative
       Whether similarity function is commutative, which means that the order of spectrums
       does not matter (similarity(A, B) == similarity(B, A)). Default is True.
    """
    # Set key characteristics as class attributes
    is_commutative = True
    # Set output data type, e.g. "float" or [("score", "float"), ("matches", "int")]
    score_datatype = np.float64

    @abstractmethod
    def pair(self, reference: SpectrumType, query: SpectrumType) -> float:
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

    def matrix(self, references: List[SpectrumType], queries: List[SpectrumType],
               array_type: str = "numpy",
               is_symmetric: bool = False) -> np.ndarray:
        """Optional: Provide optimized method to calculate an np.array of similarity scores
        for given reference and query spectrums. If no method is added here, the following naive
        implementation (i.e. a double for-loop) is used.

        Parameters
        ----------
        references
            List of reference objects
        queries
            List of query objects
        array_type
            Specify the output array type. Can be "numpy" or "sparse".
            Default is "numpy" and will return a numpy array. "sparse" will return a COO-sparse array.
        is_symmetric
            Set to True when *references* and *queries* are identical (as for instance for an all-vs-all
            comparison). By using the fact that score[i,j] = score[j,i] the calculation will be about
            2x faster.
        """
        #pylint: disable=too-many-locals
        n_rows = len(references)
        n_cols = len(queries)
        idx_row = []
        idx_col = []
        scores = []
        for i_ref, reference in enumerate(references[:n_rows]):
            if is_symmetric and self.is_commutative:
                for i_query, query in enumerate(queries[i_ref:n_cols], start=i_ref):
                    score = self.pair(reference, query)
                    if self.keep_score(score):
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

        idx_row = np.array(idx_row)
        idx_col = np.array(idx_col)
        scores_data = np.array(scores, dtype=self.score_datatype)
        # TODO: make StackedSpareseArray the default and add fixed function to output different formats (with code below)
        if array_type == "numpy":
            scores_array = np.zeros(shape=(n_rows, n_cols), dtype=self.score_datatype)
            scores_array[idx_row, idx_col] = scores_data.reshape(-1)
            return scores_array
        if array_type == "sparse":
            scores_array = StackedSparseArray(n_rows, n_cols)
            scores_array.add_sparse_data(idx_row, idx_col, scores_data, "")
            return scores_array
        raise ValueError("array_type must be 'numpy' or 'sparse'.")

    def sparse_array(self, references: List[SpectrumType], queries: List[SpectrumType],
                     idx_row, idx_col, is_symmetric: bool = False):
        """Optional: Provide optimized method to calculate an sparse matrix of similarity scores.

        Compute similarity scores for pairs of reference and query spectrums as given by the indices
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
        for i, row in enumerate(idx_row):
            col = idx_col[i]
            scores[i] = self.pair(references[row], queries[col])
        return scores

    def keep_score(self, score):
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
