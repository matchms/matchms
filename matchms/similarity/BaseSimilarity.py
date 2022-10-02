from abc import abstractmethod
from typing import List
import numpy as np
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
        is_symmetric
            Set to True when *references* and *queries* are identical (as for instance for an all-vs-all
            comparison). By using the fact that score[i,j] = score[j,i] the calculation will be about
            2x faster.
        """
        n_rows = len(references)
        n_cols = len(queries)
        scores = np.empty([n_rows, n_cols], dtype=self.score_datatype)
        for i_ref, reference in enumerate(references[:n_rows]):
            if is_symmetric and self.is_commutative:
                for i_query, query in enumerate(queries[i_ref:n_cols], start=i_ref):
                    scores[i_ref][i_query] = self.pair(reference, query)
                    scores[i_query][i_ref] = scores[i_ref][i_query]
            else:
                for i_query, query in enumerate(queries[:n_cols]):
                    scores[i_ref][i_query] = self.pair(reference, query)
        return scores

    def sort(self, scores: np.ndarray):
        """Return array of indexes for sorted list of scores.
        This method can be adapted for different styles of scores.

        Parameters
        ----------
        scores
            1D Array of scores.

        Returns
        -------
        idx_sorted
            Indexes of sorted scores.
        """
        return scores.argsort()[::-1]

    def to_dict(self) -> dict:
        """Return a dictionary representation of a similarity function."""
        return {"__Similarity__": self.__class__.__name__,
                **self.__dict__}
