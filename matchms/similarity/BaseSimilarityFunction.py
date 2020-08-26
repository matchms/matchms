from typing import List
import numpy
from matchms.typing import SpectrumType


class BaseSimilarityFunction:
    """Similarity function base class.
    When building a custom similarity measure, inherit from this class and implement
    the desired methods.
    """
    # Set key characteristics as class attributes
    is_commutative = True

    def compute_score(self, reference: SpectrumType, query: SpectrumType) -> float:
        """Required: Method to calculate the similarity for one input pair.

        Parameters
        ----------
        reference
            Single reference spectrum.
        query
            Single query spectrum.
        """
        raise NotImplementedError

    def compute_score_matrix(self, references: List[SpectrumType], queries: List[SpectrumType],
                             is_symmetric: bool = None) -> numpy.ndarray:
        """Optional: Provide optimized method to calculate an numpy.array of similarity scores
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
        scores = numpy.empty([n_rows, n_cols], dtype="object")
        for i_ref, reference in enumerate(references[:n_rows, 0]):
            if is_symmetric:
                for i_query, query in enumerate(queries[0, i_ref:n_cols], start=i_ref):
                    scores[i_ref][i_query] = self.compute_score(reference, query)
                    scores[i_query][i_ref] = scores[i_ref][i_query]
            else:
                for i_query, query in enumerate(queries[0, :n_cols]):
                    scores[i_ref][i_query] = self.compute_score(reference, query)
        return scores
