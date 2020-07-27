from __future__ import annotations
import numpy
from matchms.typing import QueriesType
from matchms.typing import ReferencesType
from matchms.typing import SimilarityFunction


class Scores:
    """Contains reference and query spectrums and the scores between them.

    The scores can be retrieved as a matrix with the :py:attr:`Scores.scores` attribute.
    The reference spectrum, query spectrum, score pairs can also be iterated over in query then reference order.

    Example to calculate scores between 2 spectrums and iterate over the scores

    .. testcode::

        import numpy as np
        from matchms import Scores, Spectrum
        from matchms.similarity import CosineGreedy

        spectrum_1 = Spectrum(mz=np.array([100, 150, 200.]),
                              intensities=np.array([0.7, 0.2, 0.1]),
                              metadata={'id': 'spectrum1'})
        spectrum_2 = Spectrum(mz=np.array([100, 140, 190.]),
                              intensities=np.array([0.4, 0.2, 0.1]),
                              metadata={'id': 'spectrum2'})
        spectrum_3 = Spectrum(mz=np.array([110, 140, 195.]),
                              intensities=np.array([0.6, 0.2, 0.1]),
                              metadata={'id': 'spectrum3'})
        spectrum_4 = Spectrum(mz=np.array([100, 150, 200.]),
                              intensities=np.array([0.6, 0.1, 0.6]),
                              metadata={'id': 'spectrum4'})
        references = [spectrum_1, spectrum_2]
        queries = [spectrum_3, spectrum_4]

        scores = Scores(references, queries, CosineGreedy()).calculate()

        for (reference, query, score, n_matching) in scores:
            print(f"Cosine score between {reference.get('id')} and {query.get('id')}" +
                  f" is {score:.2f} with {n_matching} matched peaks")

    Should output

    .. testoutput::

        Cosine score between spectrum1 and spectrum3 is 0.00 with 0 matched peaks
        Cosine score between spectrum1 and spectrum4 is 0.80 with 3 matched peaks
        Cosine score between spectrum2 and spectrum3 is 0.14 with 1 matched peaks
        Cosine score between spectrum2 and spectrum4 is 0.61 with 1 matched peaks
    """
    def __init__(self, references: ReferencesType, queries: QueriesType, similarity_function: SimilarityFunction):
        """

        Parameters
        ----------
        references
            List of reference objects
        queries
            List of query objects
        similarity_function
            Function which accepts a reference + query object and returns a score or tuple of scores

        """
        Scores._validate_input_arguments(references, queries, similarity_function)

        self.n_rows = len(references)
        self.n_cols = len(queries)
        self.references = numpy.asarray(references).reshape(self.n_rows, 1)
        self.queries = numpy.asarray(queries).reshape(1, self.n_cols)
        self.similarity_function = similarity_function
        self._scores = numpy.empty([self.n_rows, self.n_cols], dtype="object")
        self._index = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self._index < self.scores.size:
            # pylint: disable=unbalanced-tuple-unpacking
            r, c = numpy.unravel_index(self._index, self._scores.shape)
            self._index += 1
            result = self._scores[r, c]
            if not isinstance(result, tuple):
                result = (result,)
            return (self.references[r, 0], self.queries[0, c]) + result
        self._index = 0
        raise StopIteration

    def __str__(self):
        return self._scores.__str__()

    @staticmethod
    def _validate_input_arguments(references, queries, similarity_function):
        assert isinstance(references, (list, tuple, numpy.ndarray)),\
            "Expected input argument 'references' to be list or tuple or numpy.ndarray."

        assert isinstance(queries, (list, tuple, numpy.ndarray)),\
            "Expected input argument 'queries' to be list or tuple or numpy.ndarray."

        assert callable(similarity_function), "Expected input argument 'similarity_function' to be callable."

    def calculate(self) -> Scores:
        """
        Calculate the similarity between all reference objects v all query objects using a
        naive implementation (i.e. a double for-loop). Similarity functions should expect
        one reference and one query object as its input arguments.
        """
        for i_ref, reference in enumerate(self.references[:self.n_rows, 0]):
            for i_query, query in enumerate(self.queries[0, :self.n_cols]):
                self._scores[i_ref][i_query] = self.similarity_function(reference, query)
        return self

    def calculate_parallel(self) -> Scores:
        """
        Calculate the similarity between all reference objects v all query objects using a
        vectorized implementation.  Similarity functions should expect a Numpy array of
        all reference objects and a Numpy array of all query objects as its input arguments.
        """

        self._scores = self.similarity_function(self.references[:, 0], self.queries[0, :])
        return self

    @property
    def scores(self) -> numpy.ndarray:
        """Scores as numpy array

        For example

        .. testcode::

            import numpy as np
            from matchms import Scores, Spectrum
            from matchms.similarity import IntersectMz

            spectrum_1 = Spectrum(mz=np.array([100, 150, 200.]),
                                  intensities=np.array([0.7, 0.2, 0.1]))
            spectrum_2 = Spectrum(mz=np.array([100, 140, 190.]),
                                  intensities=np.array([0.4, 0.2, 0.1]))
            spectrums = [spectrum_1, spectrum_2]

            scores = Scores(spectrums, spectrums, IntersectMz()).calculate().scores

            print(score.dtype)
            print(scores.shape)
            print(scores)

        Should output

        .. testoutput::

             float64
             (2, 2)
             [[1.0 0.2]
              [0.2 1.0]]
        """
        return self._scores.copy()
