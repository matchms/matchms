from typing import Callable
from typing import List
from typing import Tuple
import numpy
from matchms.typing import QueriesType
from matchms.typing import ReferencesType


class Scores:
    """An example docstring for a class definition."""
    def __init__(self, references: ReferencesType, queries: QueriesType, similarity_function: Callable):
        """An example docstring for a constructor."""

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
        assert isinstance(references, (List, Tuple, numpy.ndarray)),\
            "Expected input argument 'references' to be list or tuple or numpy.ndarray."

        assert isinstance(queries, (List, Tuple, numpy.ndarray)),\
            "Expected input argument 'queries' to be list or tuple or numpy.ndarray."

        assert callable(similarity_function), "Expected input argument 'similarity_function' to be callable."

    def calculate(self):
        """
        Calculate the similarity between all reference objects v all query objects using a
        naive implementation (i.e. a double for-loop). Similarity functions should expect
        one reference and one query object as its input arguments.
        """
        for i_ref, reference in enumerate(self.references[:self.n_rows, 0]):
            for i_query, query in enumerate(self.queries[0, :self.n_cols]):
                self._scores[i_ref][i_query] = self.similarity_function(reference, query)
        return self

    def calculate_parallel(self):
        """
        Calculate the similarity between all reference objects v all query objects using a
        vectorized implementation.  Similarity functions should expect a Numpy array of
        all reference objects and a Numpy array of all query objects as its input arguments.
        """

        self._scores = self.similarity_function(self.references[:, 0], self.queries[0, :])
        return self

    @property
    def scores(self):
        """getter method for scores private variable"""
        return self._scores.copy()
