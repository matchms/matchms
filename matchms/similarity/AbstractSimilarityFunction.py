from typing import List
import numpy
from matchms.typing import SpectrumType


class BaseSimilarityFunction:
    """Similarity function base class."""
    # Set key characteristics as class attributes
    is_commutative = True


class SequentialSimilarityFunction(BaseSimilarityFunction):
    """When building a custom similarity measure, inherit from this class and implement
    the desired methods to add a basic sequential way of computing a similarity.

    Code example for building a new similarity measure that includes a parallel method:

    .. code-block:: python

        import numpy
        from matchms.similarity.AbstractSimilarityFunction import SequentialSimilarityFunction


        class DummySimilarityFunctionNew(SequentialSimilarityFunction):

        def __init__(self):
            # constructor

        def compute_scores(self, reference: SpectrumType, query: SpectrumType) -> float:
            # Basic method, always provided for similarity scores
            s = len(reference.peaks) + len(query.peaks)
            return s

    """
    def compute_scores(self, reference: SpectrumType, query: SpectrumType) -> float:
        raise NotImplementedError


class ParallelSimilarityFunction(BaseSimilarityFunction):
    """When building a custom similarity measure, inherit from this class and implement
    the desired methods to add an optimized/parallel implementation of computing a similarity.
    A simple naive double for loop implementation is already implemented in
    :meth:`~matchms.Scores` and hence should not be added here.

    Code example for building a new similarity measure that provides both a sequential and a
    parallel method:

    .. code-block:: python

        import numpy
        from matchms.similarity.AbstractSimilarityFunction import SequentialSimilarityFunction
        from matchms.similarity.AbstractSimilarityFunction import ParallelSimilarityFunction


        class DummySimilarityFunctionNew(SequentialSimilarityFunction, ParallelSimilarityFunction):

        def __init__(self):
            # constructor

        def compute_scores(self, reference: SpectrumType, query: SpectrumType) -> float:
            # Basic method, always provided for similarity scores
            s = len(reference.peaks) + len(query.peaks)
            return s

        def compute_scores_parallel(self, reference: List[SpectrumType],
                                    query: List[SpectrumType]) -> numpy.ndarray:
            # Optimize/parallel method, only provided for some similarity scores.
            lens1 = numpy.array([len(x.peaks) for x in reference]).reshape(1,-1)
            lens2 = numpy.array([len(x.peaks) for x in query]).reshape(1,-1)
            return lens1 * lens2.T

    """
    def compute_scores_parallel(self, reference: List[SpectrumType], query: List[SpectrumType],
                                is_symmetric: bool = None) -> numpy.ndarray:
        raise NotImplementedError
