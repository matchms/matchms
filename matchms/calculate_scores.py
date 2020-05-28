from .Scores import Scores
from .typing import SimilarityFunction, ReferencesType, QueriesType


def calculate_scores(references: ReferencesType,
                     queries: QueriesType,
                     similarity_function: SimilarityFunction) -> Scores:
    """Calculate the similarity between all reference objects versus all query objects.

    Parameters
    ----------
    references
        List of reference objects
    queries
        List of query objects
    similarity_function
        Function which accepts a reference + query object and returns a score or tuple of scores

    Returns
    -------

    Scores
    """

    return Scores(references=references,
                  queries=queries,
                  similarity_function=similarity_function).calculate()
