from typing import List, Callable
from .Scores import Scores


def calculate_scores(queries: List[object], references: List[object], similarity_function: Callable):
    """An example docstring for a unbound function."""

    return Scores(queries=queries,
                  references=references,
                  similarity_function=similarity_function).calculate()
