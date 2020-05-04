from typing import List, Callable
from .Scores import Scores


def calculate_scores(references: List[object], queries: List[object], similarity_function: Callable):

    return Scores(references=references,
                  queries=queries,
                  similarity_function=similarity_function).calculate()
