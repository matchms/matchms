from .Scores import Scores


def calculate_scores(queries, references, similarity_function):
    """An example docstring for a unbound function."""

    return Scores(queries=queries,
                  references=references,
                  similarity_function=similarity_function).calculate()
