from .Scores import Scores


def calculate_scores(references, queries, similarity_function):

    return Scores(references=references,
                  queries=queries,
                  similarity_function=similarity_function).calculate()
