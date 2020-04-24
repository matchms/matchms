from matchms import Scores
import numpy


def test_scores_init():

    scores = Scores(references=numpy.asarray(["r0", "r1", "r2"]),
                    queries=numpy.asarray(["q0", "q1"]),
                    similarity_function=None)

    assert scores.scores.shape == (3, 2)


def test_scores_next():

    class DummySimilarityFunction:

        def __init__(self):
            """constructor"""

        def __call__(self, reference, query):
            """call method"""

            s = reference + query

            return s, len(s)

    dummy_similarity_function = DummySimilarityFunction()

    scores = Scores(references=numpy.asarray(["r", "rr", "rrr"]),
                    queries=numpy.asarray(["q", "qq"]),
                    similarity_function=dummy_similarity_function).calculate()

    assert next(scores) == ("r", "q", "rq", 2)
    assert next(scores) == ("r", "qq", "rqq", 3)
    assert next(scores) == ("rr", "q", "rrq", 3)
    assert next(scores) == ("rr", "qq", "rrqq", 4)
    assert next(scores) == ("rrr", "q", "rrrq", 4)
    assert next(scores) == ("rrr", "qq", "rrrqq", 5)
