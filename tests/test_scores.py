from matchms import Scores


class DummySimilarityFunction:
    def __init__(self):
        """constructor"""

    def __call__(self, reference, query):
        """call method"""
        s = reference + query
        return s, len(s)


def test_scores_init():

    dummy_similarity_function = DummySimilarityFunction()
    scores = Scores(references=["r0", "r1", "r2"],
                    queries=["q0", "q1"],
                    similarity_function=dummy_similarity_function)
    assert scores.scores.shape == (3, 2)


def test_scores_next():

    dummy_similarity_function = DummySimilarityFunction()
    scores = Scores(references=["r", "rr", "rrr"],
                    queries=["q", "qq"],
                    similarity_function=dummy_similarity_function).calculate()

    actual = list(scores)
    expected = [
        ("r", "q", "rq", 2),
        ("r", "qq", "rqq", 3),
        ("rr", "q", "rrq", 3),
        ("rr", "qq", "rrqq", 4),
        ("rrr", "q", "rrrq", 4),
        ("rrr", "qq", "rrrqq", 5)
    ]
    assert actual == expected
