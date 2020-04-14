from matchms import Scores
import numpy


def test_scores_init():

    scores = Scores(queries=numpy.asarray(["q0", "q1"]),
                    references=numpy.asarray(["r0", "r1", "r2"]),
                    similarity_function=None)

    assert scores.scores.shape == (3, 2)


def test_scores_sort():

    scores = Scores(queries=numpy.asarray(["q0", "q1"]),
                    references=numpy.asarray(["r0", "r1", "r2"]),
                    similarity_function=None)

    scores.scores = numpy.asarray([[1, 2], [5, 4], [3, 6]], dtype="float")

    queries_sorted, references_sorted, scores_sorted = scores.sort()

    assert numpy.all(queries_sorted == numpy.asarray([["q1"], ["q0"], ["q1"], ["q0"], ["q1"], ["q0"]]))
    assert numpy.all(references_sorted == numpy.asarray([["r2"], ["r1"], ["r1"], ["r2"], ["r0"], ["r0"]]))
    assert numpy.all(scores_sorted == numpy.arange(1, 7)[::-1].reshape(6, 1))


def test_scores_top():

    scores = Scores(queries=numpy.asarray(["q0", "q1"]),
                    references=numpy.asarray(["r0", "q1", "r2"]),
                    similarity_function=None)

    scores.scores = numpy.asarray([[1, 2], [5, 4], [3, 6]], dtype="float")

    queries_top_2, references_top_2, scores_top_2 = scores.top(2, omit_self_comparisons=True)

    assert numpy.all(queries_top_2 == numpy.asarray([["q1"], ["q0"]]))
    assert numpy.all(references_top_2 == numpy.asarray([["r2"], ["q1"]]))
    assert numpy.all(scores_top_2 == numpy.asarray([[6.], [5.]], dtype="float"))


if __name__ == '__main__':
    test_scores_init()
    test_scores_sort()
    test_scores_top()
