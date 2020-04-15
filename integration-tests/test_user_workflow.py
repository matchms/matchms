import os
from matchms.importing import load_from_mgf
from matchms.filtering import default_filters
from matchms.similarity import IntersectMz
from matchms import calculate_scores


def test_user_workflow():

    module_root = os.path.join(os.path.dirname(__file__), '..')
    spectrums_file = os.path.join(module_root, 'tests', 'pesticides.mgf')

    spectrums = [default_filters(s) for s in load_from_mgf(spectrums_file)]

    queries = spectrums[:7]
    references = spectrums[6:]

    # define similarity function
    intersect_mz = IntersectMz()

    # calculate_scores
    scores = calculate_scores(queries,
                              references,
                              intersect_mz)

    queries_top10, reference_top10, scores_top10, = scores.top(10, include_self_comparisons=True)

    print(scores_top10)

    assert scores.scores[0, 6] == 1., "Intersection of spectrum with itself should yield a perfect match."
    assert scores.scores.shape == (70, 7), "Expected a table of 10 rows, 1 columns."
    assert queries_top10[0] == reference_top10[0], "Expected the best match between two copies of the same spectrum."


if __name__ == '__main__':
    test_user_workflow()
