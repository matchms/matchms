import os
from matchms.importing import load_from_mgf
from matchms.filtering import default_filters
from matchms.similarity import IntersectMz
from matchms import calculate_scores


def test_user_workflow():

    module_root = os.path.join(os.path.dirname(__file__), '..')
    references_file = os.path.join(module_root, 'tests', 'pesticides.mgf')

    reference_spectrums = [default_filters(s) for s in load_from_mgf(references_file)]

    query_spectrum = reference_spectrums[0].clone()

    # define similarity functions
    similarity_functions = [IntersectMz("intersect")]

    # calculate_scores
    scores = calculate_scores(query_spectrum,
                              reference_spectrums,
                              similarity_functions).sort_by("intersect").reverse().top(10)

    assert scores.scores[0] == 1., "Intersection of spectrum with itself should yield a perfect match."
    assert scores.scores.shape == (10, 1), "Expected a table of 10 rows, 1 columns."
    assert scores.scores[0] > scores.scores[1], "Expected a different sort order."


if __name__ == '__main__':
    test_user_workflow()
