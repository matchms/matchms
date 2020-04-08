import os
from matchms.importing import load_from_mgf
from matchms.filtering import filterfun1
from matchms.similarity import SimMeas1, IntersectMz
from matchms import calculate_scores


def test_user_workflow():

    def apply_filters(s):
        filterfun1(s)

    module_root = os.path.join(os.path.dirname(__file__), '..')
    references_file = os.path.join(module_root, 'tests', 'pesticides.mgf')

    reference_spectrums_raw = load_from_mgf(references_file)
    reference_spectrums = [s.clone() for s in reference_spectrums_raw]

    query_spectrum_raw = reference_spectrums_raw[0]
    query_spectrum = query_spectrum_raw.clone()

    # filtering
    for s in reference_spectrums:
        apply_filters(s)

    apply_filters(query_spectrum)

    # define similarity functions
    similarity_functions = [SimMeas1("simmeas1", factor=0.1), IntersectMz("intersect")]

    # calculate_scores
    scores = calculate_scores(query_spectrum,
                              reference_spectrums,
                              similarity_functions).sort_by("intersect").reverse().top(10)

    assert scores.scores[0, 1] == 1, "Intersection of spectrum with itself should yield a perfect match."
    assert scores.scores.shape == (10, 2), "Expected a table of 10 rows, 2 columns."
    assert scores.scores[0, 1] > scores.scores[1, 1], "Expected a different sort order."


if __name__ == '__main__':
    test_user_workflow()
