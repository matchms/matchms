import os
import pytest
from matchms.importing import load_from_mgf
from matchms.filtering import default_filters, add_parent_mass, normalize_intensities, require_minimum_number_of_peaks
from matchms.filtering import select_by_relative_intensity, select_by_mz
from matchms.similarity import CosineGreedy
from matchms import calculate_scores


def test_user_workflow():

    def evaluate_assert_set_1():
        assert scores.scores[0][0][0] == pytest.approx(1, 1e-6), \
            "Comparison of spectrum with itself should yield a perfect match."

        assert scores.scores.shape == (76, 76), \
            "Expected a table of 76 rows, 76 columns."

        r, q, _, _ = scores.__next__()
        scores.reset_iterator()
        assert r == references[0]
        assert q == queries[0]

    def evaluate_assert_set_2():
        # filter out self-comparisons, require at least 20 matching peaks:
        filtered = [elem for elem in scores if elem[0] != elem[1] and elem[3] > 20]
        # sort by score
        sorted_by_score = sorted(filtered, key=lambda elem: elem[2], reverse=True)

        assert sorted_by_score[0][0] != sorted_by_score[0][1], "Self-comparisons should have been filtered out."
        assert sorted_by_score[0][2] >= sorted_by_score[1][2], "Expected scores to be in order of decreasing score."
        assert sorted_by_score[0][3] > 20, "Expected number of matches to be larger than 20."
        assert sorted_by_score[0][0] == sorted_by_score[1][1], "In this symmetrical analysis, the top 2 best results " \
                                                               "should be between the same objects."

    def apply_my_filters(s):
        s = default_filters(s)
        s = require_minimum_number_of_peaks(s, n_required=5)
        s = add_parent_mass(s)
        s = normalize_intensities(s)
        s = select_by_relative_intensity(s, intensity_from=0.0, intensity_to=1.0)
        s = select_by_mz(s, mz_from=0, mz_to=1000)
        return s

    module_root = os.path.join(os.path.dirname(__file__), '..')
    spectrums_file = os.path.join(module_root, 'tests', 'pesticides.mgf')

    # apply my filters to the data
    spectrums = [apply_my_filters(s) for s in load_from_mgf(spectrums_file)]

    # omit spectrums that didn't qualify for analysis
    spectrums = [s for s in spectrums if s is not None]

    # this will be a library grouping analysis, so queries = references = spectrums
    queries = spectrums[:]
    references = spectrums[:]

    # define similarity function
    cosine_greedy = CosineGreedy()

    # calculate_scores
    scores = calculate_scores(references,
                              queries,
                              cosine_greedy)

    evaluate_assert_set_1()
    evaluate_assert_set_2()
