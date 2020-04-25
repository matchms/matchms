import os
import pytest
from matchms.importing import load_from_mgf
from matchms.filtering import default_filters, add_parent_mass, normalize_intensities, require_minimum_number_of_peaks
from matchms.filtering import select_by_relative_intensity, select_by_mz
from matchms.similarity import CosineGreedy
from matchms import calculate_scores


def test_user_workflow():

    def apply_my_filters(s):
        s = default_filters(s)
        s = add_parent_mass(s)
        s = normalize_intensities(s)
        s = select_by_relative_intensity(s, intensity_from=0.0, intensity_to=1.0)
        s = select_by_mz(s, mz_from=0, mz_to=1000)
        s = require_minimum_number_of_peaks(s, n_required=5)
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
    scores = calculate_scores(queries,
                              references,
                              cosine_greedy)

    queries_top10, reference_top10, scores_top10, = scores.top(10, include_self_comparisons=True)

    print(scores_top10)

    assert scores.scores[0][0] == pytest.approx(1, 1e-6), \
        "Comparison of spectrum with itself should yield a perfect match."

    assert scores.scores.shape == (76, 76), \
        "Expected a table of 76 rows, 76 columns."

    assert queries_top10[0][0] == reference_top10[0][0], \
        "Expected the best match between two copies of the same spectrum."
