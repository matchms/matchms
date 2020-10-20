import os
import pytest
from matchms import calculate_scores
from matchms.filtering import add_parent_mass
from matchms.filtering import default_filters
from matchms.filtering import normalize_intensities
from matchms.filtering import require_minimum_number_of_peaks
from matchms.filtering import select_by_mz
from matchms.filtering import select_by_relative_intensity
from matchms.importing import load_from_mgf
from matchms.similarity import CosineGreedy


def test_user_workflow():

    def apply_my_filters(s):
        s = default_filters(s)
        s = add_parent_mass(s)
        s = normalize_intensities(s)
        s = select_by_relative_intensity(s, intensity_from=0.0, intensity_to=1.0)
        s = select_by_mz(s, mz_from=0, mz_to=1000)
        s = require_minimum_number_of_peaks(s, n_required=5)
        return s

    module_root = os.path.join(os.path.dirname(__file__), "..")
    spectrums_file = os.path.join(module_root, "tests", "pesticides.mgf")

    # apply my filters to the data
    spectrums = [apply_my_filters(s) for s in load_from_mgf(spectrums_file)]

    # omit spectrums that didn't qualify for analysis
    spectrums = [s for s in spectrums if s is not None]

    # this will be a library grouping analysis, so queries = references = spectrums
    queries = spectrums[:]
    references = spectrums[:]

    # define similarity function
    cosine_greedy = CosineGreedy(tolerance=0.3)

    # calculate_scores
    scores = list(calculate_scores(references,
                                   queries,
                                   cosine_greedy))

    # filter out self-comparisons, require at least 20 matching peaks:
    filtered = [(reference, query, score) for (reference, query, score) in scores
                if reference != query and score["matches"] >= 20]

    sorted_by_score = sorted(filtered, key=lambda elem: elem[2]["score"], reverse=True)

    actual_top10 = sorted_by_score[:10]

    score_datatype = cosine_greedy.score_datatype
    expected_top10 = [
        (references[48], queries[50], numpy.array((0.9994783627790965, 25), dtype=score_datatype)),
        (references[50], queries[48], numpy.array((0.9994783627790965, 25), dtype=score_datatype)),
        (references[46], queries[48], numpy.array((0.9990141860269471, 27), dtype=score_datatype)),
        (references[48], queries[46], numpy.array((0.9990141860269471, 27), dtype=score_datatype)),
        (references[46], queries[50], numpy.array((0.9988793406908719, 22), dtype=score_datatype)),
        (references[50], queries[46], numpy.array((0.9988793406908719, 22), dtype=score_datatype)),
        (references[57], queries[59], numpy.array((0.9982171275552505, 46), dtype=score_datatype)),
        (references[59], queries[57], numpy.array((0.9982171275552505, 46), dtype=score_datatype)),
        (references[73], queries[74], numpy.array((0.9973823244169199, 23), dtype=score_datatype)),
        (references[74], queries[73], numpy.array((0.9973823244169199, 23), dtype=score_datatype)),
    ]
    assert actual_top10 == expected_top10
