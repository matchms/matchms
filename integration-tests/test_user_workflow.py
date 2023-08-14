import os
import numpy as np
import pytest
from matchms import Pipeline
from matchms.Pipeline import create_workflow
from matchms.similarity import CosineGreedy


def test_user_workflow():
    module_root = os.path.join(os.path.dirname(__file__), "..")
    workflow = create_workflow(predefined_processing_queries="basic",
                               additional_filters_queries=[["add_parent_mass"],
                                                           ["normalize_intensities"],
                                                           ["select_by_relative_intensity", {"intensity_from": 0.0, "intensity_to": 1.0}],
                                                           ["select_by_mz", {"mz_from": 0, "mz_to": 1000}],
                                                           ["require_minimum_number_of_peaks", {"n_required": 5}]],
                               score_computations=[["cosinegreedy",  {"tolerance": 0.3}]])
    pipeline = Pipeline(workflow)
    spectrums_file = os.path.join(module_root, "tests", "testdata", "pesticides.mgf")
    pipeline.run(spectrums_file)

    scores = pipeline.scores

    # filter out self-comparisons, require at least 20 matching peaks:
    filtered = [(reference, query, score) for (reference, query, score) in scores
                if reference != query and score[1] >= 20]

    sorted_by_score = sorted(filtered, key=lambda elem: elem[2][0], reverse=True)

    actual_top10 = sorted_by_score[:10]

    score_datatype = CosineGreedy().score_datatype
    expected_top10 = [
        (scores.references[48], scores.queries[50], np.array([(0.9994783627790967, 25)], dtype=score_datatype)[0]),
        (scores.references[50], scores.queries[48], np.array([(0.9994783627790967, 25)], dtype=score_datatype)[0]),
        (scores.references[46], scores.queries[48], np.array([(0.9990141860269471, 27)], dtype=score_datatype)[0]),
        (scores.references[48], scores.queries[46], np.array([(0.9990141860269471, 27)], dtype=score_datatype)[0]),
        (scores.references[46], scores.queries[50], np.array([(0.9988793406908721, 22)], dtype=score_datatype)[0]),
        (scores.references[50], scores.queries[46], np.array([(0.9988793406908721, 22)], dtype=score_datatype)[0]),
        (scores.references[57], scores.queries[59], np.array([(0.9982171275552503, 46)], dtype=score_datatype)[0]),
        (scores.references[59], scores.queries[57], np.array([(0.9982171275552503, 46)], dtype=score_datatype)[0]),
        (scores.references[73], scores.queries[74], np.array([(0.9973823244169199, 23)], dtype=score_datatype)[0]),
        (scores.references[74], scores.queries[73], np.array([(0.9973823244169199, 23)], dtype=score_datatype)[0]),
    ]
    assert [x[2][0] for x in actual_top10] == pytest.approx([x[2][0] for x in expected_top10], 1e-8), \
        "Expected different scores."
    assert [x[2][1] for x in actual_top10] == [x[2][1] for x in expected_top10], \
        "Expected different matches."
    assert [x[0] for x in actual_top10] == [x[0] for x in expected_top10], \
        "Expected different references."
    assert [x[1] for x in actual_top10] == [x[1] for x in expected_top10], \
        "Expected different queries."
