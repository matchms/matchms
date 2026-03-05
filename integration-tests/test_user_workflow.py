import os
import numpy as np
import pytest
from matchms import Pipeline
from matchms.Pipeline import create_workflow
from matchms.similarity import CosineGreedy


def test_user_workflow():
    module_root = os.path.join(os.path.dirname(__file__), "..")
    workflow = create_workflow(query_filters=[["add_parent_mass"],
                                              ["normalize_intensities"],
                                              ["select_by_relative_intensity",
                                               {"intensity_from": 0.0, "intensity_to": 1.0}],
                                              ["select_by_mz", {"mz_from": 0, "mz_to": 1000}],
                                              ["require_minimum_number_of_peaks", {"n_required": 5}]],
                               score_computations=[["cosinegreedy", {"tolerance": 0.3}]])
    pipeline = Pipeline(workflow)
    spectra_file = os.path.join(module_root, "tests", "testdata", "pesticides.mgf")
    pipeline.run(spectra_file)