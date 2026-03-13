import os
import numpy as np
import pytest
from matchms import Pipeline
from matchms.importing.load_spectra import load_spectra
from matchms.Pipeline import create_workflow
from matchms.similarity import ModifiedCosineGreedy
from matchms.yaml_file_functions import load_workflow_from_yaml_file


module_root = os.path.join(os.path.dirname(__file__), "..")
spectra_file_msp = os.path.join(module_root, "tests", "testdata", "massbank_five_spectra.msp")


def _score_array(scores):
    """Return the numeric score field as dense numpy array."""
    if "score" in scores.score_fields:
        return scores["score"].to_array()
    return scores.to_array()


def test_pipeline_load_and_save_yaml(tmp_path):
    workflow = create_workflow(
        spectra_1_filters=[("select_by_mz", {"mz_from": 0, "mz_to": 1000})],
        spectra_2_filters=[("select_by_mz", {"mz_from": 0, "mz_to": 1000})],
        score_computations=[
            ["precursormzmatch", {"tolerance": 120.0}],
            ["modifiedcosinegreedy", {"tolerance": 10.0}],
        ],
    )

    pipeline = Pipeline(workflow)
    yaml_file = os.path.join(tmp_path, "test_pipeline.yaml")
    pipeline.save_as_yaml(yaml_file)
    assert os.path.exists(yaml_file)

    new_pipeline = Pipeline.from_yaml(yaml_file)
    assert isinstance(new_pipeline, Pipeline)


def test_pipeline_initial_check_missing_file():
    workflow = create_workflow()
    pipeline = Pipeline(workflow)
    with pytest.raises(AssertionError) as msg:
        pipeline.run("non_existing_file.msp")
    assert "not exist" in str(msg.value)


def test_pipeline_initial_check_unknown_step():
    workflow = create_workflow(score_computations=[["precursormzOOPSmatch", {"tolerance": 120.0}]])
    with pytest.raises(ValueError) as msg:
        Pipeline(workflow)
    assert "Unknown score computation" in str(msg.value)


def test_pipeline_symmetric():
    workflow = create_workflow(
        score_computations=[
            ["precursormzmatch", {"tolerance": 120.0}],
            ["modifiedcosinegreedy", {"tolerance": 10.0}],
        ]
    )
    pipeline = Pipeline(workflow)
    pipeline.run(spectra_file_msp)

    scores = _score_array(pipeline.scores)

    assert len(pipeline.spectra_1) == 5
    assert pipeline.spectra_2 is None
    assert pipeline.is_symmetric is True
    assert pipeline.scores is not None
    assert pipeline.scores.shape == (5, 5)

    expected = np.array([[1.0, 0.30384404], [0.30384404, 1.0]])
    assert np.allclose(scores[3:, 3:], expected)
    assert np.allclose(np.diag(scores), 1.0), "Diagonal should all be 1.0"


def test_pipeline_symmetric_modified_cosine_hungarian():
    workflow = create_workflow(
        score_computations=[
            ["precursormzmatch", {"tolerance": 120.0}],
            ["modifiedcosinehungarian", {"tolerance": 10.0}],
        ]
    )
    pipeline = Pipeline(workflow)
    pipeline.run(spectra_file_msp)

    scores = _score_array(pipeline.scores)

    assert len(pipeline.spectra_1) == 5
    assert pipeline.spectra_2 is None
    assert pipeline.is_symmetric is True
    assert pipeline.scores is not None
    assert pipeline.scores.shape == (5, 5)
    assert np.allclose(np.diag(scores), 1.0), "Diagonal should all be 1.0"


def test_pipeline_symmetric_filters():
    workflow = create_workflow(
        spectra_1_filters=[("select_by_relative_intensity", {"intensity_from": 0.05})],
        score_computations=[
            ["precursormzmatch", {"tolerance": 120.0}],
            ["modifiedcosinegreedy", {"tolerance": 10.0}],
        ],
    )
    pipeline = Pipeline(workflow)
    pipeline.run(spectra_file_msp)

    scores = _score_array(pipeline.scores)

    assert len(pipeline.spectra_1) == 5
    assert pipeline.spectra_2 is None
    assert pipeline.is_symmetric is True
    assert pipeline.scores is not None
    assert pipeline.scores.shape == (5, 5)
    assert np.allclose(np.diag(scores), 1.0), "Diagonal should all be 1.0"


def test_pipeline_symmetric_masking():
    workflow = create_workflow(
        score_computations=[
            ["precursormzmatch", {"tolerance": 120.0}],
            ["mask", {"field": "score", "value": True, "operation": "=="}],
            ["modifiedcosinegreedy", {"tolerance": 10.0}],
        ]
    )
    pipeline = Pipeline(workflow)
    pipeline.run(spectra_file_msp)

    scores = _score_array(pipeline.scores)

    assert len(pipeline.spectra_1) == 5
    assert pipeline.spectra_2 is None
    assert pipeline.is_symmetric is True
    assert pipeline.scores is not None
    assert pipeline.scores.shape == (5, 5)

    expected = np.array([[1.0, 0.30384404], [0.30384404, 1.0]])
    assert np.allclose(scores[3:, 3:], expected)
    assert np.allclose(np.diag(scores), 1.0), "Diagonal should all be 1.0"


def test_pipeline_symmetric_custom_score():
    workflow = create_workflow(
        score_computations=[
            ["precursormzmatch", {"tolerance": 120.0}],
            [ModifiedCosineGreedy, {"tolerance": 10.0}],
        ]
    )
    pipeline = Pipeline(workflow)
    pipeline.run(spectra_file_msp)

    scores = _score_array(pipeline.scores)

    assert len(pipeline.spectra_1) == 5
    assert pipeline.spectra_2 is None
    assert pipeline.is_symmetric is True
    assert pipeline.scores is not None
    assert pipeline.scores.shape == (5, 5)

    expected = np.array([[1.0, 0.30384404], [0.30384404, 1.0]])
    assert np.allclose(scores[3:, 3:], expected)
    assert np.allclose(np.diag(scores), 1.0), "Diagonal should all be 1.0"


def test_pipeline_non_symmetric():
    workflow = create_workflow(
        score_computations=[
            ["precursormzmatch", {"tolerance": 120.0}],
            ["modifiedcosinegreedy", {"tolerance": 10.0}],
        ]
    )
    pipeline = Pipeline(workflow)
    pipeline.run(spectra_file_msp, [spectra_file_msp, spectra_file_msp])

    scores = _score_array(pipeline.scores)

    assert len(pipeline.spectra_1) == 5
    assert len(pipeline.spectra_2) == 10
    assert pipeline.spectra_1[0] == pipeline.spectra_2[5]
    assert pipeline.is_symmetric is False
    assert pipeline.scores is not None
    assert pipeline.scores.shape == (5, 10)

    expected = np.array([[1.0, 0.30384404], [0.30384404, 1.0]])
    assert np.allclose(scores[3:, 8:], expected)


def test_pipeline_from_yaml():
    pytest.importorskip("rdkit")
    config_file = os.path.join(module_root, "tests", "test_pipeline.yaml")
    workflow = load_workflow_from_yaml_file(config_file)

    pipeline = Pipeline(workflow)
    pipeline.run(spectra_file_msp)

    scores = _score_array(pipeline.scores)

    assert len(pipeline.spectra_1) == 5
    assert pipeline.spectra_2 is None
    assert pipeline.is_symmetric is True
    assert pipeline.scores is not None
    assert pipeline.scores.shape == (5, 5)

    expected = np.array([[1.0, 0.30384404], [0.30384404, 1.0]])
    assert np.allclose(scores[3:, 3:], expected)
    assert np.allclose(np.diag(scores), 1.0), "Diagonal should all be 1.0"


def test_pipeline_to_and_from_yaml(tmp_path):
    pytest.importorskip("rdkit")
    config_file = os.path.join(tmp_path, "test_pipeline.yaml")

    workflow = create_workflow(
        config_file,
        score_computations=[
            ["precursormzmatch", {"tolerance": 120.0}],
            ["modifiedcosinegreedy", {"tolerance": 10.0}],
        ],
    )
    assert os.path.exists(config_file)

    pipeline = Pipeline(workflow)
    pipeline.run(spectra_file_msp)
    scores_run1 = _score_array(pipeline.scores)

    workflow = load_workflow_from_yaml_file(config_file)
    pipeline = Pipeline(workflow)
    pipeline.run(spectra_file_msp)

    assert np.allclose(_score_array(pipeline.scores), scores_run1)


def test_pipeline_logging(tmp_path):
    pytest.importorskip("rdkit")
    workflow = create_workflow()
    pipeline = Pipeline(workflow)

    logfile = os.path.join(tmp_path, "testlog.log")
    pipeline.logging_file = logfile
    pipeline.run(spectra_file_msp)

    assert os.path.exists(logfile)
    with open(logfile, "r", encoding="utf-8") as f:
        content = f.read()
    assert "Start running matchms pipeline" in content


def test_pipeline_relative_intensity_filter():
    workflow = create_workflow(
        spectra_1_filters=[("select_by_relative_intensity", {"intensity_from": 0.05})],
        spectra_2_filters=[("select_by_relative_intensity", {"intensity_from": 0.05})],
        score_computations=[
            ["precursormzmatch", {"tolerance": 50}],
            ["modifiedcosinegreedy", {"tolerance": 10.0}],
        ],
    )
    pipeline = Pipeline(workflow)
    pipeline.run(spectra_file_msp, spectra_file_msp)

    assert pipeline.scores is not None
    assert pipeline.scores.shape == (5, 5)


def test_pipeline_changing_workflow():
    workflow = create_workflow(
        spectra_1_filters=["make_charge_int"],
        spectra_2_filters=["make_charge_int"],
        score_computations=["precursormzmatch"],
    )
    pipeline = Pipeline(workflow)

    pipeline.spectra_1_filters = [("select_by_relative_intensity", {"intensity_from": 0.05})]
    pipeline.spectra_2_filters = [("select_by_relative_intensity", {"intensity_from": 0.05})]
    pipeline.score_computations = [["modifiedcosinegreedy", {"tolerance": 10.0}]]

    pipeline.run(spectra_file_msp, spectra_file_msp)

    scores = _score_array(pipeline.scores)

    assert pipeline.scores is not None
    assert pipeline.scores.shape == (5, 5)
    assert np.allclose(np.diag(scores), 1.0)


def test_save_spectra_spectrum_processor(tmp_path):
    workflow = create_workflow()
    pipeline = Pipeline(workflow)
    filename = os.path.join(tmp_path, "spectra.mgf")

    pipeline.run(spectra_file_msp, cleaned_spectra_1_file=str(filename))
    assert os.path.exists(filename)

    reloaded_spectra = list(load_spectra(str(filename)))
    assert len(reloaded_spectra) == len(list(load_spectra(spectra_file_msp)))


def test_add_custom_filter():
    def select_spectra_containing_fragment(spectrum_in, fragment_of_interest=103.05, tolerance=0.01):
        for fragment_mz in spectrum_in.peaks.mz:
            if fragment_of_interest - tolerance < fragment_mz < fragment_of_interest + tolerance:
                return spectrum_in
        return None

    workflow = create_workflow(spectra_1_filters=[])
    pipeline = Pipeline(workflow)
    pipeline.processing_spectra_1.parse_and_add_filter(
        (select_spectra_containing_fragment, {"fragment_of_interest": 103.05, "tolerance": 0.01})
    )
    pipeline.run(spectra_file_msp)

    cleaned_spectra = pipeline.spectra_1
    assert len(cleaned_spectra) == 0


def test_add_custom_filter_to_spectra_1_filters():
    def select_spectra_containing_fragment(spectrum_in, fragment_of_interest=103.05, tolerance=0.01):
        for fragment_mz in spectrum_in.peaks.mz:
            if fragment_of_interest - tolerance < fragment_mz < fragment_of_interest + tolerance:
                return spectrum_in
        return None

    workflow = create_workflow(spectra_1_filters=[])
    pipeline = Pipeline(workflow)
    pipeline.processing_spectra_1.parse_and_add_filter(
        (select_spectra_containing_fragment, {"fragment_of_interest": 103.05, "tolerance": 0.01})
    )
    pipeline.run(spectra_file_msp)

    cleaned_spectra = pipeline.spectra_1
    assert len(cleaned_spectra) == 0
