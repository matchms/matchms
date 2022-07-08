import os
import tempfile
import pytest

from .test_scores import spectra
from matchms import calculate_scores
from matchms.importing.load_scores import scores_from_json, scores_from_pickle
import matchms.similarity


@pytest.fixture(params=["json", "pkl"])
def file_format(request):
    yield request.param


@pytest.fixture()
def filename(file_format):
    with tempfile.TemporaryDirectory() as tmpdir:
        yield os.path.join(tmpdir, f"test_scores.{file_format}")


@pytest.fixture(params=matchms.similarity.__all__)
def symmetrical_scores(request):
    """Return symmetrical scores for each similarity metric that matchms.similarity module exposes."""
    spectrum_1, spectrum_2, spectrum_3, spectrum_4 = spectra()
    queries = [spectrum_1, spectrum_2, spectrum_3, spectrum_4]
    references = [spectrum_1, spectrum_2, spectrum_3, spectrum_4]

    scores = calculate_scores(queries, references, similarity_function=request.param)
    yield scores


@pytest.fixture(params=matchms.similarity.__all__)
def asymmetrical_scores(request):
    """Return asymmetrical scores for each similarity metric that matchms.similarity module exposes."""
    spectrum_1, spectrum_2, spectrum_3, spectrum_4 = spectra()
    queries = [spectrum_1, spectrum_2, spectrum_3, spectrum_4]
    references = [spectrum_2, spectrum_3]

    scores = calculate_scores(queries, references, similarity_function=request.param)
    yield scores


def test_writing_scores_to_file(filename, symmetrical_scores):
    """Test if writing scores to file works."""
    symmetrical_scores.export_to_json(filename)
    assert os.path.isfile(filename)


def test_scores_write_read_symmetrical(filename, symmetrical_scores):
    """Test if writing and reading symmetrical scores does not change the scores."""
    pass


def test_scores_write_read_asymmetrical(filename, asymmetrical_scores):
    """Test if writing and reading symmetrical scores does not change the scores."""
    pass
