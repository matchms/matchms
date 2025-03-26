import os
import tempfile
import numpy as np
import pytest
import matchms.similarity
from matchms import create_scores_object_and_calculate_scores
from matchms.filtering import add_fingerprint
from matchms.importing import scores_from_json, scores_from_pickle
from .builder_Spectrum import SpectrumBuilder


pytest.importorskip("rdkit")


@pytest.fixture(params=["json", "pkl"])
def file_format(request):
    yield request.param


@pytest.fixture
def filename(file_format):
    with tempfile.TemporaryDirectory() as tmpdir:
        yield os.path.join(tmpdir, f"test_scores.{file_format}")


@pytest.fixture(params=matchms.similarity.__all__)
def similarity_function(request):
    kwargs = {}

    if request.param == "MetadataMatch":
        kwargs["field"] = "id"

    yield matchms.similarity.get_similarity_function_by_name(request.param)(**kwargs)


@pytest.fixture
def spectra(similarity_function):
    builder = SpectrumBuilder()
    spectrum_1 = builder.with_mz(np.array([100, 150, 200.])) \
        .with_intensities(np.array([0.7, 0.2, 0.1])) \
        .with_metadata({'id': 'spectrum1', "precursor_mz": 210, "parent_mass": 210, "smiles": "CCC(C)C(C(=O)O)NC(=O)CCl"}) \
        .build()
    spectrum_2 = builder.with_mz(np.array([100, 140, 190.])) \
        .with_intensities(np.array([0.4, 0.2, 0.1])) \
        .with_metadata({'id': 'spectrum2', "precursor_mz": 200, "parent_mass": 200, "smiles": "CCC(C)C(C(=O)O)NC(=O)CCl"}) \
        .build()
    spectrum_3 = builder.with_mz(np.array([110, 140, 195.])) \
        .with_intensities(
        np.array([0.6, 0.2, 0.1])) \
        .with_metadata({'id': 'spectrum3', "precursor_mz": 205, "parent_mass": 205, "smiles": "C(C(=O)O)(NC(=O)O)S"}) \
        .build()
    spectrum_4 = builder.with_mz(np.array([100, 150, 200.])) \
        .with_intensities(
        np.array([0.6, 0.1, 0.6])) \
        .with_metadata({'id': 'spectrum4', "precursor_mz": 210, "parent_mass": 210, "smiles": "C(C(=O)O)(NC(=O)O)S"}) \
        .build()

    spectra = [spectrum_1, spectrum_2, spectrum_3, spectrum_4]
    if similarity_function.__class__.__name__ == "FingerprintSimilarity":
        yield [add_fingerprint(spectrum, nbits=256) for spectrum in spectra]
    else:
        yield spectra


@pytest.fixture
def symmetrical_scores(similarity_function, spectra):
    """Return symmetrical scores for each similarity metric that matchms.similarity module exposes."""
    queries = spectra
    references = spectra

    scores = create_scores_object_and_calculate_scores(queries, references, similarity_function=similarity_function)
    yield scores


@pytest.fixture
def asymmetrical_scores(similarity_function, spectra):
    """Return asymmetrical scores for each similarity metric that matchms.similarity module exposes."""
    queries = spectra
    references = spectra[1:3]

    scores = create_scores_object_and_calculate_scores(queries, references, similarity_function=similarity_function)
    yield scores


def test_writing_scores_to_json(symmetrical_scores):
    """Test if writing scores to file works."""
    with tempfile.TemporaryDirectory() as tmpdir:
        filename = os.path.join(tmpdir, "test_scores.json")
        symmetrical_scores.to_json(filename)
        assert os.path.isfile(filename)


def test_writing_scores_to_pickle(symmetrical_scores):
    """Test if writing scores to file works."""
    with tempfile.TemporaryDirectory() as tmpdir:
        filename = os.path.join(tmpdir, "test_scores.pkl")
        symmetrical_scores.to_pickle(filename)
        assert os.path.isfile(filename)


def test_scores_write_read_symmetrical(filename, file_format, symmetrical_scores):
    """Test if writing and reading symmetrical scores does not change the scores."""
    if file_format == "json":
        symmetrical_scores.to_json(filename)
        scores = scores_from_json(filename)
    elif file_format == "pkl":
        symmetrical_scores.to_pickle(filename)
        scores = scores_from_pickle(filename)
    else:
        raise ValueError(f"Unsupported file format: {file_format}. Check 'file_format' fixture.")

    assert symmetrical_scores == scores


def test_scores_write_read_asymmetrical(filename, file_format, asymmetrical_scores):
    """Test if writing and reading symmetrical scores does not change the scores."""
    if file_format == "json":
        asymmetrical_scores.to_json(filename)
        scores = scores_from_json(filename)
    elif file_format == "pkl":
        asymmetrical_scores.to_pickle(filename)
        scores = scores_from_pickle(filename)
    else:
        raise ValueError(f"Unsupported file format: {file_format}. Check 'file_format' fixture.")

    assert asymmetrical_scores == scores
