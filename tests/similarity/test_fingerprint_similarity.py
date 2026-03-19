import numpy as np
import pytest
from rdkit.Chem import rdFingerprintGenerator
from matchms import Spectrum
from matchms.Fingerprints import Fingerprints
from matchms.Scores import Scores
from matchms.similarity import FingerprintSimilarity


pytest.importorskip("chemap")


@pytest.fixture
def fingerprint_generator():
    return rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=256)


@pytest.fixture
def spectra():
    spectrum_1 = Spectrum(
        mz=np.array([], dtype=float),
        intensities=np.array([], dtype=float),
        metadata={
            "inchikey": "OTMSDBZUPAUEDD-UHFFFAOYSA-N",
            "smiles": "CC",
        },
    )
    spectrum_2 = Spectrum(
        mz=np.array([], dtype=float),
        intensities=np.array([], dtype=float),
        metadata={
            "inchikey": "UGFAIRIUMAVXCW-UHFFFAOYSA-N",
            "smiles": "[C-]#[O+]",
        },
    )
    spectrum_3 = Spectrum(
        mz=np.array([], dtype=float),
        intensities=np.array([], dtype=float),
        metadata={
            "inchikey": "LFQSCWFLJHTTHZ-UHFFFAOYSA-N",
            "smiles": "CCC",
        },
    )
    return [spectrum_1, spectrum_2, spectrum_3]


@pytest.fixture
def spectra_with_invalid():
    spectrum_0 = Spectrum(
        mz=np.array([], dtype=float),
        intensities=np.array([], dtype=float),
        metadata={},
    )
    spectrum_1 = Spectrum(
        mz=np.array([], dtype=float),
        intensities=np.array([], dtype=float),
        metadata={
            "inchikey": "OTMSDBZUPAUEDD-UHFFFAOYSA-N",
            "smiles": "CC",
        },
    )
    spectrum_2 = Spectrum(
        mz=np.array([], dtype=float),
        intensities=np.array([], dtype=float),
        metadata={
            "inchikey": "UGFAIRIUMAVXCW-UHFFFAOYSA-N",
            "smiles": "[C-]#[O+]",
        },
    )
    return [spectrum_0, spectrum_1, spectrum_2]


@pytest.fixture
def dense_fingerprints(spectra, fingerprint_generator):
    fps = Fingerprints(
        fingerprint_generator=fingerprint_generator,
        return_csr=False,
    )
    fps.compute_fingerprints(spectra)
    return fps


@pytest.fixture
def sparse_fingerprints(spectra, fingerprint_generator):
    fps = Fingerprints(
        fingerprint_generator=fingerprint_generator,
        return_csr=True,
    )
    fps.compute_fingerprints(spectra)
    return fps


@pytest.mark.parametrize("similarity_measure", ["cosine", "tanimoto"])
def test_fingerprint_similarity_pair_not_supported(fingerprint_generator, similarity_measure, spectra):
    similarity = FingerprintSimilarity(
        fingerprint_generator=fingerprint_generator,
        similarity_measure=similarity_measure,
    )

    with pytest.raises(NotImplementedError, match="pair"):
        similarity.pair(spectra[0], spectra[1])


@pytest.mark.parametrize("similarity_measure", ["cosine", "tanimoto"])
def test_fingerprint_similarity_matrix_from_spectra(similarity_measure, fingerprint_generator, spectra):
    similarity = FingerprintSimilarity(
        fingerprint_generator=fingerprint_generator,
        similarity_measure=similarity_measure,
        return_csr=False,
    )

    scores = similarity.matrix(spectra)

    assert isinstance(scores, Scores)
    assert scores.is_sparse is False
    assert scores.is_scalar is True
    assert scores.score_fields == ("score",)

    score_array = scores.to_array()
    assert score_array.shape == (3, 3)
    assert np.allclose(score_array, score_array.T, atol=1e-6)
    assert np.allclose(np.diag(score_array), np.ones(3), atol=1e-6)


@pytest.mark.parametrize("similarity_measure", ["cosine", "tanimoto"])
def test_fingerprint_similarity_matrix_from_dense_fingerprints(
    similarity_measure, fingerprint_generator, dense_fingerprints
):
    similarity = FingerprintSimilarity(
        fingerprint_generator=fingerprint_generator,
        similarity_measure=similarity_measure,
    )

    scores = similarity.matrix(fingerprints_1=dense_fingerprints)

    assert isinstance(scores, Scores)
    assert scores.is_sparse is False
    assert scores.is_scalar is True
    assert scores.score_fields == ("score",)

    score_array = scores.to_array()
    assert score_array.shape == (3, 3)
    assert np.allclose(score_array, score_array.T, atol=1e-6)
    assert np.allclose(np.diag(score_array), np.ones(3), atol=1e-6)


@pytest.mark.parametrize("similarity_measure", ["cosine", "tanimoto"])
def test_fingerprint_similarity_matrix_from_sparse_fingerprints(
    similarity_measure, fingerprint_generator, sparse_fingerprints
):
    similarity = FingerprintSimilarity(
        fingerprint_generator=fingerprint_generator,
        similarity_measure=similarity_measure,
    )

    scores = similarity.matrix(fingerprints_1=sparse_fingerprints)

    assert isinstance(scores, Scores)
    assert scores.is_sparse is False
    assert scores.is_scalar is True

    score_array = scores.to_array()
    assert score_array.shape == (3, 3)
    assert np.allclose(score_array, score_array.T, atol=1e-6)
    assert np.allclose(np.diag(score_array), np.ones(3), atol=1e-6)


@pytest.mark.parametrize("similarity_measure", ["cosine", "tanimoto"])
def test_fingerprint_similarity_matrix_spectra_and_precomputed_dense_match(
    similarity_measure, fingerprint_generator, spectra, dense_fingerprints
):
    similarity = FingerprintSimilarity(
        fingerprint_generator=fingerprint_generator,
        similarity_measure=similarity_measure,
        return_csr=False,
    )

    scores_from_spectra = similarity.matrix(spectra)
    scores_from_fingerprints = similarity.matrix(fingerprints_1=dense_fingerprints)

    np.testing.assert_allclose(
        scores_from_spectra.to_array(),
        scores_from_fingerprints.to_array(),
        atol=1e-6,
    )


@pytest.mark.parametrize("similarity_measure", ["cosine", "tanimoto"])
def test_fingerprint_similarity_matrix_spectra_and_precomputed_sparse_match(
    similarity_measure, fingerprint_generator, spectra, sparse_fingerprints
):
    similarity = FingerprintSimilarity(
        fingerprint_generator=fingerprint_generator,
        similarity_measure=similarity_measure,
        return_csr=True,
    )

    scores_from_spectra = similarity.matrix(spectra)
    scores_from_fingerprints = similarity.matrix(fingerprints_1=sparse_fingerprints)

    np.testing.assert_allclose(
        scores_from_spectra.to_array(),
        scores_from_fingerprints.to_array(),
        atol=1e-6,
    )


@pytest.mark.parametrize("similarity_measure", ["cosine", "tanimoto"])
def test_fingerprint_similarity_matrix_mixed_input(
    similarity_measure, fingerprint_generator, spectra, dense_fingerprints
):
    similarity = FingerprintSimilarity(
        fingerprint_generator=fingerprint_generator,
        similarity_measure=similarity_measure,
        return_csr=False,
    )

    scores = similarity.matrix(
        spectra_1=spectra[:2],
        fingerprints_2=dense_fingerprints,
    )

    assert isinstance(scores, Scores)
    assert scores.shape == (2, 3)


@pytest.mark.parametrize("similarity_measure", ["cosine", "tanimoto"])
def test_fingerprint_similarity_matrix_asymmetric_input(
    similarity_measure, fingerprint_generator, spectra
):
    similarity = FingerprintSimilarity(
        fingerprint_generator=fingerprint_generator,
        similarity_measure=similarity_measure,
        return_csr=False,
    )

    scores = similarity.matrix(spectra_1=spectra[:2], spectra_2=spectra[1:])

    assert isinstance(scores, Scores)
    assert scores.shape == (2, 2)

    score_array = scores.to_array()

    # Only spectra[1] vs spectra[1] is a self-comparison in this layout.
    assert score_array[1, 0] == pytest.approx(1.0, 1e-6)

    # The diagonal entries are not self-comparisons here.
    assert score_array[0, 0] < 1.0
    assert score_array[1, 1] < 1.0


@pytest.mark.parametrize("similarity_measure", ["cosine", "tanimoto"])
def test_fingerprint_similarity_with_invalid_spectra_set_empty_nan(
    similarity_measure, fingerprint_generator, spectra_with_invalid
):
    similarity = FingerprintSimilarity(
        fingerprint_generator=fingerprint_generator,
        similarity_measure=similarity_measure,
        set_empty_scores="nan",
        return_csr=False,
    )

    scores = similarity.matrix(spectra_with_invalid)
    score_array = scores.to_array()

    expected_shape = (2, 2)
    assert score_array.shape == expected_shape
    assert np.allclose(np.diag(score_array), np.ones(2), atol=1e-6)


@pytest.mark.parametrize("similarity_measure", ["cosine", "tanimoto"])
def test_fingerprint_similarity_empty_fingerprint_container_raises(
    similarity_measure, fingerprint_generator
):
    fps = Fingerprints(
        fingerprint_generator=fingerprint_generator,
        return_csr=False,
    )
    fps.compute_fingerprints([])

    similarity = FingerprintSimilarity(
        fingerprint_generator=fingerprint_generator,
        similarity_measure=similarity_measure,
    )

    with pytest.raises(ValueError, match="Fingerprint container is empty"):
        similarity.matrix(fingerprints_1=fps)


def test_fingerprint_similarity_unknown_similarity_measure_raises(fingerprint_generator):
    with pytest.raises(AssertionError, match="Unknown similarity measure"):
        FingerprintSimilarity(
            fingerprint_generator=fingerprint_generator,
            similarity_measure="dice",
        )


def test_fingerprint_similarity_requires_input(fingerprint_generator):
    similarity = FingerprintSimilarity(
        fingerprint_generator=fingerprint_generator,
        similarity_measure="cosine",
    )

    with pytest.raises(ValueError, match="Either spectra_1 or fingerprints_1 must be provided"):
        similarity.matrix()


def test_fingerprint_similarity_rejects_duplicate_input_sources(fingerprint_generator, spectra, dense_fingerprints):
    similarity = FingerprintSimilarity(
        fingerprint_generator=fingerprint_generator,
        similarity_measure="cosine",
    )

    with pytest.raises(ValueError, match="Provide either spectra_1 or fingerprints_1, not both"):
        similarity.matrix(
            spectra_1=spectra,
            fingerprints_1=dense_fingerprints,
        )

    with pytest.raises(ValueError, match="Provide either spectra_2 or fingerprints_2, not both"):
        similarity.matrix(
            spectra_1=spectra,
            spectra_2=spectra,
            fingerprints_2=dense_fingerprints,
        )


@pytest.mark.parametrize("similarity_measure", ["cosine", "tanimoto"])
def test_fingerprint_similarity_score_field_alias_works(
    similarity_measure, fingerprint_generator, spectra
):
    similarity = FingerprintSimilarity(
        fingerprint_generator=fingerprint_generator,
        similarity_measure=similarity_measure,
    )

    scores = similarity.matrix(spectra)
    score_view = scores["score"]

    assert isinstance(score_view, Scores)
    assert score_view.score_fields == ("score",)
    np.testing.assert_array_equal(scores.to_array(), score_view.to_array())


@pytest.mark.parametrize("similarity_measure", ["cosine", "tanimoto"])
def test_fingerprint_similarity_sparse_and_dense_fingerprint_inputs_agree(
    similarity_measure, fingerprint_generator, dense_fingerprints, sparse_fingerprints
):
    similarity = FingerprintSimilarity(
        fingerprint_generator=fingerprint_generator,
        similarity_measure=similarity_measure,
    )

    dense_scores = similarity.matrix(fingerprints_1=dense_fingerprints)
    sparse_scores = similarity.matrix(fingerprints_1=sparse_fingerprints)

    np.testing.assert_allclose(
        dense_scores.to_array(),
        sparse_scores.to_array(),
        atol=1e-6,
    )
