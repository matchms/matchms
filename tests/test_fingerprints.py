import logging

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp
from rdkit.Chem import rdFingerprintGenerator

from matchms import Spectrum
from matchms.Fingerprints import Fingerprints
from tests.builder_Spectrum import SpectrumBuilder


pytest.importorskip("chemap")


LOGGER = logging.getLogger(__name__)


@pytest.fixture
def fingerprint_generator():
    return rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=256)


@pytest.fixture
def valid_spectrum() -> Spectrum:
    metadata = {
        "inchikey": "KFDYZSPFVRTLML-UHFFFAOYSA-N",
        "smiles": (
            "C1COCCN1C2=NC(=NC(=N2)NC3=CC(=C(C=C3)C=CC4=C(C=C(C=C4)NC5=NC(=NC(=N5)Cl)N6CCOCC6)"
            "S(=O)(=O)O)S(=O)(=O)O)Cl"
        ),
    }
    return SpectrumBuilder().with_metadata(metadata).build()


@pytest.fixture
def valid_inchi_spectrum() -> Spectrum:
    metadata = {
        "inchikey": "KFDYZSPFVRTLML-UHFFFAOYSA-N",
        "inchi": (
            "InChI=1S/C28H28Cl2N10O8S2/c29-23-33-25(37-27(35-23)39-7-11-47-12-8-39)31-19-5-3-17(21(15-19)"
            "49(41,42)43)1-2-18-4-6-20(16-22(18)50(44,45)46)32-26-34-24(30)36-28(38-26)40-9-13-48-14-10-40/"
            "h1-6,15-16H,7-14H2,(H,41,42,43)(H,44,45,46)(H,31,33,35,37)(H,32,34,36,38)"
        ),
    }
    return SpectrumBuilder().with_metadata(metadata).build()


@pytest.fixture
def invalid_metadata_spectrum() -> Spectrum:
    return SpectrumBuilder().build()


@pytest.fixture
def invalid_inchi_spectrum() -> Spectrum:
    metadata = {
        "inchikey": "KFDYZSPFVRTLML-UHFFFAOYSA-N",
        "inchi": "invalid",
    }
    return SpectrumBuilder().with_metadata(metadata).build()


@pytest.fixture
def invalid_smiles_spectrum() -> Spectrum:
    metadata = {
        "inchikey": "KFDYZSPFVRTLML-UHFFFAOYSA-N",
        "smiles": "invalid",
    }
    return SpectrumBuilder().with_metadata(metadata).build()


@pytest.fixture
def invalid_inchi_smiles_spectrum() -> Spectrum:
    metadata = {
        "inchikey": "KFDYZSPFVRTLML-UHFFFAOYSA-N",
        "inchi": "invalid",
        "smiles": "invalid",
    }
    return SpectrumBuilder().with_metadata(metadata).build()


@pytest.fixture
def invalid_inchikey_spectrum() -> Spectrum:
    metadata = {
        "inchikey": "invalid",
        "smiles": (
            "C1COCCN1C2=NC(=NC(=N2)NC3=CC(=C(C=C3)C=CC4=C(C=C(C=C4)NC5=NC(=NC(=N5)Cl)N6CCOCC6)"
            "S(=O)(=O)O)S(=O)(=O)O)Cl"
        ),
    }
    return SpectrumBuilder().with_metadata(metadata).build()


@pytest.fixture
def valid_spectra(valid_spectrum):
    metadata = {
        "inchikey": "HINREHSUCWWBNO-UHFFFAOYSA-N",
        "smiles": "CCOC1=C(C=CC(=C1)C=NNC(=O)COC2=C(C=CC(=C2)C)C(C)C)OC(=O)C3=C(C=C(C=C3)Cl)Cl",
    }
    diff_spectrum = SpectrumBuilder().with_metadata(metadata).build()
    return [valid_spectrum, diff_spectrum]


@pytest.fixture
def valid_same_spectra(valid_spectrum):
    return [valid_spectrum, valid_spectrum]


def test_fingerprint_defaults(fingerprint_generator):
    fp = Fingerprints(fingerprint_generator=fingerprint_generator)

    assert fp.ignore_stereochemistry is False
    assert fp.count is False
    assert fp.folded is True
    assert fp.return_csr is False
    assert fp.invalid_policy == "raise"
    assert fp.fingerprint_count == 0
    assert fp.fingerprints is None
    assert fp.inchikeys == []


def test_fingerprint_config(fingerprint_generator):
    fp = Fingerprints(
        fingerprint_generator=fingerprint_generator,
        ignore_stereochemistry=True,
        count=True,
        folded=False,
        return_csr=True,
        invalid_policy="filter",
        batch_size=128,
    )

    config = fp.config
    assert config["ignore_stereochemistry"] is True
    assert config["count"] is True
    assert config["folded"] is False
    assert config["return_csr"] is True
    assert config["invalid_policy"] == "filter"
    assert config["additional_keyword_arguments"]["batch_size"] == 128


def test_compute_fingerprints_valid_spectra_dense(valid_spectra, fingerprint_generator):
    fp = Fingerprints(
        fingerprint_generator=fingerprint_generator,
        return_csr=False,
    )
    fp.compute_fingerprints(valid_spectra)

    assert fp.fingerprint_count == 2
    assert isinstance(fp.fingerprints, np.ndarray)
    assert fp.fingerprints.shape[0] == 2
    assert fp.fingerprints.shape[1] == 256

    first_fp = fp.get_fingerprint_by_inchikey("KFDYZSPFVRTLML-UHFFFAOYSA-N")
    assert isinstance(first_fp, np.ndarray)
    assert first_fp.shape == (256,)
    assert first_fp.sum() > 0

    assert isinstance(fp.to_dataframe, pd.DataFrame)
    assert len(fp.to_dataframe.index) == 2


def test_compute_fingerprints_valid_spectra_sparse(valid_spectra, fingerprint_generator):
    fp = Fingerprints(
        fingerprint_generator=fingerprint_generator,
        return_csr=True,
    )
    fp.compute_fingerprints(valid_spectra)

    assert fp.fingerprint_count == 2
    assert sp.issparse(fp.fingerprints)
    assert fp.fingerprints.shape == (2, 256)

    first_fp = fp.get_fingerprint_by_inchikey("KFDYZSPFVRTLML-UHFFFAOYSA-N")
    assert sp.issparse(first_fp)
    assert first_fp.shape == (1, 256)


def test_compute_fingerprints_no_inchikey(invalid_metadata_spectrum, fingerprint_generator, caplog):
    with caplog.at_level(logging.WARNING):
        Fingerprints(fingerprint_generator=fingerprint_generator).compute_fingerprints([invalid_metadata_spectrum])

    assert "doesn't have valid fingerprint metadata. Skipping." in caplog.text


def test_compute_fingerprints_same_spectra_deduplicates(valid_same_spectra, fingerprint_generator):
    fp = Fingerprints(fingerprint_generator=fingerprint_generator)
    fp.compute_fingerprints(valid_same_spectra)

    assert fp.fingerprint_count == 1
    assert isinstance(fp.get_fingerprint_by_inchikey("KFDYZSPFVRTLML-UHFFFAOYSA-N"), np.ndarray)


def test_get_fingerprint_by_inchikey_dense(valid_spectrum, fingerprint_generator, caplog):
    fp = Fingerprints(fingerprint_generator=fingerprint_generator)
    fp.compute_fingerprints([valid_spectrum])

    fingerprint = fp.get_fingerprint_by_inchikey("KFDYZSPFVRTLML-UHFFFAOYSA-N")
    assert isinstance(fingerprint, np.ndarray)

    with caplog.at_level(logging.WARNING):
        missing = fp.get_fingerprint_by_inchikey("HINREHSUCWWBNO-UHFFFAOYSA-N")
    assert missing is None
    assert "Fingerprint is not present for given Spectrum/InChIKey. Use compute_fingerprints() first." in caplog.text


def test_get_fingerprint_by_inchikey_invalid_logs_warning(valid_spectrum, fingerprint_generator, caplog):
    fp = Fingerprints(fingerprint_generator=fingerprint_generator)
    fp.compute_fingerprints([valid_spectrum])

    with caplog.at_level(logging.WARNING):
        missing = fp.get_fingerprint_by_inchikey("invalid")

    assert missing is None
    assert "The provided InChIKey is not valid or may be the short form." in caplog.text


def test_get_fingerprint_by_inchikey_short_key_raises_without_ignore_stereochemistry(valid_spectrum, fingerprint_generator):
    fp = Fingerprints(fingerprint_generator=fingerprint_generator)
    fp.compute_fingerprints([valid_spectrum])

    with pytest.raises(ValueError, match="Expected full 27 character InChIKey"):
        fp.get_fingerprint_by_inchikey("KFDYZSPFVRTLML")


def test_get_fingerprint_by_inchikey_short_key_works_with_ignore_stereochemistry(valid_spectrum, fingerprint_generator):
    fp = Fingerprints(
        fingerprint_generator=fingerprint_generator,
        ignore_stereochemistry=True,
    )
    fp.compute_fingerprints([valid_spectrum])

    fingerprint = fp.get_fingerprint_by_inchikey("KFDYZSPFVRTLML")
    assert fingerprint is not None


def test_get_fingerprint_by_spectrum(valid_spectrum, fingerprint_generator):
    fp = Fingerprints(fingerprint_generator=fingerprint_generator)
    fp.compute_fingerprints([valid_spectrum])

    fingerprint = fp.get_fingerprint_by_spectrum(valid_spectrum)
    assert isinstance(fingerprint, np.ndarray)


def test_compute_fingerprint_valid(valid_spectrum, invalid_metadata_spectrum, valid_inchi_spectrum, fingerprint_generator):
    fp = Fingerprints(fingerprint_generator=fingerprint_generator)

    assert isinstance(fp.compute_fingerprint(valid_spectrum), np.ndarray)
    assert fp.compute_fingerprint(invalid_metadata_spectrum) is None
    assert isinstance(fp.compute_fingerprint(valid_inchi_spectrum), np.ndarray)


@pytest.mark.parametrize(
    "spectrum_fixture, expected_result",
    [
        ("valid_spectrum", True),
        ("valid_inchi_spectrum", True),
        ("invalid_metadata_spectrum", False),
        ("invalid_inchi_spectrum", False),
        ("invalid_smiles_spectrum", False),
        ("invalid_inchi_smiles_spectrum", False),
        ("invalid_inchikey_spectrum", False),
    ],
)
def test_compute_fingerprint_input_validation(spectrum_fixture, expected_result, request, fingerprint_generator):
    spectrum = request.getfixturevalue(spectrum_fixture)
    fp = Fingerprints(fingerprint_generator=fingerprint_generator)

    result = fp.compute_fingerprint(spectrum)

    if expected_result:
        assert result is not None
    else:
        assert result is None


def test_compute_fingerprints_ignore_stereochemistry_uses_short_inchikey(valid_spectrum, fingerprint_generator):
    fp = Fingerprints(
        fingerprint_generator=fingerprint_generator,
        ignore_stereochemistry=True,
    )
    fp.compute_fingerprints([valid_spectrum])

    assert fp.fingerprint_count == 1
    assert fp.inchikeys == ["KFDYZSPFVRTLML"]


def test_to_dataframe_dense(valid_spectra, fingerprint_generator):
    fp = Fingerprints(fingerprint_generator=fingerprint_generator, return_csr=False)
    fp.compute_fingerprints(valid_spectra)

    df = fp.to_dataframe
    assert isinstance(df, pd.DataFrame)
    assert list(df.index) == fp.inchikeys
    assert "fingerprint" in df.columns
    assert isinstance(df.iloc[0]["fingerprint"], np.ndarray)


def test_to_dataframe_sparse(valid_spectra, fingerprint_generator):
    fp = Fingerprints(fingerprint_generator=fingerprint_generator, return_csr=True)
    fp.compute_fingerprints(valid_spectra)

    df = fp.to_dataframe
    assert isinstance(df, pd.DataFrame)
    assert list(df.index) == fp.inchikeys
    assert "fingerprint" in df.columns
    assert sp.issparse(df.iloc[0]["fingerprint"])


def test_compute_fingerprints_replaces_previous_state(valid_spectrum, valid_spectra, fingerprint_generator):
    fp = Fingerprints(fingerprint_generator=fingerprint_generator)
    fp.compute_fingerprints([valid_spectrum])
    assert fp.fingerprint_count == 1

    fp.compute_fingerprints(valid_spectra)
    assert fp.fingerprint_count == 2
    assert set(fp.inchikeys) == {
        "KFDYZSPFVRTLML-UHFFFAOYSA-N",
        "HINREHSUCWWBNO-UHFFFAOYSA-N",
    }


def test_get_fingerprint_by_spectrum_without_inchikey_returns_none(invalid_metadata_spectrum, fingerprint_generator, caplog):
    fp = Fingerprints(fingerprint_generator=fingerprint_generator)
    fp.compute_fingerprints([])

    with caplog.at_level(logging.WARNING):
        result = fp.get_fingerprint_by_spectrum(invalid_metadata_spectrum)

    assert result is None
    assert "No InChIKey provided." in caplog.text
