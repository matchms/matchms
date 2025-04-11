import logging
import numpy as np
import pandas as pd
import pytest
from rdkit import Chem
from rdkit.Chem.rdchem import Mol
from matchms import Spectrum
from matchms.Fingerprints import (
    Fingerprints,
    _derive_fingerprint_from_inchi,
    _derive_fingerprint_from_smiles,
    _get_mol,
    _mol_to_fingerprint,
    _mols_to_fingerprints,
    _validate_metadata,
)
from .builder_Spectrum import SpectrumBuilder


LOGGER = logging.getLogger(__name__)


@pytest.fixture
def valid_spectrum() -> Spectrum:
    metadata = {
        "inchikey": "KFDYZSPFVRTLML-UHFFFAOYSA-N",
        "smiles": "C1COCCN1C2=NC(=NC(=N2)NC3=CC(=C(C=C3)C=CC4=C(C=C(C=C4)NC5=NC(=NC(=N5)Cl)N6CCOCC6)S(=O)(=O)O)S(=O)(=O)O)Cl",
    }
    return SpectrumBuilder().with_metadata(metadata).build()


@pytest.fixture
def valid_inchi_spectrum() -> Spectrum:
    # pylint: disable=line-too-long
    metadata = {
        "inchi": "InChI=1S/C28H28Cl2N10O8S2/c29-23-33-25(37-27(35-23)39-7-11-47-12-8-39)31-19-5-3-17(21(15-19)49(41,42)43)1-2-18-4-6-20(16-22(18)50(44,45)46)32-26-34-24(30)36-28(38-26)40-9-13-48-14-10-40/h1-6,15-16H,7-14H2,(H,41,42,43)(H,44,45,46)(H,31,33,35,37)(H,32,34,36,38)"  # noqa
    }  # noqa
    return SpectrumBuilder().with_metadata(metadata).build()


@pytest.fixture
def invalid_metadata_spectrum() -> Spectrum:
    return SpectrumBuilder().build()


@pytest.fixture
def invalid_inchi_spectrum() -> Spectrum:
    return SpectrumBuilder().with_metadata({"inchi": "invalid"}).build()


@pytest.fixture
def invalid_smiles_spectrum() -> Spectrum:
    return SpectrumBuilder().with_metadata({"smiles": "invalid"}).build()


@pytest.fixture
def invalid_inchi_smiles_spectrum() -> Spectrum:
    metadata = {"inchikey": "KFDYZSPFVRTLML-UHFFFAOYSA-N", "inchi": "invalid", "smiles": "invalid"}
    return SpectrumBuilder().with_metadata(metadata).build()


@pytest.fixture
def invalid_inchikey_spectrum() -> Spectrum:
    metadata = {
        "inchikey": "invalid",
        # pylint: disable=line-too-long
        "inchi": "InChI=1S/C28H28Cl2N10O8S2/c29-23-33-25(37-27(35-23)39-7-11-47-12-8-39)31-19-5-3-17(21(15-19)49(41,42)43)1-2-18-4-6-20(16-22(18)50(44,45)46)32-26-34-24(30)36-28(38-26)40-9-13-48-14-10-40/h1-6,15-16H,7-14H2,(H,41,42,43)(H,44,45,46)(H,31,33,35,37)(H,32,34,36,38)",  # noqa: E501
        "smiles": "C1COCCN1C2=NC(=NC(=N2)NC3=CC(=C(C=C3)C=CC4=C(C=C(C=C4)NC5=NC(=NC(=N5)Cl)N6CCOCC6)S(=O)(=O)O)S(=O)(=O)O)Cl",
    }
    return SpectrumBuilder().with_metadata(metadata).build()


@pytest.fixture
def valid_spectra(valid_spectrum):
    metadata = {"inchikey": "HINREHSUCWWBNO-UHFFFAOYSA-N", "smiles": "CCOC1=C(C=CC(=C1)C=NNC(=O)COC2=C(C=CC(=C2)C)C(C)C)OC(=O)C3=C(C=C(C=C3)Cl)Cl"}
    diff_spectrum = SpectrumBuilder().with_metadata(metadata).build()

    return [valid_spectrum, diff_spectrum]


@pytest.fixture
def valid_same_spectra(valid_spectrum):
    return [valid_spectrum, valid_spectrum]


def test_fingerprint_defaults():
    fp = Fingerprints()
    assert fp.fingerprint_algorithm == "daylight"
    assert fp.fingerprint_method == "bit"
    assert fp.nbits == 2048
    assert not fp.ignore_stereochemistry


def test_fingerprint_config():
    fp = Fingerprints(fingerprint_algorithm="morgan2", nbits=1024)
    config = fp.config
    assert config["fingerprint_algorithm"] == "morgan2"
    assert config["nbits"] == 1024
    assert config["ingore_stereochemistry"] is False


def test_compute_fingerprints_valid_spectra(valid_spectra):
    fp = Fingerprints()
    fp.compute_fingerprints(valid_spectra)
    assert fp.fingerprint_count == 2
    assert isinstance(fp.fingerprints["KFDYZSPFVRTLML-UHFFFAOYSA-N"], np.ndarray)
    assert fp.fingerprints["KFDYZSPFVRTLML-UHFFFAOYSA-N"].size == 2048
    assert isinstance(fp.to_dataframe, pd.DataFrame)
    assert len(fp.to_dataframe.index) == 2


def test_compute_fingerprints_no_inchikey(invalid_metadata_spectrum, caplog):
    with caplog.at_level(logging.WARNING):
        Fingerprints().compute_fingerprints([invalid_metadata_spectrum])
    assert "doesn't have a inchikey. Skipping." in caplog.text


def test_compute_fingerprints_same_spectra(valid_same_spectra):
    fp = Fingerprints()
    fp.compute_fingerprints(valid_same_spectra)
    assert fp.fingerprint_count == 1
    assert isinstance(fp.fingerprints["KFDYZSPFVRTLML-UHFFFAOYSA-N"], np.ndarray)
    assert fp.fingerprints["KFDYZSPFVRTLML-UHFFFAOYSA-N"].size == 2048


def test_get_fingerprint_by_inchikey(valid_spectrum, caplog):
    fp = Fingerprints()
    fp.compute_fingerprints([valid_spectrum])
    assert isinstance(fp.get_fingerprint_by_inchikey("KFDYZSPFVRTLML-UHFFFAOYSA-N"), np.ndarray)

    with caplog.at_level(logging.WARNING):
        fp.get_fingerprint_by_inchikey("invalid")
    assert "The provided inchikey is not valid or may be the short form." in caplog.text

    with caplog.at_level(logging.WARNING):
        fp.get_fingerprint_by_inchikey("HINREHSUCWWBNO-UHFFFAOYSA-N")
    assert "Fingerprint is not present for given Spectrum/InchiKey. Use compute_fingerprint() first." in caplog.text


def test_get_fingerprint_by_spectrum(valid_spectrum):
    fp = Fingerprints()
    fp.compute_fingerprints([valid_spectrum])
    assert isinstance(fp.get_fingerprint_by_spectrum(valid_spectrum), np.ndarray)


def test_compute_fingerprint_valid(valid_spectrum, invalid_metadata_spectrum, valid_inchi_spectrum):
    fp = Fingerprints()
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
    ],
)
def test_get_mol(spectrum_fixture, expected_result, request):
    spectrum = request.getfixturevalue(spectrum_fixture)

    res = _get_mol(spectrum)

    if expected_result:
        assert isinstance(res, Mol), "Object is not of RDKit Mol type"
    else:
        assert res is None


@pytest.mark.parametrize(
    "spectrum_fixture, ignore_stereochemistry, raises_exception",
    [
        ("valid_spectrum", True, False),
        ("valid_spectrum", False, False),
        ("invalid_inchikey_spectrum", True, True),
        ("invalid_metadata_spectrum", False, True),
        ("invalid_inchi_smiles_spectrum", False, True),
    ],
)
def test_validate_metadata(spectrum_fixture, ignore_stereochemistry, raises_exception, request):
    spectrum = request.getfixturevalue(spectrum_fixture)

    if raises_exception:
        with pytest.raises(ValueError, match="Inchikey is missing or invalid.|Inchi or smiles is missing or invalid."):
            _validate_metadata(spectrum, ignore_stereochemistry)
    else:
        res = _validate_metadata(spectrum, ignore_stereochemistry)
        assert isinstance(res, Spectrum), "Object is not of SpectrumType"

        if ignore_stereochemistry:
            assert res.get("inchikey") == spectrum.get("inchikey")[:14], "Inchikeys do not match"
        else:
            assert res.get("inchikey") == spectrum.get("inchikey"), "Inchikeys do not match"


@pytest.mark.parametrize(
    "fingerprint_algorithm, fingerprint_type, expected_shape",
    [
        ("daylight", "bit", (3, 1024)),
        ("morgan1", "bit", (3, 1024)),
        ("morgan2", "bit", (3, 1024)),
        ("morgan3", "count", (3, 1024)),
    ],
)
def test_mols_to_fingerprints_valid(fingerprint_algorithm, fingerprint_type, expected_shape):
    smiles = ["CCO", "CCN", "CCC"]
    mols = [Chem.MolFromSmiles(smi) for smi in smiles]
    nbits = 1024

    fingerprints = _mols_to_fingerprints(mols=mols, fingerprint_algorithm=fingerprint_algorithm, fingerprint_type=fingerprint_type, nbits=nbits)
    assert fingerprints.shape == expected_shape
    assert fingerprints.dtype == np.int8
    assert np.any(fingerprints)


@pytest.mark.parametrize(
    "fingerprint_algorithm, fingerprint_type, exception_type, match",
    [
        ("invalid_algorithm", "bit", ValueError, "Unkown fingerprint algorithm given"),
        ("daylight", "invalid_type", ValueError, "Unkown fingerprint type given"),
    ],
)
def test_mols_to_fingerprints_invalid_cases(fingerprint_algorithm, fingerprint_type, exception_type, match):
    smiles = ["CCO", "CCN", "CCC"]
    mols = [Chem.MolFromSmiles(smi) for smi in smiles]
    nbits = 1024

    with pytest.raises(exception_type, match=match):
        _mols_to_fingerprints(mols=mols, fingerprint_algorithm=fingerprint_algorithm, fingerprint_type=fingerprint_type, nbits=nbits)


@pytest.mark.parametrize(
    "fingerprint_algorithm, fingerprint_type, exception_type, match",
    [
        ("invalid_algorithm", "bit", ValueError, "Unkown fingerprint algorithm given"),
        ("daylight", "invalid_type", ValueError, "Unkown fingerprint type given"),
    ],
)
def test_mol_to_fingerprints_invalid_cases(fingerprint_algorithm, fingerprint_type, exception_type, match):
    nbits = 1024

    with pytest.raises(exception_type, match=match):
        _mol_to_fingerprint(mol=Chem.MolFromSmiles("CCO"), fingerprint_algorithm=fingerprint_algorithm, fingerprint_type=fingerprint_type, nbits=nbits)


def test_mols_to_fingerprints_empty_molecules():
    nbits = 1024
    fingerprints = _mols_to_fingerprints(mols=[], fingerprint_algorithm="daylight", fingerprint_type="bit", nbits=nbits)
    assert fingerprints.shape == (0, nbits)


def test_derive_fingerprint_invalid_mol():
    fingerprint = _derive_fingerprint_from_smiles("invalid", "daylight", "bit", 1024)
    assert fingerprint is None

    fingerprint = _derive_fingerprint_from_inchi("invalid", "daylight", "bit", 1024)
    assert fingerprint is None
