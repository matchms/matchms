import sys
from contextlib import nullcontext
from importlib import reload
from importlib.util import find_spec
from unittest import mock
import numpy as np
import pytest
import matchms.metadata_utils
from matchms.metadata_utils import (derive_fingerprint_from_inchi,
                                    derive_fingerprint_from_smiles,
                                    is_valid_inchi, is_valid_inchikey,
                                    is_valid_smiles, mol_converter)
from matchms.filtering.repair_adduct.clean_adduct import looks_like_adduct, clean_adduct


@pytest.fixture()
def reload_metadata_utils():
    """Reload metadata_utils module after test has finished."""
    yield
    reload(matchms.metadata_utils)


def test_mol_converter_smiles_to_inchi():
    """Test if smiles is correctly converted to inchi."""
    pytest.importorskip("rdkit")

    mol_input = "C[Si](Cn1cncn1)(c1ccc(F)cc1)"
    output_inchi = mol_converter(mol_input, "smiles", "inchi")
    assert output_inchi == "InChI=1S/C10H11FN3Si/c1-15(8-14-7-12-6-13-14)10-4-2-9(11)3-5-10/h2-7H,8H2,1H3"


def test_mol_converter_inchi_to_smiles():
    """Test if inchi is correctly converted to smiles."""
    pytest.importorskip("rdkit")

    mol_input = "InChI=1S/C10H11FN3Si/c1-15(8-14-7-12-6-13-14)10-4-2-9(11)3-5-10/h2-7H,8H2,1H3"
    output_smiles = mol_converter(mol_input, "inchi", "smiles")
    assert output_smiles == "C[Si](Cn1cncn1)=C1C=C[C](F)C=C1"


def test_mol_converter_smiles_to_inchikey():
    """Test if smiles is correctly converted to inchikey."""
    pytest.importorskip("rdkit")

    mol_input = "C[Si](Cn1cncn1)(c1ccc(F)cc1)"
    output_inchikey = mol_converter(mol_input, "smiles", "inchikey")
    assert output_inchikey == "HULABQRTZQYJBQ-UHFFFAOYSA-N"


def test_mol_converter_invalid_input():
    """Test invalid entry."""
    pytest.importorskip("rdkit")

    assert mol_converter("invalid_test", "smiles", "inchikey") is None, "Expected None."


def test_is_valid_inchikey():
    """Test if strings are correctly classified."""
    inchikeys_true = ["XYLJNLCSTIOKRM-UHFFFAOYSA-N"]
    inchikeys_false = [
        "XYLJNLCSTIOKRM-UHFFFAOYSA",
        "XYLJNLCSTIOKRMRUHFFFAOYSASN",
        "XYLJNLCSTIOKR-MUHFFFAOYSA-N",
        "XYLJNLCSTIOKRM-UHFFFAOYSA-NN",
        "Brcc(NC2=NCN2)-ccc3nccnc1-3",
        "2YLJNLCSTIOKRM-UHFFFAOYSA-N",
        "XYLJNLCSTIOKRM-aaaaaaaaaa-a"
    ]

    for inchikey in inchikeys_true:
        assert is_valid_inchikey(inchikey), "Expected inchikey is True."
    for inchikey in inchikeys_false:
        assert not is_valid_inchikey(inchikey), "Expected inchikey is False."


def test_is_valid_inchikey_none_input():
    """Test None entry."""
    assert not is_valid_inchikey(None), "Expected None entry to give False."




@pytest.mark.parametrize("inchi, expected", [
    ["InChI=1S/C2H7N3/c1-5-2(3)4/h1H3,(H4,3,4,5)", True],
    ['"InChI=1S/C2H7N3/c1-5-2(3)4/h1H3,(H4,3,4,5)"', True],
    ["InChI=1S/Ne", True],
    ["InChI=1S/C14H9Cl5/c15-11-5-1-9(2-6-11)13(14(17,18)19)10-3-7-12(16)8-4-10/h1-8,13H", True],
    ["1S/C2H7N3/c1-5-2(3)4/h1H3,(H4,3,4,5)", False],
    ["InChI=1S/C2H7N3/c152(3)4/h1H3,(H4,3,4,5)", False],
    ["InChI=C2H7N3/c1-5-2(3)4/h1H3,(H4,3,4,5)", False],
    ["InChI=1S/C2H7N3/c1-5-2(3)", False]
])
def test_is_valid_inchi(inchi, expected):
    pytest.importorskip("rdkit")
    assert is_valid_inchi(inchi) == expected


def test_is_valid_inchi_none_input():
    """Test None entry."""
    pytest.importorskip("rdkit")
    assert not is_valid_inchi(None), "Expected None entry to give False."


@pytest.mark.parametrize("smiles, expected", [
    [r"CN1COCN(CC2=CN=C(Cl)S2)\C1=N\[N+]([O-])=O", True],
    [r"CN1N(C(=O)C=C1C)c1ccccc1", True],
    [r"COC(=O)C1=CN=CC=N1", True],
    [r"C", True],
    [r"CF", True],
    [r"C#C", True],
    [r"C1=CC(=CC=C1C(C2=CC=C(C=C2)Cl)C(Cl)(Cl)Cl)Cl", True],
    [r"[18FH]", True],
    [r"F", True],
    [r"CN1N(C(=O)C=C1C)c1cccccx1", False],
    [r"CN1COCN(CC2=CN=C(Cl)S2)\C1=N\[N+++]([O-])=O", False],
    [r"COC(=O[)]C1=CN=CC=N1", False],
    [r"1S/C2H7N3/c1-5-2(3)4", False]
])
def test_is_valid_smiles(smiles, expected):
    pytest.importorskip("rdkit")
    assert is_valid_smiles(smiles) == expected


def test_is_valid_smiles_none_input():
    """Test None entry."""
    pytest.importorskip("rdkit")

    assert not is_valid_smiles(None), "Expected None entry to give False."


def test_derive_fingerprint_from_smiles():
    """Test if correct fingerprint is derived from given smiles."""
    pytest.importorskip("rdkit")

    fingerprint = derive_fingerprint_from_smiles("[C+]#C[O-]", "daylight", 16)
    expected_fingerprint = np.array([0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0])
    assert np.all(fingerprint == expected_fingerprint), "Expected different fingerprint."


def test_derive_fingerprint_from_inchi():
    """Test if correct fingerprint is derived from given inchi."""
    pytest.importorskip("rdkit")

    fingerprint = derive_fingerprint_from_inchi("InChI=1S/C2O/c1-2-3", "daylight", 16)
    expected_fingerprint = np.array([0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0])
    assert np.all(fingerprint == expected_fingerprint), "Expected different fingerprint."


def test_derive_fingerprint_different_types_from_smiles():
    """Test if correct fingerprints are derived from given smiles when using different types."""
    pytest.importorskip("rdkit")

    types = ["daylight", "morgan1", "morgan2", "morgan3"]
    expected_fingerprints = [
        np.array([0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0]),
        np.array([0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1]),
        np.array([0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1]),
        np.array([0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1])
    ]

    for i, fingerprint_type in enumerate(types):
        fingerprint = derive_fingerprint_from_smiles("[C+]#C[O-]", fingerprint_type, 16)
        assert np.all(fingerprint == expected_fingerprints[i]), "Expected different fingerprint."


def test_missing_rdkit_module_error(reload_metadata_utils):
    """Test if different functions return correct error when *rdkit* is not available"""
    # pylint: disable=unused-argument
    if find_spec("rdkit") is not None:
        context = mock.patch.dict(sys.modules, {"rdkit": None})
    else:
        context = nullcontext()

    expected_msg = "Conda package 'rdkit' is required for this functionality."
    with context:
        reload(matchms.metadata_utils)
        mol_input = "C[Si](Cn1cncn1)(c1ccc(F)cc1)"
        with pytest.raises(ImportError) as msg:
            _ = matchms.metadata_utils.mol_converter(mol_input, "smiles", "inchikey")
        assert expected_msg in str(msg.value), "Expected different ImportError."

        with pytest.raises(ImportError) as msg:
            _ = matchms.metadata_utils.is_valid_inchi("test")
        assert expected_msg in str(msg.value), "Expected different ImportError."

        with pytest.raises(ImportError) as msg:
            _ = matchms.metadata_utils.is_valid_smiles("test")
        assert expected_msg in str(msg.value), "Expected different ImportError."

        with pytest.raises(ImportError) as msg:
            _ = matchms.metadata_utils.derive_fingerprint_from_inchi(mol_input, "test", 0)
        assert expected_msg in str(msg.value), "Expected different ImportError."

        with pytest.raises(ImportError) as msg:
            _ = matchms.metadata_utils.derive_fingerprint_from_smiles(mol_input, "test", 0)
        assert expected_msg in str(msg.value), "Expected different ImportError."

        with pytest.raises(ImportError) as msg:
            _ = matchms.metadata_utils.mol_to_fingerprint(mol_input, "test", 0)
        assert expected_msg in str(msg.value), "Expected different ImportError."


def test_looks_like_adduct():
    """Test if adducts are correctly identified"""
    for adduct in ["M+", "M*+", "M+Cl", "[M+H]", "[2M+Na]+", "M+H+K", "[2M+ACN+H]+",
                   "MS+Na", "MS+H", "M3Cl37+Na", "[M+H+H2O]"]:
        assert looks_like_adduct(adduct), "Expected this to be identified as adduct"
    for adduct in ["N+", "B*+", "++", "--", "[--]", "H+M+K"]:
        assert not looks_like_adduct(adduct), "Expected this not to be identified as adduct"


@pytest.mark.parametrize("input_adduct, expected_adduct",
                         [("M+", "[M]+"),
                          ("M+CH3COO-", "[M+CH3COO]-"),
                          ("M+CH3COO", "[M+CH3COO]-"),
                          ("M-CH3-", "[M-CH3]-"),
                          ("M+2H++", "[M+2H]2+"),
                          ("[2M+Na]", "[2M+Na]+"),
                          ("2M+Na", "[2M+Na]+"),
                          ("M+NH3+", "[M+NH3]+"),
                          ("M-H2O+2H2+", "[M-H2O+2H]2+"),
                          (None, None)])
def test_clean_adduct_examples(input_adduct, expected_adduct):
    """Test if typical examples are correctly edited."""
    assert clean_adduct(input_adduct) == expected_adduct, \
        "Expected different cleaned adduct"
