# pylint: disable=wildcard-import, method-hidden
import sys
from contextlib import nullcontext
from importlib import reload
from importlib.util import find_spec
from unittest import mock
import numpy
import pytest
import matchms.utils
from matchms.utils import clean_adduct
from matchms.utils import derive_fingerprint_from_inchi
from matchms.utils import derive_fingerprint_from_smiles
from matchms.utils import is_valid_inchi
from matchms.utils import is_valid_inchikey
from matchms.utils import is_valid_smiles
from matchms.utils import looks_like_adduct
from matchms.utils import mol_converter


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


def test_is_valid_inchi():
    """Test if strings are correctly classified."""
    pytest.importorskip("rdkit")

    inchi_true = [
        "InChI=1S/C2H7N3/c1-5-2(3)4/h1H3,(H4,3,4,5)",
        '"InChI=1S/C2H7N3/c1-5-2(3)4/h1H3,(H4,3,4,5)"'
    ]
    inchi_false = [
        "1S/C2H7N3/c1-5-2(3)4/h1H3,(H4,3,4,5)",
        "InChI=1S/C2H7N3/c152(3)4/h1H3,(H4,3,4,5)",
        "InChI=C2H7N3/c1-5-2(3)4/h1H3,(H4,3,4,5)",
        "InChI=1S/C2H7N3/c1-5-2(3)"
    ]

    for inchi in inchi_true:
        assert is_valid_inchi(inchi), "Expected inchi is True."
    for inchi in inchi_false:
        assert not is_valid_inchi(inchi), "Expected inchi is False."


def test_is_valid_inchi_none_input():
    """Test None entry."""
    pytest.importorskip("rdkit")

    assert not is_valid_inchi(None), "Expected None entry to give False."


def test_is_valid_smiles():
    """Test if strings are correctly classified."""
    pytest.importorskip("rdkit")

    smiles_true = [
        r"CN1COCN(CC2=CN=C(Cl)S2)\C1=N\[N+]([O-])=O",
        r"CN1N(C(=O)C=C1C)c1ccccc1",
        r"COC(=O)C1=CN=CC=N1"
    ]
    smiles_false = [
        r"CN1N(C(=O)C=C1C)c1cccccx1",
        r"CN1COCN(CC2=CN=C(Cl)S2)\C1=N\[N+++]([O-])=O",
        r"COC(=O[)]C1=CN=CC=N1",
        r"1S/C2H7N3/c1-5-2(3)4"
    ]

    for smiles in smiles_true:
        assert is_valid_smiles(smiles), "Expected smiles is True."
    for smiles in smiles_false:
        assert not is_valid_smiles(smiles), "Expected smiles is False."


def test_is_valid_smiles_none_input():
    """Test None entry."""
    pytest.importorskip("rdkit")

    assert not is_valid_smiles(None), "Expected None entry to give False."


def test_derive_fingerprint_from_smiles():
    """Test if correct fingerprint is derived from given smiles."""
    pytest.importorskip("rdkit")

    fingerprint = derive_fingerprint_from_smiles("[C+]#C[O-]", "daylight", 16)
    expected_fingerprint = numpy.array([0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0])
    assert numpy.all(fingerprint == expected_fingerprint), "Expected different fingerprint."


def test_derive_fingerprint_from_inchi():
    """Test if correct fingerprint is derived from given inchi."""
    pytest.importorskip("rdkit")

    fingerprint = derive_fingerprint_from_inchi("InChI=1S/C2O/c1-2-3", "daylight", 16)
    expected_fingerprint = numpy.array([0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0])
    assert numpy.all(fingerprint == expected_fingerprint), "Expected different fingerprint."


def test_derive_fingerprint_different_types_from_smiles():
    """Test if correct fingerprints are derived from given smiles when using different types."""
    pytest.importorskip("rdkit")

    types = ["daylight", "morgan1", "morgan2", "morgan3"]
    expected_fingerprints = [
        numpy.array([0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0]),
        numpy.array([0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1]),
        numpy.array([0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1]),
        numpy.array([0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1])
    ]

    for i, fingerprint_type in enumerate(types):
        fingerprint = derive_fingerprint_from_smiles("[C+]#C[O-]", fingerprint_type, 16)
        assert numpy.all(fingerprint == expected_fingerprints[i]), "Expected different fingerprint."


def test_missing_rdkit_module_error():
    """Test if different functions return correct error when *rdkit* is not available"""
    if find_spec("rdkit") is not None:
        context = mock.patch.dict(sys.modules, {"rdkit": None})
    else:
        context = nullcontext()

    expected_msg = "Conda package 'rdkit' is required for this functionality."
    with context:
        reload(matchms.utils)
        mol_input = "C[Si](Cn1cncn1)(c1ccc(F)cc1)"
        with pytest.raises(ImportError) as msg:
            _ = matchms.utils.mol_converter(mol_input, "smiles", "inchikey")
        assert expected_msg in str(msg.value), "Expected different ImportError."

        with pytest.raises(ImportError) as msg:
            _ = matchms.utils.is_valid_inchi("test")
        assert expected_msg in str(msg.value), "Expected different ImportError."

        with pytest.raises(ImportError) as msg:
            _ = matchms.utils.is_valid_smiles("test")
        assert expected_msg in str(msg.value), "Expected different ImportError."

        with pytest.raises(ImportError) as msg:
            _ = matchms.utils.derive_fingerprint_from_inchi(mol_input, "test", 0)
        assert expected_msg in str(msg.value), "Expected different ImportError."

        with pytest.raises(ImportError) as msg:
            _ = matchms.utils.derive_fingerprint_from_smiles(mol_input, "test", 0)
        assert expected_msg in str(msg.value), "Expected different ImportError."

        with pytest.raises(ImportError) as msg:
            _ = matchms.utils.mol_to_fingerprint(mol_input, "test", 0)
        assert expected_msg in str(msg.value), "Expected different ImportError."


def test_looks_like_adduct():
    """Test if adducts are correctly identified"""
    for adduct in ["M+", "M*+", "M+Cl", "[M+H]", "[2M+Na]+", "M+H+K", "Cat",
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
