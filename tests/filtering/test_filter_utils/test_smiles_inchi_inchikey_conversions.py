from importlib import reload
import pytest
import matchms.filtering.filter_utils.smile_inchi_inchikey_conversions
import matchms.filtering.metadata_processing.add_fingerprint
from matchms.filtering.filter_utils.smile_inchi_inchikey_conversions import is_valid_inchi, is_valid_inchikey, is_valid_smiles, mol_converter


@pytest.fixture()
def reload_metadata_utils():
    """Reload metadata_utils module after test has finished."""
    yield
    reload(matchms.filtering.filter_utils.smile_inchi_inchikey_conversions)


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
        "XYLJNLCSTIOKRM-aaaaaaaaaa-a",
    ]

    for inchikey in inchikeys_true:
        assert is_valid_inchikey(inchikey), "Expected inchikey is True."
    for inchikey in inchikeys_false:
        assert not is_valid_inchikey(inchikey), "Expected inchikey is False."


def test_is_valid_inchikey_none_input():
    """Test None entry."""
    assert not is_valid_inchikey(None), "Expected None entry to give False."


@pytest.mark.parametrize(
    "inchi, expected",
    [
        ["InChI=1S/C2H7N3/c1-5-2(3)4/h1H3,(H4,3,4,5)", True],
        ['"InChI=1S/C2H7N3/c1-5-2(3)4/h1H3,(H4,3,4,5)"', True],
        ["InChI=1S/Ne", True],
        ["InChI=1S/C14H9Cl5/c15-11-5-1-9(2-6-11)13(14(17,18)19)10-3-7-12(16)8-4-10/h1-8,13H", True],
        ["1S/C2H7N3/c1-5-2(3)4/h1H3,(H4,3,4,5)", False],
        ["InChI=1S/C2H7N3/c152(3)4/h1H3,(H4,3,4,5)", False],
        ["InChI=C2H7N3/c1-5-2(3)4/h1H3,(H4,3,4,5)", False],
        ["InChI=1S/C2H7N3/c1-5-2(3)", False],
    ],
)
def test_is_valid_inchi(inchi, expected):
    pytest.importorskip("rdkit")
    assert is_valid_inchi(inchi) == expected


def test_is_valid_inchi_none_input():
    """Test None entry."""
    pytest.importorskip("rdkit")
    assert not is_valid_inchi(None), "Expected None entry to give False."


@pytest.mark.parametrize(
    "smiles, expected",
    [
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
        [r"1S/C2H7N3/c1-5-2(3)4", False],
    ],
)
def test_is_valid_smiles(smiles, expected):
    pytest.importorskip("rdkit")
    assert is_valid_smiles(smiles) == expected


def test_is_valid_smiles_none_input():
    """Test None entry."""
    pytest.importorskip("rdkit")

    assert not is_valid_smiles(None), "Expected None entry to give False."
