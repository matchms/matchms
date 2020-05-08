from matchms.utils import mol_converter
from matchms.utils import is_valid_inchi, is_valid_inchikey, is_valid_smiles


def test_mol_converter_smiles_to_inchi():
    """Test if smiles is correctly converted to inchi."""
    mol_input = "C[Si](Cn1cncn1)(c1ccc(F)cc1)"
    output_inchi = mol_converter(mol_input, "smiles", "inchi")
    assert output_inchi == "InChI=1S/C10H11FN3Si/c1-15(8-14-7-12-6-13-14)10-4-2-9(11)3-5-10/h2-7H,8H2,1H3"


def test_mol_converter_inchi_to_smiles():
    """Test if inchi is correctly converted to smiles."""
    mol_input = "InChI=1S/C10H11FN3Si/c1-15(8-14-7-12-6-13-14)10-4-2-9(11)3-5-10/h2-7H,8H2,1H3"
    output_smiles = mol_converter(mol_input, "inchi", "smiles")
    assert output_smiles == "C[Si](Cn1cncn1)=C1C=C[C](F)C=C1"


def test_mol_converter_smiles_to_inchikey():
    """Test if smiles is correctly converted to inchikey."""
    mol_input = "C[Si](Cn1cncn1)(c1ccc(F)cc1)"
    output_inchikey = mol_converter(mol_input, "smiles", "inchikey")
    assert output_inchikey == "HULABQRTZQYJBQ-UHFFFAOYSA-N"


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


def test_is_valid_inchi():
    """Test if strings are correctly classified."""
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


def test_is_valid_smiles():
    """Test if strings are correctly classified."""
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
