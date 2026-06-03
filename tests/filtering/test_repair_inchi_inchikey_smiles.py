import pytest
from matchms import SpectraCollection
from matchms.filtering import repair_inchi_inchikey_smiles
from tests.run_spectrum_and_collection import run_filter_as_spectrum_or_collection
from ..builder_Spectrum import SpectrumBuilder


REPAIR_TEST_CASES = [
    [
        {"inchi": "InChI=1/C2H4N4/c3-2-4-1-5-6-2/h1H,(H3,3,4,5,6)/f/h6H,3H2"},
        "InChI=1/C2H4N4/c3-2-4-1-5-6-2/h1H,(H3,3,4,5,6)/f/h6H,3H2",
        "",
        "",
    ],
    [
        {"inchikey": "InChI=1/C2H4N4/c3-2-4-1-5-6-2/h1H,(H3,3,4,5,6)/f/h6H,3H2"},
        "InChI=1/C2H4N4/c3-2-4-1-5-6-2/h1H,(H3,3,4,5,6)/f/h6H,3H2",
        "",
        "",
    ],
    [
        {"smiles": "InChI=1/C2H4N4/c3-2-4-1-5-6-2/h1H,(H3,3,4,5,6)/f/h6H,3H2"},
        "InChI=1/C2H4N4/c3-2-4-1-5-6-2/h1H,(H3,3,4,5,6)/f/h6H,3H2",
        "",
        "",
    ],
    [{"inchi": "ABTNALLHJFCFRZ-UHFFFAOYSA-N"}, "", "ABTNALLHJFCFRZ-UHFFFAOYSA-N", ""],
    [{"inchikey": "ABTNALLHJFCFRZ-UHFFFAOYSA-N"}, "", "ABTNALLHJFCFRZ-UHFFFAOYSA-N", ""],
    [{"smiles": "ABTNALLHJFCFRZ-UHFFFAOYSA-N"}, "", "ABTNALLHJFCFRZ-UHFFFAOYSA-N", ""],
    [{"inchi": "C[C@H](Cc1ccccc1)N(C)CC#C"}, "", "", "C[C@H](Cc1ccccc1)N(C)CC#C"],
    [{"inchikey": "C[C@H](Cc1ccccc1)N(C)CC#C"}, "", "", "C[C@H](Cc1ccccc1)N(C)CC#C"],
    [{"smiles": "C[C@H](Cc1ccccc1)N(C)CC#C"}, "", "", "C[C@H](Cc1ccccc1)N(C)CC#C"],
]


TEST_INCHIS = [
    "1S/C4H11N5.ClH/c1-7-3(5)9-4(6)8-2;/h1-2H3,(H5,5,6,7,8,9);1H",
    "InChI=1S/C11H15N3O2.ClH/c1-12-11(15)16-10-6-4-5-9(7-10)13-8-14(2)3;/h4-8H,1-3H3,(H,12,15);1H/b13-8+;",
    '"InChI=1S/C17O8/c1-9-7-12(19)14(16(20)21)13(8-9)25-15(10(2)23-3)11(5-6-18)17(22)24-4"',
    "InChI=1S/CH3/h1H3",
    "1/2C17H18N3O3S.Mg/c2*1-10-8-18-15(11(2)16(10)23-4)9-24(21)17-19-13-6-5-12(22-3)7-14(13)20-17;/h2*5-8H,9H2,1-4H3;/q2*-1;+2",
]


@pytest.mark.parametrize("as_collection", [False, True], ids=["spectrum", "collection"])
@pytest.mark.parametrize(
    "metadata, expected_inchi, expected_inchikey, expected_smiles",
    REPAIR_TEST_CASES,
)
def test_repair_inchi_inchikey_smiles(
    metadata,
    expected_inchi,
    expected_inchikey,
    expected_smiles,
    as_collection,
):
    spectrum_in = SpectrumBuilder().with_metadata(metadata).build()

    spectrum = run_filter_as_spectrum_or_collection(
        repair_inchi_inchikey_smiles,
        spectrum_in,
        as_collection,
    )

    assert spectrum.get("inchi") == expected_inchi
    assert spectrum.get("inchikey") == expected_inchikey
    assert spectrum.get("smiles") == expected_smiles


@pytest.mark.parametrize("as_collection", [False, True], ids=["spectrum", "collection"])
def test_repair_inchi_inchikey_smiles_various_inchi_entered_as_smiles(as_collection):
    """Test a wider variety of different inchis."""
    for inchi in TEST_INCHIS:
        spectrum_in = SpectrumBuilder().with_metadata({"smiles": inchi}).build()

        spectrum = run_filter_as_spectrum_or_collection(
            repair_inchi_inchikey_smiles,
            spectrum_in,
            as_collection,
        )

        assert spectrum.get("inchi") == "InChI=" + inchi.replace("InChI=", "").replace('"', "")
        assert spectrum.get("inchikey") == ""
        assert spectrum.get("smiles") == ""


def test_repair_inchi_inchikey_smiles_collection_multiple_rows():
    spectra = [
        SpectrumBuilder().with_metadata({"smiles": "ABTNALLHJFCFRZ-UHFFFAOYSA-N"}).build(),
        SpectrumBuilder().with_metadata({"inchi": "C[C@H](Cc1ccccc1)N(C)CC#C"}).build(),
        SpectrumBuilder().with_metadata({"smiles": TEST_INCHIS[0]}).build(),
    ]
    collection = SpectraCollection(spectra)

    repaired = repair_inchi_inchikey_smiles(collection)

    assert repaired is not collection
    assert len(repaired) == 3

    assert repaired.metadata.loc[0, "inchi"] == ""
    assert repaired.metadata.loc[0, "inchikey"] == "ABTNALLHJFCFRZ-UHFFFAOYSA-N"
    assert repaired.metadata.loc[0, "smiles"] == ""

    assert repaired.metadata.loc[1, "inchi"] == ""
    assert repaired.metadata.loc[1, "inchikey"] == ""
    assert repaired.metadata.loc[1, "smiles"] == "C[C@H](Cc1ccccc1)N(C)CC#C"

    assert repaired.metadata.loc[2, "inchi"] == "InChI=" + TEST_INCHIS[0].replace("InChI=", "").replace('"', "")
    assert repaired.metadata.loc[2, "inchikey"] == ""
    assert repaired.metadata.loc[2, "smiles"] == ""


def test_repair_inchi_inchikey_smiles_clone_false_modifies_collection_in_place():
    collection = SpectraCollection(
        [
            SpectrumBuilder()
            .with_metadata({"smiles": "ABTNALLHJFCFRZ-UHFFFAOYSA-N"})
            .build()
        ]
    )

    repaired = repair_inchi_inchikey_smiles(collection, clone=False)

    assert repaired is collection
    assert collection.metadata.loc[0, "inchikey"] == "ABTNALLHJFCFRZ-UHFFFAOYSA-N"


def test_repair_inchi_inchikey_smiles_empty_spectrum():
    spectrum = repair_inchi_inchikey_smiles(None)

    assert spectrum is None, "Expected different handling of None spectrum."
