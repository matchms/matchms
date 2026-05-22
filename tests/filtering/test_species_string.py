import pytest
from matchms.filtering.SpeciesString import SpeciesString


@pytest.mark.parametrize(
    "dirty, expected_target, expected_cleaned",
    [
        # Missing / empty values
        (None, None, ""),
        ("", None, ""),
        ("n/a", None, ""),
        ("no data", None, ""),

        # InChI
        (
            "InChI=1S/CH4/h1H4",
            "inchi",
            "InChI=1S/CH4/h1H4",
        ),
        (
            "1S/CH4/h1H4",
            "inchi",
            "InChI=1S/CH4/h1H4",
        ),
        (
            '"InChI=1S/CH4/h1H4"',
            "inchi",
            "InChI=1S/CH4/h1H4",
        ),

        # InChIKey
        (
            "VNWKTOKETHGBQD-UHFFFAOYSA-N",
            "inchikey",
            "VNWKTOKETHGBQD-UHFFFAOYSA-N",
        ),
        (
            "prefix VNWKTOKETHGBQD-UHFFFAOYSA-N suffix",
            "inchikey",
            "VNWKTOKETHGBQD-UHFFFAOYSA-N",
        ),

        # SMILES
        (
            "C[C@H](Cc1ccccc1)N(C)CC#C",
            "smiles",
            "C[C@H](Cc1ccccc1)N(C)CC#C",
        ),
        (
            "CCCO",
            "smiles",
            "CCCO",
        ),

        # Invalid / unsupported strings
        (
            "not a valid species string",
            None,
            "",
        ),
        (
            "J123",
            None,
            "",
        ),
    ],
)
def test_species_string_guess_target_and_clean(dirty, expected_target, expected_cleaned):
    species_string = SpeciesString(dirty)

    assert species_string.target == expected_target
    assert species_string.cleaned == expected_cleaned


def test_species_string_handles_none_as_empty_string():
    species_string = SpeciesString(None)

    assert species_string.dirty == ""
    assert species_string.target is None
    assert species_string.cleaned == ""
    assert str(species_string) == ""


def test_species_string_handles_empty_string():
    species_string = SpeciesString("")

    assert species_string.dirty == ""
    assert species_string.target is None
    assert species_string.cleaned == ""
    assert str(species_string) == ""


def test_species_string_string_representation_for_valid_inchi():
    species_string = SpeciesString("InChI=1S/CH4/h1H4")

    assert str(species_string) == "(inchi): InChI=1S/CH4/h1H4"


def test_species_string_string_representation_for_valid_inchikey():
    species_string = SpeciesString("VNWKTOKETHGBQD-UHFFFAOYSA-N")

    assert str(species_string) == "(inchikey): VNWKTOKETHGBQD-UHFFFAOYSA-N"


def test_species_string_string_representation_for_valid_smiles():
    species_string = SpeciesString("CCCCO")

    assert str(species_string) == "(smiles): CCCCO"


def test_species_string_looks_like_an_inchi():
    species_string = SpeciesString("InChI=1S/CH4/h1H4")

    assert species_string.looks_like_an_inchi() is True
    assert species_string.looks_like_an_inchikey() is False


def test_species_string_looks_like_an_inchikey():
    species_string = SpeciesString("VNWKTOKETHGBQD-UHFFFAOYSA-N")

    assert species_string.looks_like_an_inchikey() is True
    assert species_string.looks_like_an_inchi() is False


def test_species_string_looks_like_a_smiles():
    species_string = SpeciesString("CCCCO")

    assert species_string.looks_like_a_smiles() is True
    assert species_string.looks_like_an_inchi() is False
    assert species_string.looks_like_an_inchikey() is False
