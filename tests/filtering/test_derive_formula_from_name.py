import pytest
from matchms import SpectraCollection
from matchms.filtering import derive_formula_from_name
from tests.run_spectrum_and_collection import run_filter_as_spectrum_or_collection
from ..builder_Spectrum import SpectrumBuilder


@pytest.mark.parametrize("as_collection", [False, True], ids=["spectrum", "collection"])
@pytest.mark.parametrize(
    "metadata, remove_formula_from_name, expected_formula, expected_compound_name",
    [
        [{"compound_name": "peptideXYZ [M+H+K] C5H12NO2"}, True, "C5H12NO2", "peptideXYZ [M+H+K]"],
        [
            {"compound_name": "peptideXYZ C5H12NO2", "formula": "totallycorrectformula"},
            True,
            "totallycorrectformula",
            "peptideXYZ",
        ],
        [{"compound_name": "peptideXYZ [M+H+K] C5H12NO2"}, False, "C5H12NO2", "peptideXYZ [M+H+K] C5H12NO2"],
        [{}, True, None, None],
    ],
)
def test_derive_formula_from_name(
    metadata,
    remove_formula_from_name,
    expected_formula,
    expected_compound_name,
    as_collection,
):
    spectrum_in = SpectrumBuilder().with_metadata(metadata).build()

    spectrum = run_filter_as_spectrum_or_collection(
        derive_formula_from_name,
        spectrum_in,
        as_collection,
        remove_formula_from_name=remove_formula_from_name,
    )

    assert spectrum.get("formula") == expected_formula, "Expected different formula."
    assert spectrum.get("compound_name") == expected_compound_name, "Expected different cleaned name."


@pytest.mark.parametrize("as_collection", [False, True], ids=["spectrum", "collection"])
@pytest.mark.parametrize(
    "string_addition, expected_formula",
    [
        ("C6H14NO2", "C6H14NO2"),
        ("C47H83N1O8P1", "C47H83N1O8P1"),
        ("HYPOTAURINE", None),
        ("CITRATE", None),
        ("NIST14", None),
        ("HCl", None),
        ("ACID", None),
        ("B12A13", None),
        ("(12)", None),
        ("6432", None),
        ("C15", None),
    ],
)
def test_derive_formula_from_name_examples(
    string_addition,
    expected_formula,
    as_collection,
):
    spectrum_in = (
        SpectrumBuilder()
        .with_metadata({"compound_name": "peptideXYZ [M+H+K] " + string_addition})
        .build()
    )

    spectrum = run_filter_as_spectrum_or_collection(
        derive_formula_from_name,
        spectrum_in,
        as_collection,
    )

    assert spectrum.get("formula") == expected_formula, "Expected different formula."


def test_derive_formula_from_name_collection_multiple_rows():
    collection = SpectraCollection(
        [
            SpectrumBuilder().with_metadata({"compound_name": "peptideXYZ C5H12NO2"}).build(),
            SpectrumBuilder().with_metadata({"compound_name": "plain name"}).build(),
            SpectrumBuilder().with_metadata({"compound_name": "lipid C47H83N1O8P1"}).build(),
        ]
    )

    processed = derive_formula_from_name(collection)

    assert processed is not collection

    assert processed.metadata.loc[0, "formula"] == "C5H12NO2"
    assert processed.metadata.loc[0, "compound_name"] == "peptideXYZ"

    assert processed.metadata.loc[1, "compound_name"] == "plain name"

    assert processed.metadata.loc[2, "formula"] == "C47H83N1O8P1"
    assert processed.metadata.loc[2, "compound_name"] == "lipid"


def test_derive_formula_from_name_collection_clone_false_modifies_input():
    collection = SpectraCollection(
        [
            SpectrumBuilder().with_metadata({"compound_name": "peptideXYZ C5H12NO2"}).build(),
        ]
    )

    processed = derive_formula_from_name(collection, clone=False)

    assert processed is collection
    assert collection.metadata.loc[0, "formula"] == "C5H12NO2"
    assert collection.metadata.loc[0, "compound_name"] == "peptideXYZ"


def test_derive_formula_from_name_empty_spectrum():
    assert derive_formula_from_name(None) is None