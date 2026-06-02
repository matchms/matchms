import pytest
from matchms import SpectraCollection
from matchms.filtering import clean_compound_name
from tests.run_spectrum_and_collection import run_filter_as_spectrum_or_collection
from ..builder_Spectrum import SpectrumBuilder


@pytest.mark.parametrize("as_collection", [False, True], ids=["spectrum", "collection"])
@pytest.mark.parametrize(
    "name, expected",
    [
        ["MLS000863588-01!2-methoxy-3-methyl-9H-carbazole", "2-methoxy-3-methyl-9H-carbazole"],
        ["NCGC00160217-01!SOPHOCARPINE", "SOPHOCARPINE"],
        ["0072_2-Mercaptobenzothiaz", "2-Mercaptobenzothiaz"],
        [r"MassbankEU:ET110206 NPE_327.1704_12.2|N-succinylnorpheniramine", "N-succinylnorpheniramine"],
        ["Massbank:CE000307 Trans-Zeatin-[d5]", "Trans-Zeatin-[d5]"],
        ["HMDB:HMDB00500-718 4-Hydroxybenzoic acid", "4-Hydroxybenzoic acid"],
        ["MoNA:2346734 Piroxicam (Feldene)", "Piroxicam (Feldene)"],
        ["ReSpect:PS013405 option1|option2|option3", "option3"],
        ["ReSpect:PS013405 option1name", "option1name"],
        ["4,4-Dimethylcholest-8(9),24-dien-3.beta.-ol  231.2", "4,4-Dimethylcholest-8(9),24-dien-3.beta.-ol"],
        ["", ""],
    ],
)
def test_clean_compound_name(name, expected, as_collection):
    spectrum_in = SpectrumBuilder().with_metadata({"compound_name": name}).build()

    spectrum = run_filter_as_spectrum_or_collection(
        clean_compound_name,
        spectrum_in,
        as_collection,
    )

    assert spectrum.get("compound_name") == expected, "Expected different cleaned name."


@pytest.mark.parametrize("as_collection", [False, True], ids=["spectrum", "collection"])
def test_clean_compound_name_without_compound_name_and_without_name_does_nothing(as_collection):
    spectrum_in = SpectrumBuilder().with_metadata({"other": "value"}).build()

    spectrum = run_filter_as_spectrum_or_collection(
        clean_compound_name,
        spectrum_in,
        as_collection,
    )

    assert spectrum.get("compound_name") is None


def test_clean_compound_name_collection_multiple_rows():
    collection = SpectraCollection(
        [
            SpectrumBuilder()
            .with_metadata({"compound_name": "NCGC00160217-01!SOPHOCARPINE"})
            .build(),
            SpectrumBuilder()
            .with_metadata({"compound_name": "MoNA:2346734 Piroxicam (Feldene)"})
            .build(),
            SpectrumBuilder()
            .with_metadata({"compound_name": "Already clean"})
            .build(),
        ]
    )

    processed = clean_compound_name(collection)

    assert processed is not collection
    assert processed.metadata.loc[0, "compound_name"] == "SOPHOCARPINE"
    assert processed.metadata.loc[1, "compound_name"] == "Piroxicam (Feldene)"
    assert processed.metadata.loc[2, "compound_name"] == "Already clean"


def test_clean_compound_name_collection_clone_false_modifies_input():
    collection = SpectraCollection(
        [
            SpectrumBuilder()
            .with_metadata({"compound_name": "NCGC00160217-01!SOPHOCARPINE"})
            .build()
        ]
    )

    processed = clean_compound_name(collection, clone=False)

    assert processed is collection
    assert collection.metadata.loc[0, "compound_name"] == "SOPHOCARPINE"


def test_clean_compound_name_empty_spectrum():
    assert clean_compound_name(None) is None