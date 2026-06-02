import math
import pandas as pd
import pytest
from matchms import SpectraCollection
from matchms.filtering import repair_parent_mass_from_smiles
from tests.builder_Spectrum import SpectrumBuilder
from tests.run_spectrum_and_collection import run_filter_as_spectrum_or_collection


def assert_missing(value):
    assert value is None or pd.isna(value)


@pytest.mark.parametrize("as_collection", [False, True], ids=["spectrum", "collection"])
@pytest.mark.parametrize(
    "smiles, parent_mass, expected_parent_mass",
    [
        # Parent mass differs from SMILES monoisotopic mass by more than tolerance.
        ("CN1CCCC1C2=CN=CC=C2", 162.23, 162.115698455),
        # Parent mass is already close enough and should remain unchanged.
        ("CN1CCCC1C2=CN=CC=C2", 162.1, 162.1),
    ],
)
def test_repair_parent_mass_from_smiles(
    smiles,
    parent_mass,
    expected_parent_mass,
    as_collection,
):
    spectrum_in = (
        SpectrumBuilder()
        .with_metadata({"smiles": smiles, "parent_mass": parent_mass})
        .build()
    )

    spectrum_out = run_filter_as_spectrum_or_collection(
        repair_parent_mass_from_smiles,
        spectrum_in,
        as_collection,
        mass_tolerance=0.1,
    )

    assert math.isclose(
        spectrum_out.get("parent_mass"),
        expected_parent_mass,
        abs_tol=1e-9,
    )


@pytest.mark.parametrize("as_collection", [False, True], ids=["spectrum", "collection"])
def test_repair_parent_mass_from_smiles_adds_missing_parent_mass(as_collection):
    spectrum_in = (
        SpectrumBuilder()
        .with_metadata({"smiles": "CN1CCCC1C2=CN=CC=C2"})
        .build()
    )

    spectrum_out = run_filter_as_spectrum_or_collection(
        repair_parent_mass_from_smiles,
        spectrum_in,
        as_collection,
        mass_tolerance=0.1,
    )

    assert math.isclose(
        spectrum_out.get("parent_mass"),
        162.115698455,
        abs_tol=1e-9,
    )


@pytest.mark.parametrize("missing_value", [None, pd.NA])
@pytest.mark.parametrize("as_collection", [False, True], ids=["spectrum", "collection"])
def test_repair_parent_mass_from_smiles_handles_missing_parent_mass_values(
    missing_value,
    as_collection,
):
    spectrum_in = (
        SpectrumBuilder()
        .with_metadata(
            {
                "smiles": "CN1CCCC1C2=CN=CC=C2",
                "parent_mass": missing_value,
            }
        )
        .build()
    )

    spectrum_out = run_filter_as_spectrum_or_collection(
        repair_parent_mass_from_smiles,
        spectrum_in,
        as_collection,
        mass_tolerance=0.1,
    )

    assert math.isclose(
        spectrum_out.get("parent_mass"),
        162.115698455,
        abs_tol=1e-9,
    )


@pytest.mark.parametrize("as_collection", [False, True], ids=["spectrum", "collection"])
def test_repair_parent_mass_from_smiles_invalid_smiles_does_nothing(as_collection):
    spectrum_in = (
        SpectrumBuilder()
        .with_metadata({"smiles": "not-a-smiles", "parent_mass": 123.4})
        .build()
    )

    spectrum_out = run_filter_as_spectrum_or_collection(
        repair_parent_mass_from_smiles,
        spectrum_in,
        as_collection,
        mass_tolerance=0.1,
    )

    assert spectrum_out.get("smiles") == "not-a-smiles"
    assert spectrum_out.get("parent_mass") == 123.4


@pytest.mark.parametrize("as_collection", [False, True], ids=["spectrum", "collection"])
def test_repair_parent_mass_from_smiles_missing_smiles_does_nothing(as_collection):
    spectrum_in = (
        SpectrumBuilder()
        .with_metadata({"parent_mass": 123.4})
        .build()
    )

    spectrum_out = run_filter_as_spectrum_or_collection(
        repair_parent_mass_from_smiles,
        spectrum_in,
        as_collection,
        mass_tolerance=0.1,
    )

    assert spectrum_out.get("parent_mass") == 123.4


@pytest.mark.parametrize("as_collection", [False, True], ids=["spectrum", "collection"])
def test_repair_parent_mass_from_smiles_missing_smiles_and_parent_mass_does_nothing(
    as_collection,
):
    spectrum_in = SpectrumBuilder().with_metadata({}).build()

    spectrum_out = run_filter_as_spectrum_or_collection(
        repair_parent_mass_from_smiles,
        spectrum_in,
        as_collection,
        mass_tolerance=0.1,
    )

    assert_missing(spectrum_out.get("parent_mass"))


def test_repair_parent_mass_from_smiles_none_input():
    assert repair_parent_mass_from_smiles(None, mass_tolerance=0.1) is None


def test_repair_parent_mass_from_smiles_clone_true_does_not_modify_input_spectrum():
    spectrum_in = (
        SpectrumBuilder()
        .with_metadata(
            {
                "smiles": "CN1CCCC1C2=CN=CC=C2",
                "parent_mass": 162.23,
            }
        )
        .build()
    )

    spectrum_out = repair_parent_mass_from_smiles(
        spectrum_in,
        mass_tolerance=0.1,
        clone=True,
    )

    assert spectrum_out is not spectrum_in
    assert math.isclose(spectrum_out.get("parent_mass"), 162.115698455, abs_tol=1e-9)
    assert spectrum_in.get("parent_mass") == 162.23


def test_repair_parent_mass_from_smiles_clone_false_modifies_input_spectrum():
    spectrum_in = (
        SpectrumBuilder()
        .with_metadata(
            {
                "smiles": "CN1CCCC1C2=CN=CC=C2",
                "parent_mass": 162.23,
            }
        )
        .build()
    )

    spectrum_out = repair_parent_mass_from_smiles(
        spectrum_in,
        mass_tolerance=0.1,
        clone=False,
    )

    assert spectrum_out is spectrum_in
    assert math.isclose(spectrum_in.get("parent_mass"), 162.115698455, abs_tol=1e-9)


def test_repair_parent_mass_from_smiles_collection_updates_only_applicable_rows():
    spectra = [
        SpectrumBuilder()
        .with_metadata(
            {
                "id": "repair",
                "smiles": "CN1CCCC1C2=CN=CC=C2",
                "parent_mass": 162.23,
            }
        )
        .build(),
        SpectrumBuilder()
        .with_metadata(
            {
                "id": "keep",
                "smiles": "CN1CCCC1C2=CN=CC=C2",
                "parent_mass": 162.1,
            }
        )
        .build(),
        SpectrumBuilder()
        .with_metadata(
            {
                "id": "invalid",
                "smiles": "not-a-smiles",
                "parent_mass": 123.4,
            }
        )
        .build(),
    ]
    collection = SpectraCollection(spectra)

    processed = repair_parent_mass_from_smiles(
        collection,
        mass_tolerance=0.1,
    )

    assert isinstance(processed, SpectraCollection)
    assert len(processed) == 3

    assert processed.metadata.loc[0, "id"] == "repair"
    assert math.isclose(
        processed.metadata.loc[0, "parent_mass"],
        162.115698455,
        abs_tol=1e-9,
    )

    assert processed.metadata.loc[1, "id"] == "keep"
    assert processed.metadata.loc[1, "parent_mass"] == 162.1

    assert processed.metadata.loc[2, "id"] == "invalid"
    assert processed.metadata.loc[2, "parent_mass"] == 123.4


def test_repair_parent_mass_from_smiles_collection_clone_true_does_not_modify_input():
    collection = SpectraCollection(
        [
            SpectrumBuilder()
            .with_metadata(
                {
                    "smiles": "CN1CCCC1C2=CN=CC=C2",
                    "parent_mass": 162.23,
                }
            )
            .build()
        ]
    )

    processed = repair_parent_mass_from_smiles(
        collection,
        mass_tolerance=0.1,
        clone=True,
    )

    assert processed is not collection
    assert math.isclose(
        processed.metadata.loc[0, "parent_mass"],
        162.115698455,
        abs_tol=1e-9,
    )
    assert collection.metadata.loc[0, "parent_mass"] == 162.23


def test_repair_parent_mass_from_smiles_collection_clone_false_modifies_input():
    collection = SpectraCollection(
        [
            SpectrumBuilder()
            .with_metadata(
                {
                    "smiles": "CN1CCCC1C2=CN=CC=C2",
                    "parent_mass": 162.23,
                }
            )
            .build()
        ]
    )

    processed = repair_parent_mass_from_smiles(
        collection,
        mass_tolerance=0.1,
        clone=False,
    )

    assert processed is collection
    assert math.isclose(
        collection.metadata.loc[0, "parent_mass"],
        162.115698455,
        abs_tol=1e-9,
    )
    