import pandas as pd
import pytest
from matchms import SpectraCollection
from matchms.filtering import repair_smiles_of_salts
from tests.builder_Spectrum import SpectrumBuilder
from tests.run_spectrum_and_collection import run_filter_as_spectrum_or_collection


@pytest.mark.parametrize("as_collection", [False, True], ids=["spectrum", "collection"])
@pytest.mark.parametrize(
    "smiles, parent_mass, expected_smiles, expected_salt_ions",
    [
        # First part is correct.
        ("C1=NC2=NC=NC(=C2N1)N.Cl", 135.054, "C1=NC2=NC=NC(=C2N1)N", "Cl"),
        # Last part is correct.
        ("C(C(=O)O).C(C(=O)O)C(CC(=O)O)(C(=O)O)O", 192.027, "C(C(=O)O)C(CC(=O)O)(C(=O)O)O", "C(C(=O)O)"),
        # Not a salt.
        ("C(C(=O)O)C(CC(=O)O)(C(=O)O)O", 192.027, "C(C(=O)O)C(CC(=O)O)(C(=O)O)O", None),
        # All parts are incorrect.
        ("C(C(=O)O).C(C(=O)O)C(CC(=O)O)(C(=O)O)O", 150.0, "C(C(=O)O).C(C(=O)O)C(CC(=O)O)(C(=O)O)O", None),
        # Salt with > 3 parts.
        ("C(C(=O)O).C(C(=O)O)C(CC(=O)O)(C(=O)O)O.Cl", 192.027, "C(C(=O)O)C(CC(=O)O)(C(=O)O)O", "C(C(=O)O).Cl"),
        # Salt matching a combination of 2 parts.
        ("C(C(=O)O).C(C(=O)O)C(CC(=O)O)(C(=O)O)O.Cl", 228.0, "C(C(=O)O)C(CC(=O)O)(C(=O)O)O.Cl", "C(C(=O)O)"),
    ],
)
def test_repair_smiles_of_salts(
    smiles,
    parent_mass,
    expected_smiles,
    expected_salt_ions,
    as_collection,
):
    spectrum_in = (
        SpectrumBuilder()
        .with_metadata({"smiles": smiles, "parent_mass": parent_mass})
        .build()
    )

    spectrum = run_filter_as_spectrum_or_collection(
        repair_smiles_of_salts,
        spectrum_in,
        as_collection,
        mass_tolerance=0.1,
    )

    assert spectrum.get("smiles") == expected_smiles
    assert spectrum.get("salt_ions") == expected_salt_ions


@pytest.mark.parametrize("as_collection", [False, True], ids=["spectrum", "collection"])
def test_repair_smiles_of_salts_missing_smiles_does_nothing(as_collection):
    spectrum_in = (
        SpectrumBuilder()
        .with_metadata({"parent_mass": 192.027})
        .build()
    )

    spectrum = run_filter_as_spectrum_or_collection(
        repair_smiles_of_salts,
        spectrum_in,
        as_collection,
        mass_tolerance=0.1,
    )

    assert spectrum.get("smiles") is None
    assert spectrum.get("salt_ions") is None


@pytest.mark.parametrize("as_collection", [False, True], ids=["spectrum", "collection"])
def test_repair_smiles_of_salts_missing_parent_mass_does_nothing(as_collection):
    smiles = "C1=NC2=NC=NC(=C2N1)N.Cl"
    spectrum_in = SpectrumBuilder().with_metadata({"smiles": smiles}).build()

    spectrum = run_filter_as_spectrum_or_collection(
        repair_smiles_of_salts,
        spectrum_in,
        as_collection,
        mass_tolerance=0.1,
    )

    assert spectrum.get("smiles") == smiles
    assert spectrum.get("salt_ions") is None


def test_repair_smiles_of_salts_collection_updates_only_matching_rows():
    spectra = [
        SpectrumBuilder()
        .with_metadata(
            {
                "smiles": "C1=NC2=NC=NC(=C2N1)N.Cl",
                "parent_mass": 135.054,
            }
        )
        .build(),
        SpectrumBuilder()
        .with_metadata(
            {
                "smiles": "C(C(=O)O)C(CC(=O)O)(C(=O)O)O",
                "parent_mass": 192.027,
            }
        )
        .build(),
        SpectrumBuilder()
        .with_metadata(
            {
                "smiles": "C(C(=O)O).C(C(=O)O)C(CC(=O)O)(C(=O)O)O",
                "parent_mass": 150.0,
            }
        )
        .build(),
    ]
    collection = SpectraCollection(spectra)

    repaired = repair_smiles_of_salts(collection, mass_tolerance=0.1)

    assert repaired is not collection
    assert len(repaired) == 3

    assert repaired.metadata.loc[0, "smiles"] == "C1=NC2=NC=NC(=C2N1)N"
    assert repaired.metadata.loc[0, "salt_ions"] == "Cl"

    assert repaired.metadata.loc[1, "smiles"] == "C(C(=O)O)C(CC(=O)O)(C(=O)O)O"
    assert "salt_ions" not in repaired.metadata.columns or pd.isna(repaired.metadata.loc[1, "salt_ions"])

    assert repaired.metadata.loc[2, "smiles"] == "C(C(=O)O).C(C(=O)O)C(CC(=O)O)(C(=O)O)O"
    assert "salt_ions" not in repaired.metadata.columns or pd.isna(repaired.metadata.loc[2, "salt_ions"])


def test_repair_smiles_of_salts_collection_clone_false_modifies_input():
    collection = SpectraCollection(
        [
            SpectrumBuilder()
            .with_metadata(
                {
                    "smiles": "C1=NC2=NC=NC(=C2N1)N.Cl",
                    "parent_mass": 135.054,
                }
            )
            .build()
        ]
    )

    repaired = repair_smiles_of_salts(
        collection,
        mass_tolerance=0.1,
        clone=False,
    )

    assert repaired is collection
    assert collection.metadata.loc[0, "smiles"] == "C1=NC2=NC=NC(=C2N1)N"
    assert collection.metadata.loc[0, "salt_ions"] == "Cl"
