import pandas as pd
import pytest
from matchms import SpectraCollection
from matchms.filtering.metadata_processing.repair_adduct_and_parent_mass_based_on_smiles import (
    repair_adduct_and_parent_mass_based_on_smiles,
)
from tests.builder_Spectrum import SpectrumBuilder
from tests.run_spectrum_and_collection import run_filter_as_spectrum_or_collection


def assert_metadata_value(value, expected):
    if expected is None:
        assert value is None or pd.isna(value)
    else:
        assert value == expected


@pytest.mark.parametrize("as_collection", [False, True], ids=["spectrum", "collection"])
@pytest.mark.parametrize(
    "precursor_mz, expected_adduct, ionmode",
    [
        (17.0, "[M+H]+", "positive"),
        (17.5, "[M+H+NH4]2+", "positive"),
        (74.0, "[2M+ACN+H]+", "positive"),
        (15.0, "[M-H]-", "negative"),
        (51.0, "[M+Cl]-", "negative"),
        (4.33333, "[M-3H]3-", "negative"),
        # should not be fixed
        (1000, None, "negative"),
        # Should not be repaired as [M]+, since this could also be a mistake
        # with the precursor mz being the parent mass.
        (16.04, None, "positive"),
        # Should not be repaired as [M]-, since this could also be a mistake
        # with the precursor mz being the parent mass.
        (16.04, None, "negative"),
        (1000, None, "positive"),
    ],
)
def test_repair_adduct_and_parent_mass_based_on_smiles(
    precursor_mz,
    expected_adduct,
    ionmode,
    as_collection,
):
    spectrum_in = (
        SpectrumBuilder()
        .with_metadata(
            {
                "smiles": "C",
                "precursor_mz": precursor_mz,
                "ionmode": ionmode,
            }
        )
        .build()
    )

    spectrum_out = run_filter_as_spectrum_or_collection(
        repair_adduct_and_parent_mass_based_on_smiles,
        spectrum_in,
        as_collection,
        mass_tolerance=0.1,
    )

    assert_metadata_value(spectrum_out.get("adduct"), expected_adduct)

    if expected_adduct is not None:
        assert abs(spectrum_out.get("parent_mass") - 15.9589) < 0.1


@pytest.mark.parametrize("as_collection", [False, True], ids=["spectrum", "collection"])
@pytest.mark.parametrize(
    "precursor_mz, parent_mass, adduct, expected_parent_mass, expected_adduct",
    [
        # Normal repair
        (17.0, 17.0, "[M+H]+", 16.031300127999998, "[M+H]+"),
        # Parent mass should not be repaired if close enough to smiles mass
        (17.0, 16.0, "[M+H]+", 16.0, "[M+H]+"),
        # Parent mass is incorrect, but no matching adduct is available
        (19.0, 20.0, "[M+H]+", 20.0, "[M+H]+"),
    ],
)
def test_repair_adduct_and_parent_mass_based_on_smiles_correct_parent_mass(
    precursor_mz,
    parent_mass,
    adduct,
    expected_parent_mass,
    expected_adduct,
    as_collection,
):
    spectrum_in = (
        SpectrumBuilder()
        .with_metadata(
            {
                "smiles": "C",
                "adduct": adduct,
                "precursor_mz": precursor_mz,
                "parent_mass": parent_mass,
                "ionmode": "positive",
            }
        )
        .build()
    )

    spectrum_out = run_filter_as_spectrum_or_collection(
        repair_adduct_and_parent_mass_based_on_smiles,
        spectrum_in,
        as_collection,
        mass_tolerance=0.1,
    )

    assert_metadata_value(spectrum_out.get("adduct"), expected_adduct)
    assert spectrum_out.get("parent_mass") == pytest.approx(expected_parent_mass)


@pytest.mark.parametrize("as_collection", [False, True], ids=["spectrum", "collection"])
def test_repair_adduct_and_parent_mass_based_on_smiles_invalid_smiles_does_nothing(as_collection):
    spectrum_in = (
        SpectrumBuilder()
        .with_metadata(
            {
                "smiles": "not-a-smiles",
                "precursor_mz": 17.0,
                "parent_mass": 123.4,
                "adduct": "[M+H]+",
                "ionmode": "positive",
            }
        )
        .build()
    )

    spectrum_out = run_filter_as_spectrum_or_collection(
        repair_adduct_and_parent_mass_based_on_smiles,
        spectrum_in,
        as_collection,
        mass_tolerance=0.1,
    )

    assert spectrum_out.get("smiles") == "not-a-smiles"
    assert spectrum_out.get("precursor_mz") == 17.0
    assert spectrum_out.get("parent_mass") == 123.4
    assert spectrum_out.get("adduct") == "[M+H]+"


def test_repair_adduct_and_parent_mass_based_on_smiles_none_input():
    assert repair_adduct_and_parent_mass_based_on_smiles(None, mass_tolerance=0.1) is None


def test_repair_adduct_and_parent_mass_based_on_smiles_collection_multiple_rows():
    spectra = [
        SpectrumBuilder()
        .with_metadata(
            {
                "id": "repair",
                "smiles": "C",
                "precursor_mz": 17.0,
                "ionmode": "positive",
            }
        )
        .build(),
        SpectrumBuilder()
        .with_metadata(
            {
                "id": "unchanged",
                "smiles": "not-a-smiles",
                "precursor_mz": 17.0,
                "parent_mass": 123.4,
                "adduct": "[M+H]+",
                "ionmode": "positive",
            }
        )
        .build(),
    ]

    collection = SpectraCollection(spectra)
    processed = repair_adduct_and_parent_mass_based_on_smiles(
        collection,
        mass_tolerance=0.1,
    )

    assert isinstance(processed, SpectraCollection)
    assert len(processed) == 2

    assert processed.metadata.loc[0, "id"] == "repair"
    assert processed.metadata.loc[0, "adduct"] == "[M+H]+"
    assert processed.metadata.loc[0, "parent_mass"] == pytest.approx(16.031300127999998)

    assert processed.metadata.loc[1, "id"] == "unchanged"
    assert processed.metadata.loc[1, "adduct"] == "[M+H]+"
    assert processed.metadata.loc[1, "parent_mass"] == 123.4
