import pandas as pd
import pytest
from matchms import SpectraCollection
from matchms.filtering.metadata_processing.repair_adduct_based_on_parent_mass import (
    repair_adduct_based_on_parent_mass,
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
    "precursor_mz, parent_mass, expected_adduct, ionmode",
    [
        (17.0, 16.03, "[M+H]+", "positive"),
        (17.5, 16.03, "[M+H+NH4]2+", "positive"),
        (74.0, 16.03, "[2M+ACN+H]+", "positive"),
        (15.0, 16.03, "[M-H]-", "negative"),
        (51.0, 16.03, "[M+Cl]-", "negative"),
        (4.33333, 16.03, "[M-3H]3-", "negative"),
        (1000, 16.03, None, "negative"),  # should not be repaired
    ],
)
def test_repair_adduct_based_on_parent_mass(
    precursor_mz,
    parent_mass,
    expected_adduct,
    ionmode,
    as_collection,
):
    spectrum_in = (
        SpectrumBuilder()
        .with_metadata(
            {
                "precursor_mz": precursor_mz,
                "parent_mass": parent_mass,
                "ionmode": ionmode,
            }
        )
        .build()
    )

    spectrum_out = run_filter_as_spectrum_or_collection(
        repair_adduct_based_on_parent_mass,
        spectrum_in,
        as_collection,
        mass_tolerance=0.1,
    )

    assert_metadata_value(spectrum_out.get("adduct"), expected_adduct)
    assert abs(spectrum_out.get("parent_mass") - 15.9589) < 0.1


@pytest.mark.parametrize("as_collection", [False, True], ids=["spectrum", "collection"])
@pytest.mark.parametrize(
    "metadata",
    [
        {"parent_mass": 16.03, "ionmode": "positive"},  # missing precursor_mz
        {"precursor_mz": 17.0, "ionmode": "positive"},  # missing parent_mass
        {"precursor_mz": 17.0, "parent_mass": 16.03},  # missing ionmode
        {"precursor_mz": 17.0, "parent_mass": 16.03, "ionmode": "unknown"},  # invalid ionmode
    ],
)
def test_repair_adduct_based_on_parent_mass_missing_or_invalid_required_metadata(
    metadata,
    as_collection,
):
    spectrum_in = SpectrumBuilder().with_metadata(metadata).build()

    spectrum_out = run_filter_as_spectrum_or_collection(
        repair_adduct_based_on_parent_mass,
        spectrum_in,
        as_collection,
        mass_tolerance=0.1,
    )

    assert_metadata_value(spectrum_out.get("adduct"), None)


@pytest.mark.parametrize("as_collection", [False, True], ids=["spectrum", "collection"])
def test_repair_adduct_based_on_parent_mass_does_not_change_existing_adduct_if_no_match(as_collection):
    spectrum_in = (
        SpectrumBuilder()
        .with_metadata(
            {
                "precursor_mz": 1000,
                "parent_mass": 16.03,
                "ionmode": "negative",
                "adduct": "[M-H]-",
            }
        )
        .build()
    )

    spectrum_out = run_filter_as_spectrum_or_collection(
        repair_adduct_based_on_parent_mass,
        spectrum_in,
        as_collection,
        mass_tolerance=0.1,
    )

    assert spectrum_out.get("adduct") == "[M-H]-"


def test_repair_adduct_based_on_parent_mass_none_input():
    assert repair_adduct_based_on_parent_mass(None, mass_tolerance=0.1) is None


def test_repair_adduct_based_on_parent_mass_collection_multiple_rows():
    spectra = [
        SpectrumBuilder()
        .with_metadata(
            {
                "id": "positive",
                "precursor_mz": 17.0,
                "parent_mass": 16.03,
                "ionmode": "positive",
            }
        )
        .build(),
        SpectrumBuilder()
        .with_metadata(
            {
                "id": "negative",
                "precursor_mz": 15.0,
                "parent_mass": 16.03,
                "ionmode": "negative",
            }
        )
        .build(),
        SpectrumBuilder()
        .with_metadata(
            {
                "id": "unchanged",
                "precursor_mz": 1000,
                "parent_mass": 16.03,
                "ionmode": "negative",
            }
        )
        .build(),
    ]

    collection = SpectraCollection(spectra)
    processed = repair_adduct_based_on_parent_mass(
        collection,
        mass_tolerance=0.1,
    )

    assert isinstance(processed, SpectraCollection)
    assert len(processed) == 3

    assert processed.metadata.loc[0, "id"] == "positive"
    assert processed.metadata.loc[0, "adduct"] == "[M+H]+"

    assert processed.metadata.loc[1, "id"] == "negative"
    assert processed.metadata.loc[1, "adduct"] == "[M-H]-"

    assert processed.metadata.loc[2, "id"] == "unchanged"
    assert pd.isna(processed.metadata.loc[2, "adduct"])
