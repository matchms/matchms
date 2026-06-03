import pytest
from matchms import SpectraCollection
from matchms.filtering.metadata_processing.require_matching_adduct_precursor_mz_parent_mass import (
    require_matching_adduct_precursor_mz_parent_mass,
)
from ..builder_Spectrum import SpectrumBuilder


TEST_CASES = [
    ["[M+H]+", 100, 100, True],
    ["[M+H]+blabla", 100, 100, True],
    ["[M+H]+", "blabal", 100, True],
    ["[M+H]+", 100, "bla", True],
    ["[M+H]+", 100, None, True],
    ["[M+H]+", 100.0, 101.0, False],
]


@pytest.mark.parametrize(
    "adduct, parent_mass, precursor_mz, should_be_removed",
    TEST_CASES,
)
def test_require_matching_adduct_precursor_mz_parent_mass_spectrum(
    adduct, parent_mass, precursor_mz, should_be_removed
):
    spectrum_in = (
        SpectrumBuilder()
        .with_metadata(
            {
                "adduct": adduct,
                "parent_mass": parent_mass,
                "precursor_mz": precursor_mz,
            }
        )
        .build()
    )

    result = require_matching_adduct_precursor_mz_parent_mass(spectrum_in)

    if should_be_removed:
        assert result is None
    else:
        assert result == spectrum_in


@pytest.mark.parametrize(
    "adduct, parent_mass, precursor_mz, should_be_removed",
    TEST_CASES,
)
def test_require_matching_adduct_precursor_mz_parent_mass_collection_single_row(
    adduct, parent_mass, precursor_mz, should_be_removed
):
    spectrum_in = (
        SpectrumBuilder()
        .with_metadata(
            {
                "adduct": adduct,
                "parent_mass": parent_mass,
                "precursor_mz": precursor_mz,
            }
        )
        .build()
    )
    collection = SpectraCollection([spectrum_in])

    result = require_matching_adduct_precursor_mz_parent_mass(collection)

    assert result is not collection

    if should_be_removed:
        assert len(result) == 0
    else:
        assert len(result) == 1
        assert result[0] == spectrum_in


# ----------------------
# Multi-row collection level tests

def test_require_matching_adduct_precursor_mz_parent_mass_collection_multiple_rows():
    spectra = [
        SpectrumBuilder()
        .with_metadata({"adduct": "[M+H]+", "parent_mass": 100.0, "precursor_mz": 101.0})
        .build(),
        SpectrumBuilder()
        .with_metadata({"adduct": "[M+H]+", "parent_mass": 100.0, "precursor_mz": 100.0})
        .build(),
        SpectrumBuilder()
        .with_metadata({"adduct": "[M+H]+", "parent_mass": "blabal", "precursor_mz": 100.0})
        .build(),
        SpectrumBuilder()
        .with_metadata({"adduct": "[M-H]-", "parent_mass": 100.0, "precursor_mz": 98.9927})
        .build(),
    ]
    collection = SpectraCollection(spectra)

    filtered = require_matching_adduct_precursor_mz_parent_mass(collection)

    assert filtered is not collection
    assert len(filtered) == 2
    assert filtered.metadata["adduct"].tolist() == ["[M+H]+", "[M-H]-"]


def test_require_matching_adduct_precursor_mz_parent_mass_collection_clone_false_modifies_input():
    collection = SpectraCollection(
        [
            SpectrumBuilder()
            .with_metadata({"adduct": "[M+H]+", "parent_mass": 100.0, "precursor_mz": 101.0})
            .build(),
            SpectrumBuilder()
            .with_metadata({"adduct": "[M+H]+", "parent_mass": 100.0, "precursor_mz": 100.0})
            .build(),
        ]
    )

    filtered = require_matching_adduct_precursor_mz_parent_mass(collection, clone=False)

    assert filtered is collection
    assert len(collection) == 1
    assert collection.metadata.loc[0, "adduct"] == "[M+H]+"
    assert collection.metadata.loc[0, "parent_mass"] == pytest.approx(100.0)
    assert collection.metadata.loc[0, "precursor_mz"] == pytest.approx(101.0)