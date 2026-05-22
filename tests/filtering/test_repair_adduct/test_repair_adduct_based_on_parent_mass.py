import pytest
from matchms.filtering.metadata_processing.repair_adduct_and_parent_mass_based_on_smiles import (
    repair_adduct_and_parent_mass_based_on_smiles,
)
from tests.builder_Spectrum import SpectrumBuilder
from tests.run_spectrum_and_collection import run_filter_as_spectrum_or_collection


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
        # Should not be repaired as [M]+, since this could also be a mistake with the precursor mz
        # being the parent mass
        (16.04, None, "positive"),
        # Should not be repaired as [M]-, since this could also be a mistake with the precursor mz
        # being the parent mass
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
    pytest.importorskip("rdkit")

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

    assert spectrum_out.get("adduct") == expected_adduct
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
        # Parent mass is incorrect, but no adduct is available
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
    pytest.importorskip("rdkit")

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

    assert spectrum_out.get("adduct") == expected_adduct
    assert spectrum_out.get("parent_mass") == expected_parent_mass
