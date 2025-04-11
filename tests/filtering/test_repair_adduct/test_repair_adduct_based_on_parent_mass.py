import pytest
from matchms.filtering.metadata_processing.repair_adduct_based_on_parent_mass import repair_adduct_based_on_parent_mass
from tests.builder_Spectrum import SpectrumBuilder


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
def test_repair_adduct_based_on_parent_mass(precursor_mz, parent_mass, expected_adduct, ionmode):
    pytest.importorskip("rdkit")

    # CH4 is used as smiles, this has a mass of 16
    spectrum_in = SpectrumBuilder().with_metadata({"precursor_mz": precursor_mz, "parent_mass": parent_mass, "ionmode": ionmode}).build()
    spectrum_out = repair_adduct_based_on_parent_mass(spectrum_in, mass_tolerance=0.1)
    assert spectrum_out.get("adduct") == expected_adduct
    assert abs(spectrum_out.get("parent_mass") - 15.9589) < 0.1
