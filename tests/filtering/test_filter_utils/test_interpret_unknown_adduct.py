import pytest
from matchms.filtering.filter_utils.interpret_unknown_adduct import get_multiplier_and_mass_from_adduct
from matchms.filtering.filter_utils.load_known_adducts import load_known_adducts


def test_get_multiplier_and_mass():
    """Test if correct dict is imported."""
    pytest.importorskip("rdkit")
    known_adducts = load_known_adducts()
    for adduct in list(known_adducts["adduct"]):
        multiplier, correction_mass = get_multiplier_and_mass_from_adduct(adduct)
        exp_multiplier = known_adducts.loc[known_adducts["adduct"] == adduct, "mass_multiplier"].values[0]
        exp_corr_mass = known_adducts.loc[known_adducts["adduct"] == adduct, "correction_mass"].values[0]
        assert round(exp_multiplier, 5) == round(multiplier, 5), (
            f"The calculated multiplier: {multiplier} does not match the multiplier in the table: {exp_multiplier} for the adduct: {adduct}"
        )
        assert round(correction_mass, 4) == round(exp_corr_mass, 4), (
            f"The calculated correction mass: {correction_mass} does not match the correction mass in the table: {exp_corr_mass} for the adduct: {adduct}"
        )
