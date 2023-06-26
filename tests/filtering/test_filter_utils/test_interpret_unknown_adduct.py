import numpy as np
from matchms.filtering.filter_utils.interpret_unknown_adduct import \
    get_multiplier_and_mass_from_adduct
from matchms.filtering.repair_adduct.clean_adduct import load_adducts_dict


def test_get_multiplier_and_mass():
    """Test if correct dict is imported."""
    known_adducts = load_adducts_dict()
    for adduct, adduct_info in known_adducts.items():
        exp_multiplier = adduct_info["mass_multiplier"]
        exp_corr_mass = adduct_info["correction_mass"]
        multiplier, correction_mass = get_multiplier_and_mass_from_adduct(adduct)
        assert exp_multiplier == multiplier
        np.testing.assert_almost_equal(exp_corr_mass, correction_mass, decimal=5)
