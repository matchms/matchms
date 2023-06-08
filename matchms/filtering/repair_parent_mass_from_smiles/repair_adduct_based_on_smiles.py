import logging
from matchms import Spectrum
from matchms.filtering.repair_parent_mass_from_smiles.require_parent_mass_match_smiles import _mass_diff_within_tolerance
from matchms.filtering.load_adducts import load_adducts_dict

logger = logging.getLogger("matchms")


def repair_adduct_based_on_smiles(spectrum: Spectrum,
                                  mass_tolerance):
    """If the parent mass is wrong due to a wrong of is derived from the precursor mz
    To do this the charge and adduct are used"""
    spectrum = spectrum.clone()
    precursor_mz = spectrum.get("precursor_mz")
    ion_mode = spectrum.get("ionmode")
    adducts_dict = load_adducts_dict()
    if ion_mode not in ("positive", "negative"):
        logger.warning(f"Ionmode: {ion_mode} not positive or negative, first run derive_ionmode")
        return spectrum

    for adduct_name in adducts_dict:

        adduct_info = adducts_dict[adduct_name]
        if adduct_info['ionmode'] == ion_mode:
            multiplier = adduct_info["mass_multiplier"]
            correction_mass = adduct_info["correction_mass"]
            parent_mass = precursor_mz * multiplier - correction_mass
            # todo make more efficient by calculating the expected smiles mass once.
            if _mass_diff_within_tolerance(parent_mass, spectrum.get("smiles"), mass_tolerance):
                spectrum.set("parent_mass", parent_mass)
                spectrum.set("adduct", adduct_name)
                logger.info(f"Adduct was set from to {adduct_name}")
                return spectrum
    return spectrum

