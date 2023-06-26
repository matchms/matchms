import logging
from matchms import Spectrum
from matchms.filtering import repair_parent_mass_is_mol_wt
from matchms.filtering.repair_adduct.clean_adduct import load_adducts_dict
from matchms.filtering.filter_utils.get_monoisotopic_neutral_mass import get_monoisotopic_neutral_mass

logger = logging.getLogger("matchms")


def repair_adduct_based_on_smiles(spectrum_in: Spectrum,
                                  mass_tolerance,
                                  accept_parent_mass_is_mol_wt):
    """If the parent mass is wrong due to a wrong of is derived from the precursor mz
    To do this the charge and adduct are used"""
    if spectrum_in is None:
        return None
    spectrum = spectrum_in.clone()

    precursor_mz = spectrum.get("precursor_mz")
    ion_mode = spectrum.get("ionmode")
    adducts_dict = load_adducts_dict()
    if ion_mode not in ("positive", "negative"):
        logger.warning(f"Ionmode: {ion_mode} not positive or negative, first run derive_ionmode")
        return spectrum
    if precursor_mz is None:
        logger.warning("Precursor_mz is None, first run add_precursor_mz")
        return spectrum

    smiles_mass = get_monoisotopic_neutral_mass(spectrum.get("smiles"))
    for adduct_name in adducts_dict:
        adduct_info = adducts_dict[adduct_name]
        if adduct_info['ionmode'] == ion_mode and \
            adduct_info["correction_mass"] is not None and \
            adduct_info["mass_multiplier"] is not None:
            multiplier = adduct_info["mass_multiplier"]
            correction_mass = adduct_info["correction_mass"]
            parent_mass = (precursor_mz - correction_mass) / multiplier
            if abs(parent_mass - smiles_mass) < 1:
                spectrum_with_corrected_adduct = spectrum.clone()
                spectrum_with_corrected_adduct.set("parent_mass", parent_mass)
                spectrum_with_corrected_adduct.set("adduct", adduct_name)
                if accept_parent_mass_is_mol_wt:
                    spectrum_with_corrected_adduct = repair_parent_mass_is_mol_wt(spectrum_with_corrected_adduct,
                                                                                  mass_tolerance)
                if abs(spectrum_with_corrected_adduct.get("parent_mass") - smiles_mass) < mass_tolerance:
                    logger.info(f"Adduct was set from {spectrum.get('adduct')} to {adduct_name}")
                    return spectrum_with_corrected_adduct
    return spectrum
