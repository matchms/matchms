import logging
from matchms import Spectrum
from ..filter_utils.load_known_adducts import load_known_adducts


logger = logging.getLogger("matchms")


def repair_adduct_based_on_parent_mass(spectrum_in: Spectrum,
                                       mass_tolerance: float):
    """
    Corrects the adduct of a spectrum based on its parent_mass representation and the precursor m/z.

    Parameters:
    ----------
    spectrum_in : Spectrum
        The input spectrum whose adduct needs to be repaired.

    mass_tolerance : float
        Maximum allowed mass difference between the parent mass and the parent mass based on the adduct.
    """
    if spectrum_in is None:
        return None
    changed_spectrum = spectrum_in.clone()

    precursor_mz = changed_spectrum.get("precursor_mz")
    if precursor_mz is None:
        logger.warning("Precursor_mz is None, first run add_precursor_mz")
        return spectrum_in

    ion_mode = changed_spectrum.get("ionmode")
    if ion_mode not in ("positive", "negative"):
        if ion_mode is not None:
            logger.warning("Ionmode: %s not positive, negative or None, first run derive_ionmode",
                            ion_mode)
        return spectrum_in

    actual_parent_mass = changed_spectrum.get("paretn_mass")
    if actual_parent_mass is None:
        return spectrum_in

    adducts_df = load_known_adducts()
    # Only use the adducts matching the ion mode
    adducts_df = adducts_df[adducts_df["ionmode"] == ion_mode]

    parent_masses = (precursor_mz - adducts_df["correction_mass"]) / adducts_df["mass_multiplier"]
    mass_differences = abs(parent_masses-actual_parent_mass)

    # Select the lowest value
    smallest_mass_index = mass_differences.idxmin()
    adduct = adducts_df.loc[smallest_mass_index]["adduct"]

    if mass_differences[smallest_mass_index] < mass_tolerance:
        # Change spectrum. This spectrum will only be returned if the mass difference is smaller than mass tolerance
        changed_spectrum.set("adduct", adduct)
        logger.info("Adduct was set from %s to %s",
                    spectrum_in.get('adduct'), adduct)
        return changed_spectrum
    return spectrum_in
