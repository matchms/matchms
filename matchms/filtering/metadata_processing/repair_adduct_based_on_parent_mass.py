import logging
from typing import Optional
from matchms import Spectrum
from matchms.typing import SpectrumType
from ..filter_utils.load_known_adducts import load_known_adducts


logger = logging.getLogger("matchms")


def repair_adduct_based_on_parent_mass(
    spectrum_in: Spectrum, mass_tolerance: float, clone: Optional[bool] = True
) -> Optional[SpectrumType]:
    """
    Corrects the adduct of a spectrum based on its parent_mass representation and the precursor m/z.

    Parameters:
    ----------
    spectrum_in : Spectrum
        The input spectrum whose adduct needs to be repaired.

    mass_tolerance : float
        Maximum allowed mass difference between the parent mass and the parent mass based on the adduct.

    clone:
        Optionally clone the Spectrum.

    Returns
    -------
    Spectrum or None
        Spectrum with repaired parent adduct, or `None` if not present.
    """
    if spectrum_in is None:
        return None
    changed_spectrum = spectrum_in.clone() if clone else spectrum_in

    new_adduct = _get_matching_adduct(
        precursor_mz=spectrum_in.get("precursor_mz"),
        parent_mass=spectrum_in.get("parent_mass"),
        ion_mode=spectrum_in.get("ionmode"),
        mass_tolerance=mass_tolerance,
    )
    if new_adduct is None:
        return spectrum_in

    changed_spectrum.set("adduct", new_adduct)
    logger.info("Adduct was set from %s to %s", spectrum_in.get("adduct"), new_adduct)
    return changed_spectrum


def _get_matching_adduct(precursor_mz, parent_mass, ion_mode, mass_tolerance):
    if precursor_mz is None:
        logger.warning("Precursor_mz is None, first run add_precursor_mz")
        return None

    if ion_mode not in ("positive", "negative"):
        if ion_mode is not None:
            logger.warning("Ionmode: %s not positive, negative or None, first run derive_ionmode", ion_mode)
        return None

    if parent_mass is None:
        return None

    adducts_df = load_known_adducts()
    # Only use the adducts matching the ion mode
    adducts_df = adducts_df[adducts_df["ionmode"] == ion_mode]

    # M+ and M- should not be used, since these could accidentally repair cases, where the parent mass is filled in
    #   instead of the precursor_mz. Since we cannot differentiate between the two options, we won't repair them.
    adducts_df = adducts_df[~adducts_df["adduct"].isin(("[M]+", "[M]-"))]

    parent_masses = (precursor_mz - adducts_df["correction_mass"]) / adducts_df["mass_multiplier"]
    mass_differences = abs(parent_masses - parent_mass)

    # Select the lowest value
    smallest_mass_index = mass_differences.idxmin()
    adduct = adducts_df.loc[smallest_mass_index]["adduct"]

    if mass_differences[smallest_mass_index] < mass_tolerance:
        return adduct
    return None
