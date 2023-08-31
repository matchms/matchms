import logging
from matchms import Spectrum
from matchms.filtering.filter_utils.get_neutral_mass_from_smiles import \
    get_monoisotopic_neutral_mass
from ..filter_utils.load_known_adducts import load_known_adducts
from .repair_parent_mass_is_mol_wt import repair_parent_mass_is_mol_wt


logger = logging.getLogger("matchms")


def repair_adduct_based_on_smiles(spectrum_in: Spectrum,
                                  mass_tolerance: float,
                                  accept_parent_mass_is_mol_wt: bool = True):
    """
    Corrects the adduct of a spectrum based on its SMILES representation and the precursor m/z.

    Given a spectrum, this function tries to match the spectrum's parent mass, derived from its
    precursor m/z and known adducts, to the neutral monoisotopic mass of the molecule derived
    from its SMILES representation. If a match is found within a given mass tolerance, the
    adduct and parent mass of the spectrum are updated.

    Parameters:
    ----------
    spectrum_in : Spectrum
        The input spectrum whose adduct needs to be repaired.

    mass_tolerance : float
        Maximum allowed mass difference between the calculated parent mass and the neutral
        monoisotopic mass derived from the SMILES.

    accept_parent_mass_is_mol_wt : bool, optional (default=True)
        Allows the function to attempt repairing the spectrum's parent mass by assuming it
        represents the molecule's weight. If True, further checks and corrections are made
        using `repair_parent_mass_is_mol_wt`.
    """
    if spectrum_in is None:
        return None
    changed_spectrum = spectrum_in.clone()

    precursor_mz = changed_spectrum.get("precursor_mz")
    ion_mode = changed_spectrum.get("ionmode")
    if ion_mode not in ("positive", "negative"):
        logger.warning("Ionmode: %s not positive or negative, first run derive_ionmode",
                        ion_mode)
        return changed_spectrum
    if precursor_mz is None:
        logger.warning("Precursor_mz is None, first run add_precursor_mz")
        return changed_spectrum

    adducts_df = load_known_adducts()
    smiles_mass = get_monoisotopic_neutral_mass(changed_spectrum.get("smiles"))
    if smiles_mass is None:
        return changed_spectrum
    parent_masses = (precursor_mz - adducts_df["correction_mass"]) / adducts_df["mass_multiplier"]
    mass_differences = abs(parent_masses-smiles_mass)

    # Select the lowest value
    smalles_mass_index = mass_differences.idxmin()
    parent_mass = parent_masses[smalles_mass_index]
    adduct = adducts_df.iloc[smalles_mass_index]["adduct"]
    # Change spectrum. This spectrum will only be returned if the mass difference is smaller than mass tolerance
    changed_spectrum.set("parent_mass", parent_mass)
    changed_spectrum.set("adduct", adduct)
    if mass_differences[smalles_mass_index] < mass_tolerance:
        logger.info("Adduct was set from %s to %s",
                    spectrum_in.get('adduct'), adduct)
        return changed_spectrum
    if accept_parent_mass_is_mol_wt:
        changed_spectrum = repair_parent_mass_is_mol_wt(changed_spectrum, mass_tolerance)
        if abs(changed_spectrum.get("parent_mass") - smiles_mass) < mass_tolerance:
            logger.info("Adduct was set from %s to %s",
                        spectrum_in.get('adduct'), adduct)
            return changed_spectrum
    return spectrum_in
