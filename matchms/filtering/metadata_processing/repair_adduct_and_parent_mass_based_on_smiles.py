import logging
from matchms import Spectrum
from matchms.filtering.filter_utils.get_neutral_mass_from_smiles import get_monoisotopic_neutral_mass
from ..filter_utils.derive_precursor_mz_and_parent_mass import derive_parent_mass_from_precursor_mz
from .repair_adduct_based_on_parent_mass import _get_matching_adduct


logger = logging.getLogger("matchms")


def repair_adduct_and_parent_mass_based_on_smiles(spectrum_in: Spectrum, mass_tolerance: float):
    """
    Corrects the adduct and parent mass of a spectrum based on its SMILES representation and the precursor m/z.

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
    """
    if spectrum_in is None:
        return None
    changed_spectrum = spectrum_in.clone()
    smiles_mass = get_monoisotopic_neutral_mass(changed_spectrum.get("smiles"))
    if smiles_mass is None:
        return spectrum_in
    parent_mass = spectrum_in.get("parent_mass")

    # First check if the given adduct and precursor mz already match the monoisotopic mass of the smiles
    estimated_parent_mass = derive_parent_mass_from_precursor_mz(changed_spectrum, estimate_from_adduct=True, estimate_from_charge=False)
    need_to_update_adduct = False
    if estimated_parent_mass is not None:
        if abs(estimated_parent_mass - smiles_mass) > mass_tolerance:
            need_to_update_adduct = True
    else:
        need_to_update_adduct = True

    if need_to_update_adduct:
        # Otherwise check if any of the common adducts matches the smiles mass
        new_adduct = _get_matching_adduct(
            precursor_mz=spectrum_in.get("precursor_mz"), parent_mass=smiles_mass, ion_mode=spectrum_in.get("ionmode"), mass_tolerance=mass_tolerance
        )
        if new_adduct is None:
            return spectrum_in

        changed_spectrum.set("adduct", new_adduct)
        logger.info("Adduct was set from %s to %s", spectrum_in.get("adduct"), new_adduct)

    # if no parent_mass is set always overwrite
    if parent_mass is None:
        changed_spectrum.set("parent_mass", smiles_mass)
        logger.info("Parent mass was set to match the smiles mass: %s", smiles_mass)
    # Only overwrite if the mass difference is too large
    elif abs(smiles_mass - parent_mass) > mass_tolerance:
        changed_spectrum.set("parent_mass", smiles_mass)
        logger.info("Parent mass was updated from %s to %s to match the smiles mass", parent_mass, smiles_mass)
    return changed_spectrum
