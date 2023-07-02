import logging
from typing import Optional
from matchms.filtering.metadata_processing.repair_adduct_based_on_smiles import \
    repair_adduct_based_on_smiles
from matchms.typing import SpectrumType
from .repair_parent_mass_is_mol_wt import repair_parent_mass_is_mol_wt
from .repair_precursor_is_parent_mass import repair_precursor_is_parent_mass
from .repair_smiles_of_salts import repair_smiles_of_salts
from .require_parent_mass_match_smiles import require_parent_mass_match_smiles


logger = logging.getLogger("matchms")


def repair_parent_mass_match_smiles_wrapper(spectrum_in: SpectrumType, mass_tolerance: float = 0.2) -> Optional[SpectrumType]:
    """Wrapper function for repairing a mismatch between parent mass and smiles mass"""
    if spectrum_in is None:
        return None
    spectrum = spectrum_in.clone()

    if require_parent_mass_match_smiles(spectrum, mass_tolerance) is not None:
        return spectrum

    spectrum = repair_smiles_of_salts(spectrum, mass_tolerance)
    if require_parent_mass_match_smiles(spectrum, mass_tolerance) is not None:
        return spectrum

    spectrum = repair_precursor_is_parent_mass(spectrum, mass_tolerance)
    if require_parent_mass_match_smiles(spectrum, mass_tolerance) is not None:
        return spectrum

    spectrum = repair_parent_mass_is_mol_wt(spectrum, mass_tolerance)
    if require_parent_mass_match_smiles(spectrum, mass_tolerance) is not None:
        return spectrum

    spectrum = repair_adduct_based_on_smiles(spectrum, mass_tolerance, accept_parent_mass_is_mol_wt=True)
    if require_parent_mass_match_smiles(spectrum, mass_tolerance) is not None:
        return spectrum
    return None
