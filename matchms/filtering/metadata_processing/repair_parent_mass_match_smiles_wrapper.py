import logging
from typing import Optional
from matchms.filtering.metadata_processing.repair_adduct_based_on_smiles import \
    repair_adduct_based_on_smiles
from matchms.typing import SpectrumType
from .repair_parent_mass_is_mol_wt import repair_parent_mass_is_mol_wt
from .repair_smiles_of_salts import repair_smiles_of_salts
from .require_parent_mass_match_smiles import \
    _check_smiles_and_parent_mass_match


logger = logging.getLogger("matchms")


def repair_parent_mass_match_smiles_wrapper(spectrum_in: SpectrumType,
                                            mass_tolerance: float = 0.2) -> Optional[SpectrumType]:
    """Wrapper function for repairing a mismatch between parent mass and smiles mass"""
    if spectrum_in is None:
        return None
    spectrum = spectrum_in.clone()

    filters_to_apply = [repair_smiles_of_salts,
                        repair_parent_mass_is_mol_wt,
                        repair_adduct_based_on_smiles,
                        ]
    for filter_function in filters_to_apply:
        if _check_smiles_and_parent_mass_match(smiles=spectrum.get("smiles"),
                                               parent_mass=spectrum.get("parent_mass"),
                                               mass_tolerance=mass_tolerance):
            return spectrum
        spectrum = filter_function(spectrum_in=spectrum,
                                   mass_tolerance=mass_tolerance)
    return spectrum
