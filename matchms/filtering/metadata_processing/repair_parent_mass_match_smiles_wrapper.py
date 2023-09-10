from typing import Optional
from matchms.typing import SpectrumType
from matchms.filtering.filters.repair_parent_mass_match_smiles_wrapper import RepairParentMassMatchSmilesWrapper


def repair_parent_mass_match_smiles_wrapper(spectrum_in: SpectrumType, mass_tolerance: float = 0.2) -> Optional[SpectrumType]:
    """Wrapper function for repairing a mismatch between parent mass and smiles mass"""

    spectrum = RepairParentMassMatchSmilesWrapper(mass_tolerance).process(spectrum_in)
    return spectrum
