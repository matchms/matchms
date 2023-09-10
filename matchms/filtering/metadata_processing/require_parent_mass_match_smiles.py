from typing import Optional
from matchms.typing import SpectrumType
from matchms.filtering.filters.require_parent_mass_match_smiles import RequireParentMassMatchSmiles


def require_parent_mass_match_smiles(spectrum_in: SpectrumType,
                                     mass_tolerance) -> Optional[SpectrumType]:
    """
    Validates if the parent mass of the given spectrum matches the mass calculated
    from its associated SMILES string within a specified tolerance.

    Parameters
    ----------
    spectrum_in: Spectrum
        The input spectrum to be validated. If `None`, the function will return `None`.

    mass_tolerance: float
        The tolerance for the mass difference between the spectrum's parent mass and
        the mass calculated from its SMILES string.

    Returns
    -------
    Spectrum or None
        The validated spectrum if its parent mass matches the SMILES mass within the
        specified tolerance, or `None` otherwise.
    """

    spectrum = RequireParentMassMatchSmiles(mass_tolerance).process(spectrum_in)
    return spectrum
