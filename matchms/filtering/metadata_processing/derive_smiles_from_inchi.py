from matchms.typing import SpectrumType
from matchms.filtering.filters.derive_smiles_from_inchi import DeriveSmilesFromInchi


def derive_smiles_from_inchi(spectrum_in: SpectrumType) -> SpectrumType:
    """Find missing smiles and derive from Inchi where possible."""

    spectrum = DeriveSmilesFromInchi().process(spectrum_in)
    return spectrum