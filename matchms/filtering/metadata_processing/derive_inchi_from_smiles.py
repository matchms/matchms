from matchms.typing import SpectrumType
from matchms.filtering.filters.derive_inchi_from_smiles import DeriveInchiFromSmiles


def derive_inchi_from_smiles(spectrum_in: SpectrumType) -> SpectrumType:
    """Find missing Inchi and derive from smiles where possible."""

    spectrum = DeriveInchiFromSmiles().process(spectrum_in)
    return spectrum