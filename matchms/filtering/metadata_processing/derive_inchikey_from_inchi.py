from matchms.typing import SpectrumType
from matchms.filtering.filters.derive_inchikey_from_inchi import DeriveInchikeyFromInchi


def derive_inchikey_from_inchi(spectrum_in: SpectrumType) -> SpectrumType:
    """Find missing InchiKey and derive from Inchi where possible."""

    spectrum = DeriveInchikeyFromInchi().process(spectrum_in)
    return spectrum