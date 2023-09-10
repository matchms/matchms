from matchms.typing import SpectrumType
from matchms.filtering.filters.derive_adduct_from_name import DeriveAdductFromName


def derive_adduct_from_name(spectrum_in: SpectrumType,
                            remove_adduct_from_name: bool = True) -> SpectrumType:
    """Find adduct in compound name and add to metadata (if not present yet).

    Method to interpret the given compound name to find the adduct.

    Parameters
    ----------
    spectrum_in:
        Input spectrum.
    remove_adduct_from_name:
        Remove found adducts from compound name if set to True. Default is True.
    """

    spectrum = DeriveAdductFromName(remove_adduct_from_name).process(spectrum_in)
    return spectrum
