from matchms.typing import SpectrumType
from matchms.filtering.filters.derive_ionmode import DeriveIonmode


def derive_ionmode(spectrum_in: SpectrumType) -> SpectrumType:
    """Derive missing ionmode based on adduct.

    Some input formates (e.g. MGF files) do not always provide a correct ionmode.
    This function reads the adduct from the metadata and uses this to fill in the
    correct ionmode where missing.

    Parameters
    ----------
    spectrum
        Input spectrum.

    Returns
    -------
    Spectrum object with `ionmode` attribute set.
    """

    spectrum = DeriveIonmode().process(spectrum_in)
    return spectrum
