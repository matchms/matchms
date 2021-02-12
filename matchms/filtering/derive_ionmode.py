from ..importing import load_adducts_dict
from ..typing import SpectrumType
from ..utils import clean_adduct


def derive_ionmode(spectrum_in: SpectrumType) -> SpectrumType:
    """Derive missing ionmode based on adduct.

    Some input formates (e.g. MGF files) do not always provide a correct ionmode.
    This function reads the adduct from the metadata and uses this to fill in the
    correct ionmode where missing.

    Parameters
    ----------
    spectrum:
        Input spectrum.

    Returns:
    --------
    Returns Spectrum object with `ionmode` attribute set.
    """

    if spectrum_in is None:
        return None

    spectrum = spectrum_in.clone()

    # Load lists of known adducts
    known_adducts = load_adducts_dict()

    adduct = spectrum.get("adduct", None)
    # Harmonize adduct string
    if adduct:
        adduct = clean_adduct(adduct)

    ionmode = spectrum.get("ionmode")
    if ionmode:
        assert ionmode == ionmode.lower(), ("Ionmode field not harmonized.",
                                            "Apply 'make_ionmode_lowercase' filter first.")
    if ionmode in ["positive", "negative"]:
        return spectrum

    # Try completing missing or incorrect ionmodes
    if adduct in known_adducts:
        ionmode = known_adducts[adduct]["ionmode"]
    else:
        ionmode = "n/a"

    spectrum.set("ionmode", ionmode)

    return spectrum
