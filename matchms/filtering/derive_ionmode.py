from ..importing import load_adducts
from ..typing import SpectrumType


def derive_ionmode(spectrum_in: SpectrumType, adducts_filename: str = None) -> SpectrumType:
    """Derive missing ionmode based on adduct.

    Some input formates (e.g. MGF files) do not always provide a correct ionmode.
    This function reads the adduct from the metadata and uses this to fill in the
    correct ionmode where missing.

    Parameters
    ----------
    spectrum:
        Input spectrum.
    adducts_filename:
        Load known adducts from file, if filename is given. Default is None.
        Method makes sure that file loading is cached.

    Returns:
    --------

    Returns Spectrum object with `ionmode` attribute set.
    """

    if spectrum_in is None:
        return None

    spectrum = spectrum_in.clone()

    # Load lists of known adducts
    known_adducts = load_adducts(filename=adducts_filename)

    adduct = spectrum.get("adduct", None)
    # Harmonize adduct string
    if adduct:
        adduct = adduct.replace("\n", "") \
                       .replace(" ", "") \
                       .replace("[", "") \
                       .replace("]", "") \
                       .replace("*", "")

    ionmode = spectrum.get("ionmode")
    if ionmode:
        assert ionmode == ionmode.lower(), ("Ionmode field not harmonized.",
                                            "Apply 'make_ionmode_lowercase' filter first.")

    # Try completing missing or incorrect ionmodes
    if ionmode not in ["positive", "negative"]:
        if adduct in known_adducts["adducts_positive"]:
            ionmode = "positive"
            print("Added ionmode '" + ionmode + "' based on adduct: ", adduct)
        elif adduct in known_adducts["adducts_negative"]:
            ionmode = "negative"
            print("Added ionmode '" + ionmode + "' based on adduct: ", adduct)
        else:
            ionmode = "n/a"
    spectrum.set("ionmode", ionmode)

    return spectrum
