from matchms.importing import load_adducts


def derive_ionmode(spectrum_in):
    """Derive missing ionmode based on adduct.

    MGF files do not always provide a correct ionmode. This function reads
    the adduct from the metadata and uses this to fill in the correct ionmode
    where missing.

    Args:
    ----
    spectrum: matchms.Spectrum.Spectrum()
        Input spectrum.
    """

    spectrum = spectrum_in.clone()

    # Load lists of known adducts
    known_adducts = load_adducts()

    adduct = spectrum.get("adduct", None)

    ionmode = spectrum.get("ionmode")

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
