from matchms.data import load_known_adducts


def derive_ionmode(spectrum):
    """Derive missing ionmode based on adduct.

    MGF files do not always provide a correct ionmode. This function reads
    the adduct from the metadata and uses this to fill in the correct ionmode
    where missing.

    Args:
    ----
    spectrum: matchms.Spectrum.Spectrum()
        Input spectrum.
    """

    # Load lists of known adducts
    known_adducts = load_known_adducts()

    adduct = spectrum.metadata.get("adduct", None)

    ionmode = spectrum.metadata["ionmode"]

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
    spectrum.metadata["ionmode"] = ionmode
