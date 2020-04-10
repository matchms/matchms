def make_ionmode_lowercase(spectrum_in):

    spectrum = spectrum_in.clone()

    # if the ionmode key exists in the metadata, lowercase its value
    if spectrum.get("ionmode") is not None:
        spectrum.set("ionmode", spectrum.get("ionmode").lower())

    return spectrum
