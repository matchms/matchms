def make_ionmode_lowercase(spectrum_in):

    spectrum = spectrum_in.clone()

    # if the ionmode key exists in the metadata, lowercase its value
    if "ionmode" in spectrum.metadata:
        spectrum.metadata["ionmode"] = spectrum.metadata["ionmode"].lower()

    return spectrum_in
