def make_ionmode_lowercase(spectrum):

    # if the ionmode key exists in the metadata, lowercase its value
    if "ionmode" in spectrum.metadata:
        spectrum.metadata["ionmode"] = spectrum.metadata["ionmode"].lower()
