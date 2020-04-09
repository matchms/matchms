def set_ionmode_na_when_missing(spectrum):

    # Set ionmode to "n/a" when ionmode is missing from the metadata
    if "ionmode" not in spectrum.metadata:
        spectrum.metadata["ionmode"] = "n/a"
