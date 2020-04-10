def make_charge_scalar(spectrum_in):
    spectrum = spectrum_in.clone()

    # Avoid pyteomics ChargeList
    if isinstance(spectrum.metadata.get("charge", None), list):
        spectrum.metadata["charge"] = int(spectrum.metadata["charge"][0])

    return spectrum
