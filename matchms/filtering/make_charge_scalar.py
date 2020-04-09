def make_charge_scalar(spectrum):

    # Avoid pyteomics ChargeList
    if isinstance(spectrum.metadata.get("charge", None), list):
        spectrum.metadata["charge"] = int(spectrum.metadata["charge"][0])
