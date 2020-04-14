def add_parent_mass(spectrum_in):
    """Add parentmass to metadata (if not present yet).

    Method to calculate the parent mass from given precursor mass
    and charge.
    """
    PROTON_MASS = 1.00727645199076  # TODO: where to put this constant?
    spectrum = spectrum_in.clone()

    if spectrum.get("parent_mass", None) is None:
        try:
            int_charge = int(spectrum.metadata["charge"])
            precursor_mass = spectrum.metadata["pepmass"][0]
            parent_mass = precursor_mass * abs(int_charge)
            parent_mass -= int_charge * PROTON_MASS
        except KeyError:
            print("Not sufficient spectrum metadata to derive parent mass.")

    return spectrum
