def add_parent_mass(spectrum_in):
    """Add parentmass to metadata (if not present yet).

    Method to calculate the parent mass from given precursor mass
    and charge.
    """
    proton_mass = 1.00727645199076  # TODO: where to put this constant?
    spectrum = spectrum_in.clone()

    if spectrum.get("parent_mass", None) is None:
        try:
            int_charge = int(spectrum.get("charge"))
            precursor_mass = spectrum.get("pepmass")[0]
            parent_mass = precursor_mass * abs(int_charge)
            parent_mass -= int_charge * proton_mass
            if parent_mass:
                spectrum.set("parent_mass", parent_mass)
        except KeyError:
            print("Not sufficient spectrum metadata to derive parent mass.")

    return spectrum
