from matchms.constants import PROTON_MASS


def add_parent_mass(spectrum_in):
    """Add parentmass to metadata (if not present yet).

    Method to calculate the parent mass from given precursor mass
    and charge.
    """
    if spectrum_in is None:
        return None

    spectrum = spectrum_in.clone()

    if spectrum.get("parent_mass", None) is None:
        try:
            charge = spectrum.get("charge")
            precursor_mass = spectrum.get("pepmass")[0]
            parent_mass = precursor_mass * abs(int_charge)
            parent_mass -= charge * PROTON_MASS
            if parent_mass:
                spectrum.set("parent_mass", parent_mass)
        except KeyError:
            print("Not sufficient spectrum metadata to derive parent mass.")

    return spectrum
