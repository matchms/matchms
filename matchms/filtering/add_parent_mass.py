from ..constants import PROTON_MASS
from ..typing import SpectrumType


def add_parent_mass(spectrum_in: SpectrumType) -> SpectrumType:
    """Add parentmass to metadata (if not present yet).

    Method to calculate the parent mass from given precursor m/z
    and charge. Will take precursor m/z from either "precursor_mz"
    or "pepmass" field.
    """
    if spectrum_in is None:
        return None

    spectrum = spectrum_in.clone()

    if spectrum.get("parent_mass", None) is None:
        try:
            charge = spectrum.get("charge")
            protons_mass = PROTON_MASS * charge
            precursor_mz = spectrum.get("precursor_mz", None)
            if precursor_mz is None:
                precursor_mz = spectrum.get("pepmass")[0]
            precursor_mass = precursor_mz * abs(charge)
            parent_mass = precursor_mass - protons_mass
            if parent_mass:
                spectrum.set("parent_mass", parent_mass)
        except KeyError:
            print("Not sufficient spectrum metadata to derive parent mass.")

    return spectrum
