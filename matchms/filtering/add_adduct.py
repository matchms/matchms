from ..typing import SpectrumType
from ..utils import looks_like_adduct


def add_adduct(spectrum_in: SpectrumType) -> SpectrumType:
    """Find adduct in compound name and add to metadata (if not present yet).

    Method to interpret the given compound name to find the adduct.
    """
    if spectrum_in is None:
        return None

    spectrum = spectrum_in.clone()

    if spectrum.get("name", None):
        name = spectrum.get("name")
    else:
        name = spectrum.get("compound_name", None)

    # Name in metadata, but no adduct
    if name and not spectrum.get("adduct", None):
        adduct = name.split(" ")[-1].strip().replace("*", "")
        if looks_like_adduct(adduct):
            spectrum.set("adduct", adduct)
            print("Added adduct to metadata:", adduct)

    return spectrum
