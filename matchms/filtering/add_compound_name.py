from ..typing import SpectrumType


def add_compound_name(spectrum_in: SpectrumType) -> SpectrumType:
    """Add compound_name to correct field: "compound_name" in metadata."""
    if spectrum_in is None:
        return None

    spectrum = spectrum_in.clone()

    if spectrum.get("compound_name", None) is None:
        if isinstance(spectrum.get("name", None), str):
            spectrum.set("compound_name", spectrum.get("name"))
            return spectrum

        if isinstance(spectrum.get("title", None), str):
            spectrum.set("compound_name", spectrum.get("title"))
            return spectrum

        print("No compound name found in metadata.")

    return spectrum
