from ..typing import SpectrumType


def harmonize_undefined_inchi(spectrum_in: SpectrumType, undefined="", aliases=None):

    if spectrum_in is None:
        return None

    spectrum = spectrum_in.clone()

    if aliases is None:
        aliases = [
            "",
            "N/A",
            "NA",
            "n/a"
        ]

    inchi = spectrum.get("inchi")
    if inchi is None:
        # spectrum does not have an "inchi" key in its metadata
        spectrum.set("inchi", undefined)
        return spectrum

    if inchi in aliases:
        # harmonize aliases for undefined values
        spectrum.set("inchi", undefined)

    return spectrum
