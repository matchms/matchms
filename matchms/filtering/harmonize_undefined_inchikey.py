from ..typing import SpectrumType


def harmonize_undefined_inchikey(spectrum_in: SpectrumType, undefined="", aliases=None):

    if spectrum_in is None:
        return None

    spectrum = spectrum_in.clone()

    if aliases is None:
        aliases = [
            "",
            "N/A",
            "NA",
            "n/a",
            "no data"
        ]

    inchikey = spectrum.get("inchikey")
    if inchikey is None:
        # spectrum does not have an "inchikey" key in its metadata
        spectrum.set("inchikey", undefined)
        return spectrum

    if inchikey in aliases:
        # harmonize aliases for undefined values
        spectrum.set("inchikey", undefined)

    return spectrum
