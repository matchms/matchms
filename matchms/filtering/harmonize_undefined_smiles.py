from ..typing import SpectrumType


def harmonize_undefined_smiles(spectrum_in: SpectrumType, undefined="", aliases=None):

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

    smiles = spectrum.get("smiles")
    if smiles is None:
        # spectrum does not have a "smiles" key in its metadata
        spectrum.set("smiles", undefined)
        return spectrum

    if smiles in aliases:
        # harmonize aliases for undefined values
        spectrum.set("smiles", undefined)

    return spectrum
