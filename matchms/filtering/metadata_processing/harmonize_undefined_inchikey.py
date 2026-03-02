from typing import List, Optional
from matchms.typing import SpectrumType


def harmonize_undefined_inchikey(
    spectrum_in: SpectrumType, undefined: str = "", aliases: List[str] = None, clone: Optional[bool] = True
) -> Optional[SpectrumType]:
    """Replace all aliases for empty/undefined inchikey entries by ``undefined``.

    Parameters
    ----------
    spectrum_in:
        Input spectrum.
    undefined:
        Give desired entry for undefined inchikey fields. Default is "".
    aliases:
        Enter list of strings that are expected to represent undefined entries.
        Default is ["", "N/A", "NA", "n/a", "no data"].
    clone:
        Optionally clone the Spectrum.

    Returns
    -------
    Spectrum or None
        Spectrum with undefined INCHIKEY if not present or N/A, or `None` if not present.
    """
    if spectrum_in is None:
        return None

    spectrum = spectrum_in.clone() if clone else spectrum_in

    if aliases is None:
        aliases = ["", "N/A", "NA", "n/a", "no data"]

    inchikey = spectrum.get("inchikey")
    if inchikey is None:
        # spectrum does not have an "inchikey" key in its metadata
        spectrum.set("inchikey", undefined)
        return spectrum

    if inchikey in aliases:
        # harmonize aliases for undefined values
        spectrum.set("inchikey", undefined)

    return spectrum
