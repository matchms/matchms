from typing import List, Optional
from matchms.typing import SpectrumType
from matchms.utils import ALIASES_FOR_NONE


def harmonize_undefined_inchi(
    spectrum_in: SpectrumType, undefined: str = "", aliases: List[str] = None, clone: Optional[bool] = True
) -> Optional[SpectrumType]:
    """Replace all aliases for empty/undefined inchi entries by value of ``undefined`` argument.

    Parameters
    ----------
    spectrum_in:
        Input spectrum.
    undefined:
        Give desired entry for undefined inchi fields. Default is "".
    aliases:
        Enter list of strings that are expected to represent undefined entries.
        Default is ["", "N/A", "NA", "n/a"].
    clone:
        Optionally clone the Spectrum.

    Returns
    -------
    Spectrum or None
        Spectrum with undefined INCHI if not present or N/A, or `None` if not present.
    """
    if spectrum_in is None:
        return None

    spectrum = spectrum_in.clone() if clone else spectrum_in

    if aliases is None:
        aliases = ALIASES_FOR_NONE

    inchi = spectrum.get("inchi")
    if inchi is None:
        # spectrum does not have an "inchi" key in its metadata
        spectrum.set("inchi", undefined)
        return spectrum

    if inchi in aliases:
        # harmonize aliases for undefined values
        spectrum.set("inchi", undefined)

    return spectrum
