import logging
from matchms.typing import SpectrumType
from matchms.filtering.filters.add_fingerprint import AddFingerprint


logger = logging.getLogger("matchms")


def add_fingerprint(spectrum_in: SpectrumType, fingerprint_type: str = "daylight",
                    nbits: int = 2048) -> SpectrumType:
    """Add molecular finterprint to spectrum.

    If smiles or inchi present in metadata, derive a molecular finterprint and
    add it to the spectrum.

    Parameters
    ----------
    spectrum_in:
        Input spectrum.
    fingerprint_type:
        Determine method for deriving molecular fingerprints. Supported choices
        are "daylight", "morgan1", "morgan2", "morgan3". Default is "daylight".
    nbits:
        Dimension or number of bits of generated fingerprint. Default is 2048.
    """

    spectrum = AddFingerprint(fingerprint_type, nbits).process(spectrum_in)
    return spectrum