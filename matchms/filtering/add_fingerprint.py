import numpy
from ..typing import SpectrumType
from ..utils import derive_fingerprint_from_inchi
from ..utils import derive_fingerprint_from_smiles


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
    if spectrum_in is None:
        return None

    spectrum = spectrum_in.clone()

    # First try to get fingerprint from smiles
    if spectrum.get("smiles", None):
        fingerprint = derive_fingerprint_from_smiles(spectrum.get("smiles"),
                                                     fingerprint_type, nbits)
        if isinstance(fingerprint, numpy.ndarray) and fingerprint.sum() > 0:
            spectrum.set("fingerprint", fingerprint)
            return spectrum

    # Second try to get fingerprint from inchi
    if spectrum.get("inchi", None):
        fingerprint = derive_fingerprint_from_inchi(spectrum.get("inchi"),
                                                    fingerprint_type, nbits)
        if isinstance(fingerprint, numpy.ndarray) and fingerprint.sum() > 0:
            spectrum.set("fingerprint", fingerprint)
            return spectrum

    return spectrum
