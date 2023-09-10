import logging
import numpy as np
from matchms.typing import SpectrumType
from matchms.filtering.filters.base_spectrum_filter import BaseSpectrumFilter
from ...metadata_utils import (derive_fingerprint_from_inchi,
                               derive_fingerprint_from_smiles)


logger = logging.getLogger("matchms")


class AddFingerprint(BaseSpectrumFilter):
    def __init__(self, fingerprint_type: str = "daylight", nbits: int = 2048):
        self.fingerprint_type = fingerprint_type
        self.nbits = nbits

    def apply_filter(self, spectrum: SpectrumType) -> SpectrumType:
        if spectrum.get("smiles", None):
            fingerprint = derive_fingerprint_from_smiles(spectrum.get("smiles"),
                                                         self.fingerprint_type, self.nbits)
            if isinstance(fingerprint, np.ndarray) and fingerprint.sum() > 0:
                spectrum.set("fingerprint", fingerprint)
                return spectrum

        # Second try to get fingerprint from inchi
        if spectrum.get("inchi", None):
            fingerprint = derive_fingerprint_from_inchi(spectrum.get("inchi"),
                                                        self.fingerprint_type, self.nbits)
            if isinstance(fingerprint, np.ndarray) and fingerprint.sum() > 0:
                spectrum.set("fingerprint", fingerprint)
                return spectrum

        logger.info("No fingerprint was added (name: %s).", spectrum.get("compound_name"))
        return spectrum
