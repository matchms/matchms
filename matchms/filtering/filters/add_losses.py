import logging
import numpy as np
from matchms.Fragments import Fragments
from matchms.typing import SpectrumType
from matchms.filtering.filters.base_spectrum_filter import BaseSpectrumFilter


logger = logging.getLogger("matchms")


class AddLosses(BaseSpectrumFilter):
    def __init__(self, loss_mz_from=0.0, loss_mz_to=1000.0):
        self.loss_mz_from = loss_mz_from
        self.loss_mz_to = loss_mz_to

    def apply_filter(self, spectrum: SpectrumType) -> SpectrumType:
        precursor_mz = spectrum.get("precursor_mz", None)
        if precursor_mz:
            assert isinstance(precursor_mz, (float, int)), ("Expected 'precursor_mz' to be a scalar number.",
                                                            "Consider applying 'add_precursor_mz' filter first.")
            peaks_mz, peaks_intensities = spectrum.peaks.mz, spectrum.peaks.intensities
            losses_mz = (precursor_mz - peaks_mz)[::-1]
            losses_intensities = peaks_intensities[::-1]
            # Add losses which are within given boundaries
            mask = np.where((losses_mz >= self.loss_mz_from)
                               & (losses_mz <= self.loss_mz_to))
            spectrum.losses = Fragments(mz=losses_mz[mask],
                                        intensities=losses_intensities[mask])
        else:
            logger.warning("No precursor_mz found. Consider applying 'add_precursor_mz' filter first.")

        return spectrum
