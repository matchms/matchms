import logging
from math import ceil
from typing import Optional
import numpy as np
from matchms.Fragments import Fragments
from matchms.typing import SpectrumType
from matchms.filtering.filters.base_spectrum_filter import BaseSpectrumFilter


logger = logging.getLogger("matchms")


class ReduceToNumberOfPeaks(BaseSpectrumFilter):
    def __init__(self, n_required: int = 1, n_max: int = np.inf, ratio_desired: Optional[float] = None):
        self.n_required = n_required
        self.n_max = n_max
        self.ratio_desired = ratio_desired

    def apply_filter(self, spectrum: SpectrumType) -> SpectrumType:
        def _set_maximum_number_of_peaks_to_keep():
            parent_mass = spectrum.get("parent_mass", None)
            if parent_mass and self.ratio_desired:
                n_desired_by_mass = int(ceil(self.ratio_desired * parent_mass))
                return min(max(self.n_required, n_desired_by_mass), self.n_max)
            if not self.ratio_desired:
                return self.n_max
            raise ValueError("Cannot use ratio_desired for spectrum without parent_mass.")

        def _remove_lowest_intensity_peaks():
            mz, intensities = spectrum.peaks.mz, spectrum.peaks.intensities
            idx = intensities.argsort()[-threshold:]
            idx_sort_by_mz = mz[idx].argsort()
            spectrum.peaks = Fragments(mz=mz[idx][idx_sort_by_mz],
                                       intensities=intensities[idx][idx_sort_by_mz])

        if spectrum.peaks.intensities.size < self.n_required:
            logger.info("Spectrum with %s (<%s) peaks was set to None.",
                        str(spectrum.peaks.intensities.size), str(self.n_required))
            return None

        threshold = _set_maximum_number_of_peaks_to_keep()
        if spectrum.peaks.intensities.size < threshold:
            return spectrum

        _remove_lowest_intensity_peaks()

        return spectrum
