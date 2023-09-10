import numpy as np
from matchms.Fragments import Fragments
from matchms.typing import SpectrumType
from matchms.filtering.filters.base_spectrum_filter import BaseSpectrumFilter


class RemovePeaksAroundPrecursorMz(BaseSpectrumFilter):
    def __init__(self, mz_tolerance: float = 17):
        self.mz_tolerance = mz_tolerance

    def apply_filter(self, spectrum: SpectrumType) -> SpectrumType:
        precursor_mz = spectrum.get("precursor_mz", None)
        assert precursor_mz is not None, "Precursor mz absent."
        assert isinstance(precursor_mz, (float, int)), ("Expected 'precursor_mz' to be a scalar number.",
                                                        "Consider applying 'add_precursor_mz' filter first.")
        assert self.mz_tolerance >= 0, "mz_tolerance must be a positive scalar."

        mzs, intensities = spectrum.peaks.mz, spectrum.peaks.intensities
        peaks_to_remove = ((np.abs(precursor_mz-mzs) <= self.mz_tolerance) & (mzs != precursor_mz))
        new_mzs, new_intensities = mzs[~peaks_to_remove], intensities[~peaks_to_remove]
        spectrum.peaks = Fragments(mz=new_mzs, intensities=new_intensities)

        return spectrum
