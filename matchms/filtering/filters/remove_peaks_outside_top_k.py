import numpy as np
from matchms.Fragments import Fragments
from matchms.typing import SpectrumType
from matchms.filtering.filters.base_spectrum_filter import BaseSpectrumFilter


class RemovePeaksOutsideTopK(BaseSpectrumFilter):
    def __init__(self, k: int = 6, mz_window: float = 50):
        self.k = k
        self.mz_window = mz_window

    def apply_filter(self, spectrum: SpectrumType) -> SpectrumType:
        assert self.k >= 1, "k must be a positive nonzero integer."
        assert self.mz_window >= 0, "mz_window must be a positive scalar."
        mzs, intensities = spectrum.peaks.mz, spectrum.peaks.intensities
        top_k = intensities.argsort()[::-1][0:self.k]
        k_ordered_mzs = mzs[top_k]
        indices = [i for i in range(len(mzs)) if i not in top_k]
        keep_idx = top_k.tolist()
        for i in indices:
            compare = abs(mzs[i]-k_ordered_mzs) <= self.mz_window
            if np.any(compare):
                keep_idx.append(i)

        keep_idx.sort()
        new_mzs, new_intensities = mzs[keep_idx], intensities[keep_idx]
        spectrum.peaks = Fragments(mz=new_mzs, intensities=new_intensities)

        return spectrum
