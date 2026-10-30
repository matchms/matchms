import logging
import numpy as np
from matchms.typing import SpectrumType
from ._precursor_validation import get_valid_precursor_mz
from .cosine_linear import CosineLinear
from .cosine_linear_functions import modified_linear_cosine_score, sirius_merge_close_peaks
from .default_parameters import (
    DEFAULT_INTENSITY_POWER,
    DEFAULT_MZ_POWER,
    DEFAULT_MZ_TOLERANCE,
    DEFAULT_NOISE_CUTOFF,
    DEFAULT_OFFSET_TO_PRECURSOR,
)
from .spectrum_similarity_functions import _preprocess_peak_array


logger = logging.getLogger("matchms")


class ModifiedCosineLinear(CosineLinear):
    """Calculate the exact modified cosine score between mass spectra in linear time.

    Linear-time modified cosine from SIRIUS (BOECKER lab). Close peaks are merged as
    in :class:`~matchms.similarity.CosineLinear`, peaks are matched directly and
    shifted by the precursor m/z difference with one sweep each, and conflicts between
    the two are resolved optimally. The score equals
    :class:`~matchms.similarity.ModifiedCosineHungarian` on the merged spectra.

    See Watrous et al. [PNAS, 2012, https://www.pnas.org/content/109/26/E1743].

    For example

    .. testcode::

        import numpy as np
        from matchms import Spectrum
        from matchms.similarity import ModifiedCosineLinear

        reference = Spectrum(mz=np.array([100.0, 114.01565]),
                             intensities=np.array([0.6, 0.8]),
                             metadata={"precursor_mz": 300.0})
        query = Spectrum(mz=np.array([114.01565, 128.0313]),
                         intensities=np.array([0.8, 0.6]),
                         metadata={"precursor_mz": 314.01565})

        modified_cosine = ModifiedCosineLinear(tolerance=0.1)
        score = modified_cosine.pair(reference, query)

        print(f"Modified cosine score is {score['score']:.2f} with {score['matches']} matched peaks")

    Should output

    .. testoutput::

        Modified cosine score is 0.96 with 2 matched peaks

    """

    def __init__(
            self,
            tolerance: float = DEFAULT_MZ_TOLERANCE,
            mz_power: float = DEFAULT_MZ_POWER,
            intensity_power: float = DEFAULT_INTENSITY_POWER,
            noise_cutoff: float | None = DEFAULT_NOISE_CUTOFF,
            remove_precursor: bool = True,
            offset_to_precursor: float = DEFAULT_OFFSET_TO_PRECURSOR
            ):
        """
        Parameters
        ----------
        tolerance:
            Peaks will be considered a match when <= tolerance apart. Default is 0.01.
            Peaks closer than 2 * tolerance are merged before scoring.
        mz_power:
            The power to raise m/z to in the cosine function. The default is 0, in which
            case the peak intensity products will not depend on the m/z ratios.
        intensity_power:
            The power to raise intensity to in the cosine function. The default is 1.
        noise_cutoff:
            Minimum relative intensity for a peak to be considered. Default is 0.01.
        remove_precursor:
            Whether to remove peaks with m/z values larger than the precursor-m/z (plus offset).
        offset_to_precursor:
            The offset to add to the precursor-m/z when removing peaks.
        """
        super().__init__(
            tolerance=tolerance,
            mz_power=mz_power,
            intensity_power=intensity_power,
            noise_cutoff=noise_cutoff,
            remove_precursor=remove_precursor,
            offset_to_precursor=offset_to_precursor,
        )

    def _prepare_spectrum(self, spectrum: SpectrumType) -> tuple[np.ndarray, float]:  # type: ignore[override]
        """Preprocess and merge one spectrum, paired with its precursor m/z."""
        precursor_mz = get_valid_precursor_mz(spectrum, logger)
        peaks = _preprocess_peak_array(
            spectrum.peaks.to_numpy,
            precursor_mz=precursor_mz,
            remove_precursor=self.remove_precursor,
            offset_to_precursor=self.offset_to_precursor,
            noise_cutoff=self.noise_cutoff,
        )
        return sirius_merge_close_peaks(peaks, self.tolerance), float(precursor_mz)

    def _score(self, prepared_1, prepared_2) -> tuple[float, int]:
        peaks_1, precursor_mz_1 = prepared_1
        peaks_2, precursor_mz_2 = prepared_2
        return modified_linear_cosine_score(
            peaks_1,
            peaks_2,
            precursor_mz_1,
            precursor_mz_2,
            self.tolerance,
            self.mz_power,
            self.intensity_power,
        )

    def pair(self, spectrum_1: SpectrumType, spectrum_2: SpectrumType) -> tuple[float, int]:
        """Calculate the modified cosine score and number of matched peaks between two spectra."""
        return super().pair(spectrum_1, spectrum_2)
