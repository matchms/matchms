import numpy
from matchms.typing import SpectrumType
from ._intenstiy_manager import IntensityManagerBase


class IntensityManagerNormalize(IntensityManagerBase):
    """Class to manage remplacement of a spectrum instensity to its normalized intensity."""

    max_intensity = None

    def __init__(self, *args, **kwargs):
        """Set max intensity."""

        super().__init__(*args, **kwargs)
        if (self.spectrum is not None) and (len(self.spectrum.peaks) > 0):
            self.max_intensity = numpy.max(self.spectrum.peaks.intensities)

    def _get_new_intensities(self, intensities: numpy.ndarray) -> numpy.ndarray:
        """Normalize intensities."""

        if self.max_intensity is None:
            return intensities
        return intensities / self.max_intensity


def normalize_intensities(spectrum_in: SpectrumType) -> SpectrumType:
    """Normalize intensities of peaks (and losses) to unit height."""

    manager = IntensityManagerNormalize(spectrum_in)

    return manager.replace_intensities()
