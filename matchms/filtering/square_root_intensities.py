import numpy
from matchms.typing import SpectrumType
from ._intenstiy_manager import IntensityManagerBase


class IntensityManagerSquareRoot(IntensityManagerBase):
    """Class to manage remplacement of a spectrum instensity to its square root intensity."""

    def _get_new_intensities(self, intensities: numpy.ndarray) -> numpy.ndarray:
        return numpy.sqrt(intensities)


def square_root_intensities(spectrum_in: SpectrumType) -> SpectrumType:
    """Replace intensities of an MS/MS spectrum with their square-root
    to minimize/maximize effects of high/low intensity peaks"""

    manager = IntensityManagerSquareRoot(spectrum_in)

    return manager.replace_intensities()
