import numpy
from matchms.typing import SpectrumType
from ..Spikes import Spikes


class IntensityManagerBase:
    """Base class to replace intensities of a spectrum.
    This base class is designed to be extended by modyfing the ._get_new_intensities() method."""

    spectrum = None

    def __init__(self, spectrum_in: SpectrumType):

        if spectrum_in is not None:
            self.spectrum = spectrum_in.clone()

    def replace_intensities(self) -> SpectrumType:
        """Replace intensities of an MS/MS spectrum with their square-root."""

        spectrum = self.spectrum

        if spectrum is None:
            return None

        if len(spectrum.peaks) == 0:
            return spectrum

        self._replace_spectrum_attribute("peaks")
        if spectrum.losses is not None and len(spectrum.losses) > 0:
            self._replace_spectrum_attribute("losses")

        return spectrum

    def _replace_spectrum_attribute(self, attr_name: str) -> None:
        mz, intensities = getattr(self.spectrum, attr_name)
        new_intensities = self._get_new_intensities(intensities)
        setattr(self.spectrum, attr_name, Spikes(mz=mz, intensities=new_intensities))

    def _get_new_intensities(self, intensities: numpy.ndarray) -> numpy.ndarray:
        """This method perform the actual computation on intensities.
        Change this method when extend this class."""

        return intensities
