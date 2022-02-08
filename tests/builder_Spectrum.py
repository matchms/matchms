import numpy
from matchms import Spectrum


class SpectrumBuilder:
    """Builder class to better handle Spectrum creation throughout all matchms unit tests."""
    def __init__(self):
        self._mz = numpy.array([], dtype="float")
        self._intensities = numpy.array([], dtype="float")
        self._metadata = {}
        self._default_metadata_filtering = False

    def from_spectrum(self, spectrum: Spectrum):
        return self.with_mz(spectrum.peaks.mz).with_intensities(spectrum.peaks.intensities).with_metadata(spectrum.metadata)

    def with_mz(self, mz: numpy.ndarray):
        self._mz = numpy.copy(mz)
        return self

    def with_intensities(self, intensities: numpy.ndarray):
        self._intensities = numpy.copy(intensities)
        return self

    def with_metadata(self, metadata: dict,
                      default_metadata_filtering: bool = False):
        self._metadata = metadata.copy()
        self._default_metadata_filtering = default_metadata_filtering
        return self

    def build(self) -> Spectrum:
        spectrum = Spectrum(mz=self._mz,
                            intensities=self._intensities,
                            metadata=self._metadata,
                            default_metadata_filtering=self._default_metadata_filtering)
        return spectrum


def spectra_factory(key, values):
    builder = SpectrumBuilder()
    spectra = [builder.with_metadata({key: x}).build() for x in values]
    return spectra
