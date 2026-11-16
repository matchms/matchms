import numpy as np
from matchms import Spectrum


class SpectrumBuilder:
    """Builder class to better handle Spectrum creation throughout all matchms unit tests."""

    def __init__(self):
        self._mz = np.array([], dtype="float")
        self._intensities = np.array([], dtype="float")
        self._metadata = {}
        self._metadata_harmonization = False

    def from_spectrum(self, spectrum: Spectrum):
        return (
            self.with_mz(spectrum.peaks.mz)
            .with_intensities(spectrum.peaks.intensities)
            .with_metadata(spectrum.metadata)
        )

    def with_mz(self, mz: np.ndarray):
        if isinstance(mz, np.ndarray):
            self._mz = np.copy(mz)
        else:
            self._mz = np.array(mz, dtype="float")
        return self

    def with_intensities(self, intensities: np.ndarray):
        if isinstance(intensities, np.ndarray):
            self._intensities = np.copy(intensities)
        else:
            self._intensities = np.array(intensities, dtype="float")
        return self

    def with_metadata(self, metadata: dict, metadata_harmonization: bool = False):
        self._metadata = metadata.copy()
        self._metadata_harmonization = metadata_harmonization
        return self

    def build(self) -> Spectrum:
        spectrum = Spectrum(
            mz=self._mz,
            intensities=self._intensities,
            metadata=self._metadata,
            metadata_harmonization=self._metadata_harmonization,
        )
        return spectrum


def spectra_factory(key, values):
    builder = SpectrumBuilder()
    spectra = [builder.with_metadata({key: x}).build() for x in values]
    return spectra
