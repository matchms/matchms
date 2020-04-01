# file: matchms/Spectrum.py

from spectrum_utils.spectrum import MsmsSpectrum
from spectrum_utils.plot import spectrum as plot

class Spectrum:

    def __init__(self, pyteomics_spectrum):
        self.pyteomics_spectrum = pyteomics_spectrum
        self.mz = pyteomics_spectrum["m/z array"]
        self.intensities = pyteomics_spectrum["intensity array"]
        self.something_from_the_metadata = pyteomics_spectrum.get("params", None).get("something_from_the_metadata", None)

    def plot(self):

        s = MsmsSpectrum(identifier=None,
                         precursor_mz=None,
                         precursor_charge=None,
                         mz=self.mz, intensity=self.intensities)
        plot(s)
