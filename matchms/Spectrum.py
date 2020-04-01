# file: matchms/Spectrum.py

class Spectrum:

    def __init__(self, pyteomics_spectrum):
        self.pyteomics_spectrum = pyteomics_spectrum
        self.mz = pyteomics_spectrum["m/z array"]
        self.intensities = pyteomics_spectrum["intensity array"]
        self.something_from_the_metadata = pyteomics_spectrum.get("params", None).get("something_from_the_metadata", None)
