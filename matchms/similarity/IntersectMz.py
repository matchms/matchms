from matchms.typing import SpectrumType


class IntersectMz:
    """Example score for illustrating how to build custom spectra similarity score.

    Example of how matchms similarity functions can be used:

    .. code-block:: python

        from matchms import Scores, Spectrum
        from matchms.similarity import IntersectMz

        spectrum_1 = Spectrum(mz=np.array([100, 150, 200.]),
                              intensities=np.array([0.7, 0.2, 0.1]),
                              metadata={'id': 'spectrum1'})
        spectrum_2 = Spectrum(mz=np.array([100, 140, 190.]),
                              intensities=np.array([0.4, 0.2, 0.1]),
                              metadata={'id': 'spectrum2'})
        spectrum_3 = Spectrum(mz=np.array([110, 140, 195.]),
                              intensities=np.array([0.6, 0.2, 0.1]),
                              metadata={'id': 'spectrum3'})
        spectrum_4 = Spectrum(mz=np.array([100, 150, 200.]),
                              intensities=np.array([0.6, 0.1, 0.6]),
                              metadata={'id': 'spectrum4'})
        references = [spectrum_1, spectrum_2]
        queries = [spectrum_3, spectrum_4]

        scores = Scores(references, queries, IntersectMz()).calculate()

    """

    def __init__(self, scaling: float = 1.0):
        """Constructor. Here, function parameters are defined."""
        self.scaling = scaling

    def __call__(self, spectrum: SpectrumType, reference_spectrum: SpectrumType) -> float:
        """Call method. This will calculate the similarity score between two spectra."""
        mz = set(spectrum.peaks.mz)
        mz_ref = set(reference_spectrum.peaks.mz)
        intersected = mz.intersection(mz_ref)
        unioned = mz.union(mz_ref)

        if len(unioned) == 0:
            return 0

        return self.scaling * len(intersected) / len(unioned)
