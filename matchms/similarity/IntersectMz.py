import numpy as np
from matchms.typing import SpectrumType
from .BaseSimilarity import BaseSimilarity


class IntersectMz(BaseSimilarity):
    """Example score for illustrating how to build custom spectra similarity score.

    IntersectMz will count all exact matches of peaks and divide it by all unique
    peaks found in both spectrums.

    Example of how matchms similarity functions can be used:

    .. testcode::

        import numpy as np
        from matchms import Spectrum
        from matchms.similarity import IntersectMz

        spectrum_1 = Spectrum(mz=np.array([100, 150, 200.]),
                              intensities=np.array([0.7, 0.2, 0.1]))
        spectrum_2 = Spectrum(mz=np.array([100, 140, 190.]),
                              intensities=np.array([0.4, 0.2, 0.1]))

        # Construct a similarity function
        similarity_measure = IntersectMz(scaling=1.0)

        score = similarity_measure.pair(spectrum_1, spectrum_2)

        print(f"IntersectMz score is {score:.2f}")

    Should output

    .. testoutput::

        IntersectMz score is 0.20

    """

    def __init__(self, scaling: float = 1.0):
        """Constructor. Here, function parameters are defined.

        Parameters
        ----------
        scaling
            Scale scores to maximum possible score being 'scaling'.
        """
        self.scaling = scaling

    def pair(self, reference: SpectrumType, query: SpectrumType) -> float:
        """This will calculate the similarity score between two spectra."""
        mz_ref = set(reference.peaks.mz)
        mz_query = set(query.peaks.mz)
        intersected = mz_query.intersection(mz_ref)
        unioned = mz_query.union(mz_ref)

        if len(unioned) == 0:
            return 0

        return np.float64(self.scaling * len(intersected) / len(unioned))
