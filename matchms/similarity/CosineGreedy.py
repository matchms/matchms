from typing import Tuple
import numpy
from matchms.typing import SpectrumType
from .BaseSimilarity import BaseSimilarity
from .spectrum_similarity_functions import collect_peak_pairs
from .spectrum_similarity_functions import get_peaks_array
from .spectrum_similarity_functions import score_best_matches


class CosineGreedy(BaseSimilarity):
    """Calculate 'cosine similarity score' between two spectra.

    The cosine score aims at quantifying the similarity between two mass spectra.
    The score is calculated by finding best possible matches between peaks
    of two spectra. Two peaks are considered a potential match if their
    m/z ratios lie within the given 'tolerance'.
    The underlying peak assignment problem is here solved in a 'greedy' way.
    This can perform notably faster, but does occasionally deviate slightly from
    a fully correct solution (as with the Hungarian algorithm, see
    :class:`~matchms.similarity.CosineHungarian`). In practice this will rarely
    affect similarity scores notably, in particular for smaller tolerances.

    For example

    .. testcode::

        import numpy as np
        from matchms import Spectrum
        from matchms.similarity import CosineGreedy

        reference = Spectrum(mz=np.array([100, 150, 200.]),
                             intensities=np.array([0.7, 0.2, 0.1]))
        query = Spectrum(mz=np.array([100, 140, 190.]),
                         intensities=np.array([0.4, 0.2, 0.1]))

        # Use factory to construct a similarity function
        cosine_greedy = CosineGreedy(tolerance=0.2)

        score = cosine_greedy.pair(reference, query)

        print(f"Cosine score is {score['score']:.2f} with {score['matches']} matched peaks")

    Should output

    .. testoutput::

        Cosine score is 0.83 with 1 matched peaks

    """
    # Set key characteristics as class attributes
    is_commutative = True
    # Set output data type, e.g. ("score", "float") or [("score", "float"), ("matches", "int")]
    score_datatype = [("score", numpy.float64), ("matches", "int")]

    def __init__(self, tolerance: float = 0.1, mz_power: float = 0.0,
                 intensity_power: float = 1.0):
        """
        Parameters
        ----------
        tolerance:
            Peaks will be considered a match when <= tolerance apart. Default is 0.1.
        mz_power:
            The power to raise m/z to in the cosine function. The default is 0, in which
            case the peak intensity products will not depend on the m/z ratios.
        intensity_power:
            The power to raise intensity to in the cosine function. The default is 1.
        """
        self.tolerance = tolerance
        self.mz_power = mz_power
        self.intensity_power = intensity_power

    def pair(self, reference: SpectrumType, query: SpectrumType) -> Tuple[float, int]:
        """Calculate cosine score between two spectra.

        Parameters
        ----------
        reference
            Single reference spectrum.
        query
            Single query spectrum.

        Returns
        -------
        Score
            Tuple with cosine score and number of matched peaks.
        """
        def get_matching_pairs():
            """Get pairs of peaks that match within the given tolerance."""
            matching_pairs = collect_peak_pairs(spec1, spec2, self.tolerance,
                                                shift=0.0, mz_power=self.mz_power,
                                                intensity_power=self.intensity_power)
            if matching_pairs is None:
                return None
            matching_pairs = matching_pairs[numpy.argsort(matching_pairs[:, 2])[::-1], :]
            return matching_pairs

        spec1 = get_peaks_array(reference)
        spec2 = get_peaks_array(query)
        matching_pairs = get_matching_pairs()
        if matching_pairs is None:
            return numpy.asarray((float(0), 0), dtype=self.score_datatype)
        score = score_best_matches(matching_pairs, spec1, spec2,
                                   self.mz_power, self.intensity_power)
        return numpy.asarray(score, dtype=self.score_datatype)
