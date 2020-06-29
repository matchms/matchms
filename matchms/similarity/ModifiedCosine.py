from typing import Tuple
from matchms.similarity.spectrum_similarity_functions import collect_peak_pairs
from matchms.similarity.spectrum_similarity_functions import get_peaks_array
from matchms.similarity.spectrum_similarity_functions import score_best_matches
from matchms.typing import SpectrumType


class ModifiedCosine:
    """Calculate 'modified cosine score' between mass spectra.

    The modified cosine score aims at quantifying the similarity between two
    mass spectra. The score is calculated by finding best possible matches between
    peaks of two spectra. Two peaks are considered a potential match if their
    m/z ratios lie within the given 'tolerance', or if their m/z ratios
    lie within the tolerance once a mass-shift is applied. The mass shift is
    simply the difference in precursor-m/z between the two spectra.
    See Watrous et al. [PNAS, 2012, https://www.pnas.org/content/109/26/E1743]
    for further details.

    For example

    .. testcode::

        import numpy as np
        from matchms import Spectrum
        from matchms.similarity import ModifiedCosine

        spectrum_1 = Spectrum(mz=np.array([100, 150, 200.]),
                              intensities=np.array([0.7, 0.2, 0.1]),
                              metadata={"precursor_mz": 100.0})
        spectrum_2 = Spectrum(mz=np.array([104.9, 140, 190.]),
                              intensities=np.array([0.4, 0.2, 0.1]),
                              metadata={"precursor_mz": 105.0})

        # Use factory to construct a similarity function
        modified_cosine = ModifiedCosine(tolerance=0.2)

        score, n_matches = modified_cosine(spectrum_1, spectrum_2)

        print(f"Modified cosine score is {score:.2f} with {n_matches} matched peaks")

    Should output

    .. testoutput::

        Modified cosine score is 0.52 with 1 matched peaks

    """
    def __init__(self, tolerance: float = 0.1):
        """
        Args:
        -----
        tolerance
            Peaks will be considered a match when <= tolerance a part. Default is 0.1.
        """
        self.tolerance = tolerance

    def __call__(self, spectrum1: SpectrumType, spectrum2: SpectrumType) -> Tuple[float, int]:
        """Calculate modified cosine score between two spectra.
        Args:
        -----
        spectrum1: SpectrumType
            Input spectrum 1.
        spectrum2: SpectrumType
            Input spectrum 2.

        Returns:
        --------

        Tuple with cosine score and number of matched peaks.
        """
        def get_matching_pairs():
            """Find all pairs of peaks that match within the given tolerance."""
            zero_pairs = collect_peak_pairs(spec1, spec2, self.tolerance, shift=0.0)
            message = "Precursor_mz missing. Apply 'add_precursor_mz' filter first."
            assert spectrum1.get("precursor_mz") and spectrum2.get("precursor_mz"), message
            mass_shift = spectrum1.get("precursor_mz") - spectrum2.get("precursor_mz")
            nonzero_pairs = collect_peak_pairs(spec1, spec2, self.tolerance, shift=mass_shift)
            unsorted_matching_pairs = zero_pairs + nonzero_pairs
            return sorted(unsorted_matching_pairs, key=lambda x: x[2], reverse=True)

        spec1 = get_peaks_array(spectrum1)
        spec2 = get_peaks_array(spectrum2)
        matching_pairs = get_matching_pairs()
        return score_best_matches(matching_pairs, spec1, spec2)
