from typing import Tuple
import numpy
from matchms.typing import SpectrumType


class CosineGreedyVectorial:
    """Factory to calculate 'greedy cosine score' between mass spectra.

    The cosine score aims at quantifying the similarity between two mass spectra.
    The score is calculated by finding best possible matches between peaks
    of two spectra. Two peaks are considered a potential match if their
    m/z ratios lie within the given 'tolerance'.
    The underlying peak assignment problem is here solved in a 'greedy' way.
    This can perform notably faster, but does occasionally deviate slightly from
    a fully correct solution (as with the Hungarian algorithm). In practice this
    will rarely affect similarity scores notably, in particular for smaller
    tolerances.

    For example

    .. testcode::

        import numpy as np
        from matchms import Spectrum
        from matchms.similarity import CosineGreedyVectorial

        spectrum_1 = Spectrum(mz=np.array([100, 150, 200.]),
                              intensities=np.array([0.7, 0.2, 0.1]))
        spectrum_2 = Spectrum(mz=np.array([100, 140, 190.]),
                              intensities=np.array([0.4, 0.2, 0.1]))

        # Use factory to construct a similarity function
        cosine_greedy = CosineGreedyVectorial(tolerance=0.2)

        score, n_matches = cosine_greedy(spectrum_1, spectrum_2)

        print(f"Cosine score is {score:.2f} with {n_matches} matched peaks")

    Should output

    .. testoutput::

        Cosine score is 0.52 with 1 matched peaks

    """
    def __init__(self, tolerance: float = 0.3):
        """

        Parameters
        ----------
        tolerance
            Peaks will be considered a match when <= tolerance apart.
        """
        self.tolerance = tolerance

    def __call__(self, spectrum: SpectrumType, reference_spectrum: SpectrumType) -> Tuple[float, int]:
        """Calculate 'greedy cosine score' between mass spectra.

        Args:
        -----
        spectrum
            First spectrum
        reference_spectrum
            Second spectrum

        Returns:
        --------

        Tuple with cosine score and number of matched peaks.
        """
        def calc_mz_distance():
            mz_row_vector = spectrum.peaks.mz
            mz_col_vector = numpy.reshape(reference_spectrum.peaks.mz, (n_rows, 1))

            mz1 = numpy.tile(mz_row_vector, (n_rows, 1))
            mz2 = numpy.tile(mz_col_vector, (1, n_cols))

            return mz1 - mz2

        def calc_intensities_product():
            intensities_row_vector = spectrum.peaks.intensities
            intensities_col_vector = numpy.reshape(reference_spectrum.peaks.intensities, (n_rows, 1))

            intensities1 = numpy.tile(intensities_row_vector, (n_rows, 1))
            intensities2 = numpy.tile(intensities_col_vector, (1, n_cols))

            return intensities1 * intensities2

        def calc_intensities_product_within_tolerance():

            mz_distance = calc_mz_distance()
            intensities_product = calc_intensities_product()

            within_tolerance = numpy.absolute(mz_distance) <= self.tolerance

            return numpy.where(within_tolerance, intensities_product, numpy.zeros_like(intensities_product))

        def calc_score():
            r_unordered, c_unordered = intensities_product_within_tolerance.nonzero()
            v_unordered = intensities_product_within_tolerance[r_unordered, c_unordered]
            sortorder = numpy.argsort(v_unordered)[::-1]
            r_sorted = r_unordered[sortorder]
            c_sorted = c_unordered[sortorder]

            score = 0
            n_matches = 0
            for r, c in zip(r_sorted, c_sorted):
                if intensities_product_within_tolerance[r, c] > 0:
                    score += intensities_product_within_tolerance[r, c]
                    n_matches += 1
                    intensities_product_within_tolerance[r, :] = 0
                    intensities_product_within_tolerance[:, c] = 0
            return score / max(sum(squared1), sum(squared2)), n_matches

        n_rows = reference_spectrum.peaks.mz.size
        n_cols = spectrum.peaks.mz.size

        intensities_product_within_tolerance = calc_intensities_product_within_tolerance()

        squared1 = numpy.power(spectrum.peaks.intensities, 2)
        squared2 = numpy.power(reference_spectrum.peaks.intensities, 2)

        return calc_score()
