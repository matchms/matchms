import numpy
from matchms.typing import SpectrumType


class CosineGreedy:

    def __init__(self, tolerance=0.3):
        self.tolerance = tolerance

    def __call__(self, spectrum: SpectrumType, reference_spectrum: SpectrumType) -> float:
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
