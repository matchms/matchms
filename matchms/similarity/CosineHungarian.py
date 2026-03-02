from typing import List, Optional, Tuple
import numpy as np
from scipy.optimize import linear_sum_assignment
from matchms.similarity.spectrum_similarity_functions import collect_peak_pairs
from matchms.typing import SpectrumType
from .BaseSimilarity import BaseSimilarity
from .ScoreFilter import FilterScoreByValue


class CosineHungarian(BaseSimilarity):
    """Calculate 'cosine similarity score' between two spectra using the Hungarian algorithm.

    The cosine score quantifies the similarity between two mass spectra by finding
    the optimal one-to-one matching between their peaks. Two peaks are considered a
    potential match if their m/z ratios lie within the given *tolerance*.

    The peak assignment is solved using the Hungarian algorithm
    (``scipy.optimize.linear_sum_assignment``), which finds the assignment that
    maximises the sum of intensity products. This is mathematically optimal but can
    be notably slower than the greedy heuristic in
    :class:`~matchms.similarity.CosineGreedy`.
    """

    is_commutative = True
    score_datatype = [("score", np.float64), ("matches", "int")]

    def __init__(
        self,
        tolerance: float = 0.1,
        mz_power: float = 0.0,
        intensity_power: float = 1.0,
        score_filters: Optional[Tuple[FilterScoreByValue, ...]] = None,
    ):
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
        super().__init__(score_filters)
        self.tolerance = tolerance
        self.mz_power = mz_power
        self.intensity_power = intensity_power

    def pair(self, reference: SpectrumType, query: SpectrumType) -> np.ndarray:
        """Calculate cosine score between two spectra.

        Parameters
        ----------
        reference
            Single reference spectrum.
        query
            Single query spectrum.

        Returns
        -------
        Tuple with cosine score and number of matched peaks.
        """

        def get_matching_pairs() -> Optional[np.ndarray]:
            """Find all within-tolerance peak pairs, sorted by descending intensity product.

            Returns ``None`` when no peaks fall within *tolerance*.
            """
            matching_pairs = collect_peak_pairs(
                spec1, spec2, self.tolerance, shift=0.0, mz_power=self.mz_power, intensity_power=self.intensity_power
            )
            if matching_pairs is None:
                return None
            matching_pairs = matching_pairs[np.argsort(matching_pairs[:, 2], kind="mergesort")[::-1], :]
            return matching_pairs

        def get_matching_pairs_matrix() -> Tuple[Optional[List[float]], Optional[List[float]], Optional[np.ndarray]]:
            """Build a cost matrix for the Hungarian algorithm.

            Rows correspond to the unique reference peaks that participate in at
            least one within-tolerance pair; columns correspond to the unique query
            peaks. The matrix is initialised to ``1.0`` (maximum cost) and actual
            within-tolerance pairs are set to ``1 - intensity_product``.

            Returns
            -------
            paired_peaks1
                Unique reference-peak indices involved in at least one match.
            paired_peaks2
                Unique query-peak indices involved in at least one match.
            cost_matrix
                Cost matrix of shape ``(len(paired_peaks1), len(paired_peaks2))``.
                Cells that remain ``1.0`` have **no** real within-tolerance pair.
            """
            if matching_pairs is None:
                return None, None, None
            paired_peaks1 = list(set(matching_pairs[:, 0]))
            paired_peaks2 = list(set(matching_pairs[:, 1]))
            matrix_size = (len(paired_peaks1), len(paired_peaks2))
            cost_matrix = np.ones(matrix_size)
            for i in range(matching_pairs.shape[0]):
                cost_matrix[paired_peaks1.index(matching_pairs[i, 0]), paired_peaks2.index(matching_pairs[i, 1])] = (
                    1 - matching_pairs[i, 2]
                )
            return paired_peaks1, paired_peaks2, cost_matrix

        def solve_hungarian() -> Tuple[float, List[Tuple[int, int]]]:
            """Solve the optimal peak assignment via the Hungarian algorithm.

            The algorithm assigns ``min(rows, cols)`` pairs. Some of those
            assignments may land on cells that were never overwritten from the
            initial ``1.0`` — i.e. peak pairs that are **not** within tolerance
            ("phantom pairs"). These contribute ``1 - 1.0 = 0`` to the score, so
            the score is unaffected, but they must be excluded from the match count.

            The score formula ``len(row_ind) - sum(costs)`` relies on phantoms
            contributing exactly ``1.0`` to the sum, so the score line must use
            **all** assignments (including phantoms).

            Returns
            -------
            score
                Un-normalised cosine score (sum of intensity products for matched
                peaks).
            used_matches
                List of ``(reference_peak_idx, query_peak_idx)`` tuples for the
                real (non-phantom) matched pairs only.
            """
            row_ind, col_ind = linear_sum_assignment(matching_pairs_matrix)
            # Score uses ALL assignments: phantoms add 1.0 to the sum, cancelling
            # with the len(row_ind) term and contributing 0 to the score.
            score = len(row_ind) - matching_pairs_matrix[row_ind, col_ind].sum()
            # Match count excludes phantoms (cells still at 1.0).
            used_matches = [
                (int(paired_peaks1[x]), int(paired_peaks2[y]))
                for (x, y) in zip(row_ind, col_ind)
                if matching_pairs_matrix[x, y] < 1.0
            ]
            return score, used_matches

        def calc_score() -> np.ndarray:
            """Compute the normalised cosine score and match count."""
            if matching_pairs_matrix is None:
                return np.asarray((0.0, 0), dtype=self.score_datatype)
            score, used_matches = solve_hungarian()
            spec1_power = np.power(spec1[:, 0], self.mz_power) * np.power(spec1[:, 1], self.intensity_power)
            spec2_power = np.power(spec2[:, 0], self.mz_power) * np.power(spec2[:, 1], self.intensity_power)
            score = score / (np.sqrt(np.sum(spec1_power**2)) * np.sqrt(np.sum(spec2_power**2)))
            return np.asarray((score, len(used_matches)), dtype=self.score_datatype)

        spec1 = reference.peaks.to_numpy
        spec2 = query.peaks.to_numpy
        matching_pairs = get_matching_pairs()
        paired_peaks1, paired_peaks2, matching_pairs_matrix = get_matching_pairs_matrix()
        return calc_score()
