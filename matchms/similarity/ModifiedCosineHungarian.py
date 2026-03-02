import logging
from typing import Dict, Tuple
import numpy as np
from scipy.optimize import linear_sum_assignment
from matchms.similarity.spectrum_similarity_functions import collect_peak_pairs
from matchms.typing import SpectrumType
from ._precursor_validation import get_valid_precursor_mz
from .BaseSimilarity import BaseSimilarity


logger = logging.getLogger("matchms")


class ModifiedCosineHungarian(BaseSimilarity):
    """Calculate exact modified cosine score between mass spectra.

    The modified cosine score quantifies similarity between two mass spectra with
    optional precursor-based mass shift. Potential matches are all peak pairs that
    are within ``tolerance`` either unshifted or shifted by
    ``precursor_mz(reference) - precursor_mz(query)``.

    Peak assignment is solved globally via Hungarian assignment
    (linear sum assignment), which yields an exact one-to-one maximum-weight
    matching.

    See Watrous et al. [PNAS, 2012, https://www.pnas.org/content/109/26/E1743]
    for the modified cosine concept.
    """

    is_commutative = True
    score_datatype = [("score", np.float64), ("matches", "int")]

    def __init__(self, tolerance: float = 0.1, mz_power: float = 0.0, intensity_power: float = 1.0):
        """Initialize exact modified cosine.

        Parameters
        ----------
        tolerance:
            Peaks will be considered a match when <= tolerance apart. Default is 0.1.
        mz_power:
            The power to raise mz to in the cosine function. The default is 0, in which
            case the peak intensity products will not depend on the m/z ratios.
        intensity_power:
            The power to raise intensity to in the cosine function. The default is 1.
        """
        self.tolerance = tolerance
        self.mz_power = mz_power
        self.intensity_power = intensity_power

    def pair(self, reference: SpectrumType, query: SpectrumType) -> Tuple[float, int]:
        """Calculate exact modified cosine score between two spectra."""

        def get_matching_pairs():
            """Find all candidate matching peak pairs for unshifted and shifted views."""
            zero_pairs = collect_peak_pairs(
                spec1, spec2, self.tolerance, shift=0.0,
                mz_power=self.mz_power, intensity_power=self.intensity_power
            )

            precursor_mz_ref = get_valid_precursor_mz(reference, logger)
            precursor_mz_query = get_valid_precursor_mz(query, logger)
            mass_shift = precursor_mz_ref - precursor_mz_query
            nonzero_pairs = collect_peak_pairs(
                spec1, spec2, self.tolerance, shift=mass_shift,
                mz_power=self.mz_power, intensity_power=self.intensity_power
            )

            if zero_pairs is None:
                zero_pairs = np.zeros((0, 3))
            if nonzero_pairs is None:
                nonzero_pairs = np.zeros((0, 3))
            if zero_pairs.shape[0] == 0 and nonzero_pairs.shape[0] == 0:
                return np.zeros((0, 3))
            return np.concatenate((zero_pairs, nonzero_pairs), axis=0)

        def build_weight_matrix(matching_pairs: np.ndarray):
            """Build dense weight matrix from matching pairs.

            Duplicate (i, j) edges can occur when both shift=0 and shift!=0 generate the
            same pair; keep the maximum weight for the edge.
            """
            if matching_pairs.shape[0] == 0:
                return None, None, None

            deduplicated_edges: Dict[Tuple[int, int], float] = {}
            for peak_i, peak_j, weight in matching_pairs:
                edge = (int(peak_i), int(peak_j))
                current_weight = deduplicated_edges.get(edge)
                if current_weight is None or weight > current_weight:
                    deduplicated_edges[edge] = float(weight)

            if len(deduplicated_edges) == 0:
                return None, None, None

            paired_peaks1 = sorted({edge[0] for edge in deduplicated_edges})
            paired_peaks2 = sorted({edge[1] for edge in deduplicated_edges})

            idx_map1 = {peak_idx: i for i, peak_idx in enumerate(paired_peaks1)}
            idx_map2 = {peak_idx: j for j, peak_idx in enumerate(paired_peaks2)}

            weights = np.zeros((len(paired_peaks1), len(paired_peaks2)), dtype=np.float64)
            for (peak_i, peak_j), weight in deduplicated_edges.items():
                weights[idx_map1[peak_i], idx_map2[peak_j]] = weight

            return paired_peaks1, paired_peaks2, weights

        def solve_hungarian(weights: np.ndarray):
            """Solve maximum weight matching with Hungarian assignment."""
            n_rows, n_cols = weights.shape
            size = max(n_rows, n_cols)

            padded_weights = np.zeros((size, size), dtype=np.float64)
            padded_weights[:n_rows, :n_cols] = weights
            max_weight = padded_weights.max()
            costs = max_weight - padded_weights

            row_ind, col_ind = linear_sum_assignment(costs)

            score = 0.0
            used_matches = []
            for i, j in zip(row_ind, col_ind):
                if i < n_rows and j < n_cols and weights[i, j] > 0:
                    score += weights[i, j]
                    used_matches.append((i, j))
            return score, used_matches

        spec1 = reference.peaks.to_numpy
        spec2 = query.peaks.to_numpy
        matching_pairs = get_matching_pairs()

        paired_peaks1, paired_peaks2, weights = build_weight_matrix(matching_pairs)
        if weights is None:
            return np.asarray((0.0, 0), dtype=self.score_datatype)

        score, used_matches = solve_hungarian(weights)

        spec1_power = np.power(spec1[:, 0], self.mz_power) * np.power(spec1[:, 1], self.intensity_power)
        spec2_power = np.power(spec2[:, 0], self.mz_power) * np.power(spec2[:, 1], self.intensity_power)
        denominator = np.sqrt(np.sum(spec1_power ** 2)) * np.sqrt(np.sum(spec2_power ** 2))
        if denominator > 0:
            score = score / denominator
        else:
            score = 0.0

        used_matches = [(paired_peaks1[i], paired_peaks2[j]) for i, j in used_matches]
        return np.asarray((score, len(used_matches)), dtype=self.score_datatype)
