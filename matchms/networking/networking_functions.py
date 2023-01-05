""" Helper functions to build and handle spectral networks
"""
from typing import Tuple
import numpy as np
from matchms import Scores


def get_top_hits(scores: Scores, identifier_key: str = "spectrum_id",
                 top_n: int = 25, search_by: str = "queries",
                 score_name: str = None,
                 ignore_diagonal: bool = False) -> Tuple[dict, dict]:
    """Get top_n highest scores (and indices) for every entry.

    Parameters
    ----------
    scores
        Matchms Scores object containing all similarities.
    identifier_key
        Metadata key for unique intentifier for each spectrum in scores.
        Will also be used for the naming the network nodes. Default is 'spectrum_id'.
    top_n
        Return the indexes and scores for the top_n highest scores. Scores between
        a spectrum with itself (diagonal of scores.scores) will not be taken into
        account.
    search_by
        Chose between 'queries' or 'references' which decides if the top_n matches
        for every spectrum in scores.queries or in scores.references will be
        collected and returned.
    score_name
        Name of the score that should be used (if scores contains multiple different
        scores).
    ignore_diagonal
        Set to True if scores.scores is symmetric (i.e. if references and queries
        were the same) and if scores between spectra with themselves should be
        excluded.
    """
    # pylint: disable=protected-access, too-many-arguments, too-many-locals
    assert search_by in ["queries", "references"], \
        "search_by must be 'queries' or 'references"
    if score_name is None:
        score_name = scores._scores.guess_score_name()

    similars_idx = {}
    similars_scores = {}

    if search_by == "queries":
        for i, spec in enumerate(scores.queries):
            spec_id = spec.get(identifier_key)
            r, _, v = scores.scores[:, i, score_name]
            idx = np.argsort(v)[::-1]
            if ignore_diagonal:
                idx = idx[r[idx] != i]
            similars_idx[spec_id] = r[idx][:top_n]
            similars_scores[spec_id] = v[idx][:top_n]
    elif search_by == "references":
        for i, spec in enumerate(scores.references):
            spec_id = spec.get(identifier_key)
            _, c, v = scores.scores[i, :, score_name]
            idx = np.argsort(v)[::-1][:top_n]
            if ignore_diagonal:
                idx = idx[c[idx] != i]
            similars_idx[spec_id] = c[idx][:top_n]
            similars_scores[spec_id] = v[idx][:top_n]
    return similars_idx, similars_scores
