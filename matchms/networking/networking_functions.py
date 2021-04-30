""" Helper functions to build and handle spectral networks
"""
from typing import Tuple
from matchms import Scores


def get_top_hits(scores: Scores, identifier_key: str = "spectrumid",
                 top_n: int = 25, search_by: str = "queries") -> Tuple[dict, dict]:
    """Get top_n highest scores (and indices) for every entry.

    Parameters
    ----------
    scores
        Matchms Scores object containing all similarities.
    identifier_key
        Metadata key for unique intentifier for each spectrum in scores.
        Will also be used for the naming the network nodes. Default is 'spectrumid'.
    top_n
        Return the indexes and scores for the top_n highest scores.
    search_by
        Chose between 'queries' or 'references' which decides if the top_n matches
        for every spectrum in scores.queries or in scores.references will be
        collected and returned
    """
    assert search_by in ["queries", "references"], \
        "search_by must be 'queries' or 'references"
    if top_n < 2:
        top_n = 2
        print("Set top_n to minimum value of 2")

    similars_idx = dict()
    similars_scores = dict()

    if search_by == "queries":
        for i, spec in enumerate(scores.queries):
            spec_id = spec.get(identifier_key)
            similars_idx[spec_id] = scores.scores[:, i].argsort()[::-1][:top_n]
            similars_scores[spec_id] = scores.scores[similars_idx[spec_id], i]
    elif search_by == "references":
        for i, spec in enumerate(scores.references):
            spec_id = spec.get(identifier_key)
            similars_idx[spec_id] = scores.scores[i, :].argsort()[::-1][:top_n]
            similars_scores[spec_id] = scores.scores[i, similars_idx[spec_id]]
    return similars_idx, similars_scores
