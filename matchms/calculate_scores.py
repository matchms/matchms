from __future__ import annotations
from collections.abc import Sequence
from typing import TYPE_CHECKING
from .scores import Scores
from .similarity.base_similarity import BaseSimilarity


if TYPE_CHECKING:
    from matchms.spectrum import Spectrum


def calculate_scores(spectra_1: Sequence[Spectrum], spectra_2: Sequence[Spectrum],
                     similarity_function: BaseSimilarity) -> Scores:
    """Calculate the similarity between all reference objects versus all query objects.

    Parameters
    ----------
    spectra_1
        List of reference objects
    spectra_2
        List of query objects
    similarity_function
        Function which accepts a reference + query object and returns a score or tuple of scores

    Returns
    -------

    ~matchms.scores.Scores
    """
    return similarity_function.matrix(spectra_1, spectra_2)
