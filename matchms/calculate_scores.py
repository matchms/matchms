from typing import Sequence
from .Scores import Scores
from .similarity.BaseSimilarity import BaseSimilarity
from .typing import SpectrumType


def calculate_scores(spectra_1: Sequence[SpectrumType], spectra_2: Sequence[SpectrumType],
                     similarity_function: BaseSimilarity) -> Scores:
    """Calculate the similarity between all reference objects versus all query objects.

    Example to calculate scores between 2 spectra and iterate over the scores

    .. testcode::

        import numpy as np
        from matchms import calculate_scores, Spectrum
        from matchms.similarity import CosineGreedy

        spectrum_1 = Spectrum(mz=np.array([100, 150, 200.]),
                              intensities=np.array([0.7, 0.2, 0.1]),
                              metadata={'id': 'spectrum1'})
        spectrum_2 = Spectrum(mz=np.array([100, 140, 190.]),
                              intensities=np.array([0.4, 0.2, 0.1]),
                              metadata={'id': 'spectrum2'})
        spectra = [spectrum_1, spectrum_2]

        scores = calculate_scores(spectra, spectra, CosineGreedy())

        for (reference, query, score) in scores:
            print(f"Cosine score between {spectrum_1.get('id')} and {spectrum_2.get('id')}" +
                  f" is {score[0]:.2f} with {score[1]} matched peaks")

    Should output

    .. testoutput::

        Cosine score between spectrum1 and spectrum1 is 1.00 with 3 matched peaks
        Cosine score between spectrum1 and spectrum2 is 0.83 with 1 matched peaks
        Cosine score between spectrum2 and spectrum1 is 0.83 with 1 matched peaks
        Cosine score between spectrum2 and spectrum2 is 1.00 with 3 matched peaks

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

    ~matchms.Scores.Scores
    """
    return similarity_function.matrix(spectra_1, spectra_2)
