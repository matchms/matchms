from .Scores import Scores
from .similarity.BaseSimilarity import BaseSimilarity
from .typing import QueriesType
from .typing import ReferencesType


def calculate_scores(references: ReferencesType, queries: QueriesType,
                     similarity_function: BaseSimilarity,
                     is_symmetric: bool = False) -> Scores:
    """Calculate the similarity between all reference objects versus all query objects.

    Example to calculate scores between 2 spectrums and iterate over the scores

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
        spectrums = [spectrum_1, spectrum_2]

        scores = calculate_scores(spectrums, spectrums, CosineGreedy())

        for (reference, query, score) in scores:
            print(f"Cosine score between {reference.get('id')} and {query.get('id')}" +
                  f" is {score['score']:.2f} with {score['matches']} matched peaks")

    Should output

    .. testoutput::

        Cosine score between spectrum1 and spectrum1 is 1.00 with 3 matched peaks
        Cosine score between spectrum1 and spectrum2 is 0.83 with 1 matched peaks
        Cosine score between spectrum2 and spectrum1 is 0.83 with 1 matched peaks
        Cosine score between spectrum2 and spectrum2 is 1.00 with 3 matched peaks

    Parameters
    ----------
    references
        List of reference objects
    queries
        List of query objects
    similarity_function
        Function which accepts a reference + query object and returns a score or tuple of scores
    is_symmetric
        Set to True when *references* and *queries* are identical (as for instance for an all-vs-all
        comparison). By using the fact that score[i,j] = score[j,i] the calculation will be about
        2x faster. Default is False.

    Returns
    -------

    ~matchms.Scores.Scores
    """

    return Scores(references=references, queries=queries,
                  similarity_function=similarity_function,
                  is_symmetric=is_symmetric).calculate()
