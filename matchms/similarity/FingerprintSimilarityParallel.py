from typing import List
from typing import Union
import numpy
from matchms.similarity.vector_similarity_functions import \
    cosine_similarity_matrix
from matchms.similarity.vector_similarity_functions import \
    dice_similarity_matrix
from matchms.similarity.vector_similarity_functions import \
    jaccard_similarity_matrix
from matchms.typing import SpectrumType


class FingerprintSimilarityParallel:
    """Calculate similarity between molecules based on their fingerprints.

    For this similarity measure to work, fingerprints are expected to be derived
    by running the "add_fingerprint()" filter function.
    """
    def __init__(self, similarity_measure: str = "jaccard",
                 set_empty_scores: Union[float, int, str] = "nan"):
        """

        Parameters
        ----------
        similarity_measure:
            Chose similarity measure form "cosine", "dice", "jaccard".
            The default is "jaccard".
        set_empty_scores:
            Define what should be given instead of a similarity score in cases
            where fingprints are missing. The default is "nan", which will return
            numpy.nan's in such cases.
        """
        self.set_empty_scores = set_empty_scores
        assert similarity_measure in ["cosine", "dice", "jaccard"], "Unknown similarity measure."
        self.similarity_measure = similarity_measure

    def __call__(self, references: List[SpectrumType], queries: List[SpectrumType]) -> numpy.array:
        """Calculate matrix of fingerprint based similarity scores.

        Parameters
        ----------
        references:
            List of reference spectrums.
        queries:
            List of query spectrums.
        """
        def get_fingerprints(spectrums):
            for index, spectrum in enumerate(spectrums):
                yield index, spectrum.get("fingerprint")

        def collect_fingerprints(spectrums):
            """Collect fingerprints and indices of spectrum with finterprints."""
            idx_fingerprints = []
            fingerprints = []
            for index, fp in get_fingerprints(spectrums):
                if fp is not None:
                    idx_fingerprints.append(index)
                    fingerprints.append(fp)
            return numpy.asarray(fingerprints), numpy.asarray(idx_fingerprints)

        def create_full_matrix():
            """Create matrix for all similarities."""
            similarity_matrix = numpy.zeros((len(references), len(queries)))
            if self.set_empty_scores == "nan":
                similarity_matrix[:] = numpy.nan
            elif isinstance(self.set_empty_scores, (float, int)):
                similarity_matrix[:] = self.set_empty_scores
            return similarity_matrix

        fingerprints1, idx_fingerprints1 = collect_fingerprints(references)
        fingerprints2, idx_fingerprints2 = collect_fingerprints(queries)
        assert idx_fingerprints1.size > 0 and idx_fingerprints2.size > 0, ("Not enouth molecular fingerprints.",
                                                                           "Apply 'add_fingerprint'filter first.")

        # Calculate similarity score matrix following specified method
        similarity_matrix = create_full_matrix()
        if self.similarity_measure == "jaccard":
            similarity_matrix[numpy.ix_(idx_fingerprints1,
                                        idx_fingerprints2)] = jaccard_similarity_matrix(fingerprints1,
                                                                                        fingerprints2)
        elif self.similarity_measure == "dice":
            similarity_matrix[numpy.ix_(idx_fingerprints1,
                                        idx_fingerprints2)] = dice_similarity_matrix(fingerprints1,
                                                                                     fingerprints2)
        elif self.similarity_measure == "cosine":
            similarity_matrix[numpy.ix_(idx_fingerprints1,
                                        idx_fingerprints2)] = cosine_similarity_matrix(fingerprints1,
                                                                                       fingerprints2)
        return similarity_matrix
