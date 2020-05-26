import numpy

from matchms.similarity.vector_similarity_functions import (cosine_similarity_matrix,
                                                            dice_similarity_matrix,
                                                            jaccard_similarity_matrix)


class FingerprintSimilarityParallel:
    """Calculate similarity between molecules based on their fingerprints.

    Args:
    ----
    set_empty_scores: "nan", int, float
        Set values to this value if no fingerprint is found. Default is "nan",
        in which case all similarity values in cases without fingerprint will be
        set to numpy.nan's.
    """
    def __init__(self, similarity_measure="jaccard", set_empty_scores="nan"):
        self.set_empty_scores = set_empty_scores
        assert similarity_measure in ["cosine", "dice", "jaccard"], "Unknown similarity measure."
        self.similarity_measure = similarity_measure

    def __call__(self, references, queries):
        """Calculate matrix of fingerprint based similarity scores."""
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
            fingerprints = numpy.array(fingerprints)
            return numpy.array(fingerprints), numpy.array(idx_fingerprints)

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
