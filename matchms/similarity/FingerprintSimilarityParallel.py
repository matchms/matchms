import numba
import numpy


class FingerprintSimilarityParallel:
    """Calculate similarity between molecules based on their fingerprints.

    Args:
    ----
    set_empty_scores: "nan" or 0
        Set values to this value if no fingerprint is found.
    """
    def __init__(self, set_empty_scores="nan"):
        self.set_empty_scores = set_empty_scores

    def __call__(self, references, queries):
        """Calculate matrix of fingerprint based similarity scores."""
        def get_fingerprints(spectrums):
            for index, spectrum in enumerate(spectrums):
                yield index, spectrum.get("fingerprint")

        def collect_fingerprints(spectrums):
            """Collect fingerprints and indices of spectrum with finterprints."""
            with_fingerprints = []
            without_fingerprints = []
            fingerprints = []
            for index, fp in get_fingerprints(spectrums):
                if fp is not None:
                    with_fingerprints.append(index)
                    fingerprints.append(fp)
                else:
                    without_fingerprints.append(index)
            fingerprints = numpy.array(fingerprints)
            return numpy.array(fingerprints), numpy.array(with_fingerprints), numpy.array(without_fingerprints)

        fingerprints1, with_fp1, without_fp1 = collect_fingerprints(references)
        fingerprints2, with_fp2, without_fp2 = collect_fingerprints(queries)
        assert with_fp1.size > 0 and with_fp2.size > 0, ("Not enouth molecular fingerprints.",
                                                         "Apply 'add_fingerprint'filter first.")
        return fingerprint_cosine_score_matrix(fingerprints1, with_fp1, without_fp1,
                                               fingerprints2, with_fp2, without_fp2,
                                               set_empty_scores=self.set_empty_scores)


@numba.njit
def cosine_similarity_numba(u: numpy.ndarray, v: numpy.ndarray):
    """Calculate cosine similarity."""
    assert u.shape[0] == v.shape[0], "Input vector must have same shape."
    uv = 0
    uu = 0
    vv = 0
    for i in range(u.shape[0]):
        uv += u[i] * v[i]
        uu += u[i] * u[i]
        vv += v[i] * v[i]
    cos_theta = 1
    if uu != 0 and vv != 0:
        cos_theta = uv / numpy.sqrt(uu * vv)
    return cos_theta


@numba.njit
def fingerprint_cosine_score_matrix(fingerprints1, with_fingerprints1, without_fingerprints1,
                                    fingerprints2, with_fingerprints2, without_fingerprints2,
                                    set_empty_scores="nan"):
    """Calculate cosine scores between fingerprints."""
    size1 = len(with_fingerprints1) + len(without_fingerprints1)
    size2 = len(with_fingerprints2) + len(without_fingerprints2)
    scores = numpy.zeros((size1, size2))
    for i, index1 in enumerate(with_fingerprints1):
        for j, index2 in enumerate(with_fingerprints2):
            scores[index1, index2] = cosine_similarity_numba(fingerprints1[i], fingerprints2[j])
    if set_empty_scores == "nan":
        for index1 in without_fingerprints1:
            scores[index1, :] = numpy.nan
        for index2 in without_fingerprints2:
            scores[:, index2] = numpy.nan
    elif set_empty_scores == 0:
        pass
    else:
        print("Unknown entry for set_empty_scores.")
    return scores
