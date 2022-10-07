from typing import List, Union
import numpy as np
from matchms.typing import SpectrumType
from .BaseSimilarity import BaseSimilarity
from .vector_similarity_functions import (cosine_similarity,
                                          cosine_similarity_matrix,
                                          dice_similarity,
                                          dice_similarity_matrix,
                                          jaccard_index,
                                          jaccard_similarity_matrix)


class FingerprintSimilarity(BaseSimilarity):
    """Calculate similarity between molecules based on their fingerprints.

    For this similarity measure to work, fingerprints are expected to be derived
    by running :meth:`~matchms.filtering.add_fingerprint`.

    Code example:

    .. testcode::

        import numpy as np
        from matchms import calculate_scores
        from matchms import Spectrum
        from matchms.filtering import add_fingerprint
        from matchms.similarity import FingerprintSimilarity

        spectrum_1 = Spectrum(mz=np.array([], dtype="float"),
                              intensities=np.array([], dtype="float"),
                              metadata={"smiles": "CCC(C)C(C(=O)O)NC(=O)CCl"})

        spectrum_2 = Spectrum(mz=np.array([], dtype="float"),
                              intensities=np.array([], dtype="float"),
                              metadata={"smiles": "CC(C)C(C(=O)O)NC(=O)CCl"})

        spectrum_3 = Spectrum(mz=np.array([], dtype="float"),
                              intensities=np.array([], dtype="float"),
                              metadata={"smiles": "C(C(=O)O)(NC(=O)O)S"})

        spectrums = [spectrum_1, spectrum_2, spectrum_3]
        # Add fingerprints
        spectrums = [add_fingerprint(x, nbits=256) for x in spectrums]

        # Specify type and calculate similarities
        similarity_measure = FingerprintSimilarity("jaccard")
        scores = calculate_scores(spectrums, spectrums, similarity_measure)
        print(np.round(scores.scores.to_array(), 3))

    Should output

    .. testoutput::

        [[1.    0.878 0.415]
         [0.878 1.    0.444]
         [0.415 0.444 1.   ]]

    """
    # Set key characteristics as class attributes
    is_commutative = True
    # Set output data type, e.g.  "float" or [("score", "float"), ("matches", "int")]
    score_datatype = np.float64

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
            np.nan's in such cases.
        """
        self.set_empty_scores = set_empty_scores
        assert similarity_measure in ["cosine", "dice", "jaccard"], "Unknown similarity measure."
        self.similarity_measure = similarity_measure

    def pair(self, reference: SpectrumType, query: SpectrumType) -> float:
        """Calculate fingerprint based similarity score between two spectra.

        Parameters
        ----------
        reference
            Single reference spectrum.
        query
            Single query spectrum.
        """
        fingerprint_ref = reference.get("fingerprint")
        fingerprint_query = query.get("fingerprint")
        if self.similarity_measure == "jaccard":
            return jaccard_index(fingerprint_ref, fingerprint_query)

        if self.similarity_measure == "dice":
            return dice_similarity(fingerprint_ref, fingerprint_query)

        if self.similarity_measure == "cosine":
            score = cosine_similarity(fingerprint_ref, fingerprint_query)
            return np.asarray(score, dtype=self.score_datatype)

        raise NotImplementedError

    def matrix(self, references: List[SpectrumType], queries: List[SpectrumType],
               array_type: str = "numpy",
               is_symmetric: bool = False) -> np.array:
        """Calculate matrix of fingerprint based similarity scores.

        Parameters
        ----------
        references:
            List of reference spectrums.
        queries:
            List of query spectrums.
        array_type
            Specify the output array type. Can be "numpy" or "sparse".
            Default is "numpy" and will return a numpy array. "sparse" will return a COO-sparse array
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
            return np.asarray(fingerprints), np.asarray(idx_fingerprints)

        def create_full_matrix():
            """Create matrix for all similarities."""
            similarity_matrix = np.zeros((len(references), len(queries)))
            if self.set_empty_scores == "nan":
                similarity_matrix[:] = np.nan
            elif isinstance(self.set_empty_scores, (float, int)):
                similarity_matrix[:] = self.set_empty_scores
            return similarity_matrix

        if array_type != "numpy":
            raise NotImplementedError("Output array type other than numpy is not yet implemented.")
        fingerprints1, idx_fingerprints1 = collect_fingerprints(references)
        fingerprints2, idx_fingerprints2 = collect_fingerprints(queries)
        assert idx_fingerprints1.size > 0 and idx_fingerprints2.size > 0, ("Not enouth molecular fingerprints.",
                                                                           "Apply 'add_fingerprint'filter first.")

        # Calculate similarity score matrix following specified method
        similarity_matrix = create_full_matrix()
        if self.similarity_measure == "jaccard":
            similarity_matrix[np.ix_(idx_fingerprints1,
                                        idx_fingerprints2)] = jaccard_similarity_matrix(fingerprints1,
                                                                                        fingerprints2)
        elif self.similarity_measure == "dice":
            similarity_matrix[np.ix_(idx_fingerprints1,
                                        idx_fingerprints2)] = dice_similarity_matrix(fingerprints1,
                                                                                     fingerprints2)
        elif self.similarity_measure == "cosine":
            similarity_matrix[np.ix_(idx_fingerprints1,
                                        idx_fingerprints2)] = cosine_similarity_matrix(fingerprints1,
                                                                                       fingerprints2)
        return similarity_matrix.astype(self.score_datatype)
