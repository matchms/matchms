from __future__ import annotations
import numpy as np
from deprecated.sphinx import deprecated
from matchms.similarity.BaseSimilarity import BaseSimilarity
from matchms.StackedSparseScores import StackedSparseScores
from matchms.typing import QueriesType
from matchms.typing import ReferencesType


class Scores:
    """Contains reference and query spectrums and the scores between them.

    The scores can be retrieved as a matrix with the :py:attr:`Scores.scores` attribute.
    The reference spectrum, query spectrum, score pairs can also be iterated over in query then reference order.

    Example to calculate scores between 2 spectrums and iterate over the scores

    .. testcode::

        import numpy as np
        from matchms import calculate_scores
        from matchms import Spectrum
        from matchms.similarity import CosineGreedy

        spectrum_1 = Spectrum(mz=np.array([100, 150, 200.]),
                              intensities=np.array([0.7, 0.2, 0.1]),
                              metadata={'id': 'spectrum1'})
        spectrum_2 = Spectrum(mz=np.array([100, 140, 190.]),
                              intensities=np.array([0.4, 0.2, 0.1]),
                              metadata={'id': 'spectrum2'})
        spectrum_3 = Spectrum(mz=np.array([110, 140, 195.]),
                              intensities=np.array([0.6, 0.2, 0.1]),
                              metadata={'id': 'spectrum3'})
        spectrum_4 = Spectrum(mz=np.array([100, 150, 200.]),
                              intensities=np.array([0.6, 0.1, 0.6]),
                              metadata={'id': 'spectrum4'})
        references = [spectrum_1, spectrum_2]
        queries = [spectrum_3, spectrum_4]

        similarity_measure = CosineGreedy()
        scores = calculate_scores(references, queries, similarity_measure)

        for (reference, query, score) in scores:
            print(f"Cosine score between {reference.get('id')} and {query.get('id')}" +
                  f" is {score['score']:.2f} with {score['matches']} matched peaks")

    Should output

    .. testoutput::

        Cosine score between spectrum1 and spectrum3 is 0.00 with 0 matched peaks
        Cosine score between spectrum1 and spectrum4 is 0.80 with 3 matched peaks
        Cosine score between spectrum2 and spectrum3 is 0.14 with 1 matched peaks
        Cosine score between spectrum2 and spectrum4 is 0.61 with 1 matched peaks
    """
    def __init__(self, references: ReferencesType, queries: QueriesType,
                 is_symmetric: bool = False):
        """

        Parameters
        ----------
        references
            List of reference objects
        queries
            List of query objects
        is_symmetric
            Set to True when *references* and *queries* are identical (as for instance for an all-vs-all
            comparison). By using the fact that score[i,j] = score[j,i] the calculation will be about
            2x faster. Default is False.
        """
        Scores._validate_input_arguments(references, queries)

        self.n_rows = len(references)
        self.n_cols = len(queries)
        self.references = np.asarray(references)
        self.queries = np.asarray(queries)
        self.is_symmetric = is_symmetric
        self._scores = StackedSparseScores(self.n_rows, self.n_cols) #, dtype="object")
        self._index = 0
        self.similarity_functions = {}

    def __iter__(self):
        return self

    def __next__(self):
        if self._index < len(self._scores.col):
            # pylint: disable=unbalanced-tuple-unpacking
            i = self._index
            result = [self._scores.data[name][i] for name in self._scores.score_names]
            if not isinstance(result, tuple):
                result = (result,)
            self._index += 1
            return (self.references[self._scores.row[i]],
                    self.queries[self._scores.col[i]]) + result
        self._index = 0
        raise StopIteration

    def __str__(self):
        return self._scores.__str__()

    @staticmethod
    def _validate_input_arguments(references, queries):
        assert isinstance(references, (list, tuple, np.ndarray)),\
            "Expected input argument 'references' to be list or tuple or numpy.ndarray."

        assert isinstance(queries, (list, tuple, np.ndarray)),\
            "Expected input argument 'queries' to be list or tuple or numpy.ndarray."

    @deprecated(version='0.6.0', reason="Calculate scores via calculate_scores() function.")
    def calculate(self, similarity_function, name=None) -> Scores:
        """
        Calculate the similarity between all reference objects v all query objects using
        the most suitable available implementation of the given similarity_function.
        Advised method to calculate similarity scores is :meth:`~matchms.calculate_scores`.
        """
        if name is None:
            name = similarity_function.__class__.__name__
        self.similarity_functions[name] = similarity_function
        if self.n_rows == self.n_cols == 1:
            score = similarity_function.pair(self.references[0],
                                             self.queries[0])
            self._scores.add_dense_matrix(np.array([score]), name)
        else:
            scores_matrix = similarity_function.matrix(self.references,
                                                       self.queries,
                                                       is_symmetric=self.is_symmetric)
            self._scores.add_dense_matrix(scores_matrix, name)
        return self

    def scores_by_reference(self, reference: ReferencesType,
                            name: str = None, sort: bool = False) -> np.ndarray:
        """Return all scores of given name for the given reference spectrum.

        Parameters
        ----------
        reference
            Single reference Spectrum.
        name
            Name of the score that should be returned (if multiple scores are stored).
        sort
            Set to True to obtain the scores in a sorted way (relying on the
            :meth:`~.BaseSimilarity.sort` function from the given similarity_function).
        """
        if name is None:
            name = self._scores._guess_name()
        assert reference in self.references, "Given input not found in references."
        selected_idx = int(np.where(self.references == reference)[0])
        _, r, scores_for_ref = self._scores[selected_idx, :, name]
        if sort:
            query_idx_sorted = np.argsort(scores_for_ref)[::-1]
            return list(zip(self.queries[r[query_idx_sorted]],
                            scores_for_ref[query_idx_sorted].copy()))
        return list(zip(self.queries[r], scores_for_ref.copy()))

    def scores_by_query(self, query: QueriesType,
                        name: str = None, sort: bool = False) -> np.ndarray:
        """Return all scores for the given query spectrum.

        For example

        .. testcode::

            import numpy as np
            from matchms import calculate_scores, Scores, Spectrum
            from matchms.similarity import CosineGreedy

            spectrum_1 = Spectrum(mz=np.array([100, 150, 200.]),
                                  intensities=np.array([0.7, 0.2, 0.1]),
                                  metadata={'id': 'spectrum1'})
            spectrum_2 = Spectrum(mz=np.array([100, 140, 190.]),
                                  intensities=np.array([0.4, 0.2, 0.1]),
                                  metadata={'id': 'spectrum2'})
            spectrum_3 = Spectrum(mz=np.array([110, 140, 195.]),
                                  intensities=np.array([0.6, 0.2, 0.1]),
                                  metadata={'id': 'spectrum3'})
            spectrum_4 = Spectrum(mz=np.array([100, 150, 200.]),
                                  intensities=np.array([0.6, 0.1, 0.6]),
                                  metadata={'id': 'spectrum4'})
            references = [spectrum_1, spectrum_2, spectrum_3]
            queries = [spectrum_2, spectrum_3, spectrum_4]

            scores = calculate_scores(references, queries, CosineGreedy())
            selected_scores = scores.scores_by_query(spectrum_4, sort=True)
            print([x[1]["score"].round(3) for x in selected_scores])

        Should output

        .. testoutput::

            [0.796, 0.613, 0.0]

        Parameters
        ----------
        query
            Single query Spectrum.
        name
            Name of the score that should be returned (if multiple scores are stored).
        sort
            Set to True to obtain the scores in a sorted way (relying on the
            :meth:`~.BaseSimilarity.sort` function from the given similarity_function).

        """
        assert query in self.queries, "Given input not found in queries."
        selected_idx = int(np.where(self.queries == query)[0])
        c, _, scores_for_query = self._scores[:, selected_idx, name]
        if sort:
            # TODO: add option to use other sorting algorithm
            references_idx_sorted = np.argsort(scores_for_query)[::-1]
            return list(zip(self.references[c[references_idx_sorted]],
                            scores_for_query[references_idx_sorted].copy()))
        return list(zip(self.references[c], scores_for_query.copy()))

    @property
    def shape(self):
        return self._scores.shape

    @property
    def score_names(self):
        return self._scores.score_names    

    @property
    def scores(self):
        return self._scores
        
    def get_scores_array(self, name=None, array_type="numpy") -> np.ndarray:
        """Scores as numpy array

        For example

        .. testcode::

            import numpy as np
            from matchms import calculate_scores, Scores, Spectrum
            from matchms.similarity import IntersectMz

            spectrum_1 = Spectrum(mz=np.array([100, 150, 200.]),
                                  intensities=np.array([0.7, 0.2, 0.1]))
            spectrum_2 = Spectrum(mz=np.array([100, 140, 190.]),
                                  intensities=np.array([0.4, 0.2, 0.1]))
            spectrums = [spectrum_1, spectrum_2]

            scores = calculate_scores(spectrums, spectrums, IntersectMz()).scores

            print(scores[0].dtype)
            print(scores.shape)
            print(scores)

        Should output

        .. testoutput::

             float64
             (2, 2)
             [[1.  0.2]
              [0.2 1. ]]
        """
        if name is None:
            name = self._scores._guess_name()
        if array_type == "numpy":
            return self._scores.to_array(name)
        elif array_type in ["coo", "sparse"]:
            self._scores.to_coo(name)
        else:
            raise TypeError("Unknown type for output matrix")
