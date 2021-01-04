from __future__ import annotations
import numpy
from deprecated.sphinx import deprecated
from matchms.similarity.BaseSimilarity import BaseSimilarity
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
                 similarity_function: BaseSimilarity, is_symmetric: bool = False):
        """

        Parameters
        ----------
        references
            List of reference objects
        queries
            List of query objects
        similarity_function
            Expected input is an object based on :class:`~matchms.similarity.BaseSimilarity`. It is
            expected to provide a *.pair()* and *.matrix()* method for computing similarity scores between
            references and queries.
        is_symmetric
            Set to True when *references* and *queries* are identical (as for instance for an all-vs-all
            comparison). By using the fact that score[i,j] = score[j,i] the calculation will be about
            2x faster. Default is False.
        """
        # pylint: disable=too-many-arguments
        Scores._validate_input_arguments(references, queries, similarity_function)

        self.n_rows = len(references)
        self.n_cols = len(queries)
        self.references = numpy.asarray(references)
        self.queries = numpy.asarray(queries)
        self.similarity_function = similarity_function
        self.is_symmetric = is_symmetric
        self._scores = numpy.empty([self.n_rows, self.n_cols], dtype="object")
        self._index = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self._index < self.scores.size:
            # pylint: disable=unbalanced-tuple-unpacking
            r, c = numpy.unravel_index(self._index, self._scores.shape)
            self._index += 1
            result = self._scores[r, c]
            if not isinstance(result, tuple):
                result = (result,)
            return (self.references[r], self.queries[c]) + result
        self._index = 0
        raise StopIteration

    def __str__(self):
        return self._scores.__str__()

    @staticmethod
    def _validate_input_arguments(references, queries, similarity_function):
        assert isinstance(references, (list, tuple, numpy.ndarray)),\
            "Expected input argument 'references' to be list or tuple or numpy.ndarray."

        assert isinstance(queries, (list, tuple, numpy.ndarray)),\
            "Expected input argument 'queries' to be list or tuple or numpy.ndarray."

        assert isinstance(similarity_function, BaseSimilarity),\
            "Expected input argument 'similarity_function' to have BaseSimilarity as super-class."

    @deprecated(version='0.6.0', reason="Calculate scores via calculate_scores() function.")
    def calculate(self) -> Scores:
        """
        Calculate the similarity between all reference objects v all query objects using
        the most suitable available implementation of the given similarity_function.
        Advised method to calculate similarity scores is :meth:`~matchms.calculate_scores`.
        """
        if self.n_rows == self.n_cols == 1:
            self._scores[0, 0] = self.similarity_function.pair(self.references[0],
                                                               self.queries[0])
        else:
            self._scores = self.similarity_function.matrix(self.references,
                                                           self.queries,
                                                           is_symmetric=self.is_symmetric)
        return self

    def scores_by_reference(self, reference: ReferencesType,
                            sort: bool = False) -> numpy.ndarray:
        """Return all scores for the given reference spectrum.

        Parameters
        ----------
        reference
            Single reference Spectrum.
        sort
            Set to True to obtain the scores in a sorted way (relying on the
            :meth:`~.BaseSimilarity.sort` function from the given similarity_function).
        """
        assert reference in self.references, "Given input not found in references."
        selected_idx = int(numpy.where(self.references == reference)[0])
        if sort:
            query_idx_sorted = self.similarity_function.sort(self._scores[selected_idx, :])
            return list(zip(self.queries[query_idx_sorted],
                            self._scores[selected_idx, query_idx_sorted].copy()))
        return list(zip(self.queries, self._scores[selected_idx, :].copy()))

    def scores_by_query(self, query: QueriesType, sort: bool = False) -> numpy.ndarray:
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
        sort
            Set to True to obtain the scores in a sorted way (relying on the
            :meth:`~.BaseSimilarity.sort` function from the given similarity_function).

        """
        assert query in self.queries, "Given input not found in queries."
        selected_idx = int(numpy.where(self.queries == query)[0])
        if sort:
            references_idx_sorted = self.similarity_function.sort(self._scores[:, selected_idx])
            return list(zip(self.references[references_idx_sorted],
                            self._scores[references_idx_sorted, selected_idx].copy()))
        return list(zip(self.references, self._scores[:, selected_idx].copy()))

    @property
    def scores(self) -> numpy.ndarray:
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
        return self._scores.copy()
