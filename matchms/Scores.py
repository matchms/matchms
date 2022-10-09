from __future__ import annotations
import json
import pickle
import numpy as np
import numpy.lib.recfunctions
from deprecated.sphinx import deprecated
from matchms.exporting.save_as_json import ScoresJSONEncoder
from matchms.importing.load_from_json import scores_json_decoder
from matchms.similarity import get_similarity_function_by_name
from matchms.similarity.BaseSimilarity import BaseSimilarity
from matchms.typing import QueriesType, ReferencesType


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
        Scores._validate_input_arguments(references, queries, similarity_function)

        self.n_rows = len(references)
        self.n_cols = len(queries)
        self.references = np.asarray(references)
        self.queries = np.asarray(queries)
        self.similarity_function = similarity_function
        self.is_symmetric = is_symmetric
        self._scores = np.empty([self.n_rows, self.n_cols], dtype="object")
        self._index = 0

    def __eq__(self, other):
        if isinstance(other, Scores):
            if self.n_rows != other.n_rows or self.n_cols != other.n_cols:
                return False
            if not np.array_equal(self.references, other.references):
                return False
            if not np.array_equal(self.queries, other.queries):
                return False
            if self.similarity_function.__class__ != other.similarity_function.__class__:
                return False
            if self._scores.dtype != other._scores.dtype:
                return False
            if not np.array_equal(self._scores, other._scores):
                return False
            return True
        return NotImplemented

    def __iter__(self):
        return self

    def __next__(self):
        if self._index < self.scores.size:
            # pylint: disable=unbalanced-tuple-unpacking
            r, c = np.unravel_index(self._index, self._scores.shape)
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
        assert isinstance(references, (list, tuple, np.ndarray)), \
            "Expected input argument 'references' to be list or tuple or np.ndarray."

        assert isinstance(queries, (list, tuple, np.ndarray)), \
            "Expected input argument 'queries' to be list or tuple or np.ndarray."

        assert isinstance(similarity_function, BaseSimilarity), \
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
                            sort: bool = False) -> np.ndarray:
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
        selected_idx = int(np.where(self.references == reference)[0])
        if sort:
            query_idx_sorted = self.similarity_function.sort(self._scores[selected_idx, :])
            return list(zip(self.queries[query_idx_sorted],
                            self._scores[selected_idx, query_idx_sorted].copy()))
        return list(zip(self.queries, self._scores[selected_idx, :].copy()))

    def scores_by_query(self, query: QueriesType, sort: bool = False) -> np.ndarray:
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
        selected_idx = int(np.where(self.queries == query)[0])
        if sort:
            references_idx_sorted = self.similarity_function.sort(self._scores[:, selected_idx])
            return list(zip(self.references[references_idx_sorted],
                            self._scores[references_idx_sorted, selected_idx].copy()))
        return list(zip(self.references, self._scores[:, selected_idx].copy()))

    def to_json(self, filename: str):
        """Export :py:class:`~matchms.Scores.Scores` to a JSON file.

        Parameters
        ----------
        filename
            Path to file to write to
        """
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(self, f, cls=ScoresJSONEncoder)

    def to_pickle(self, filename: str):
        """Export :py:class:`~matchms.Scores.Scores` to a Pickle file.

        Parameters
        ----------
        filename
            Path to file to write to
        """
        with open(filename, "wb") as f:
            pickle.dump(self, f)

    def to_dict(self) -> dict:
        """Return a dictionary representation of scores."""
        return {"__Scores__": True,
                "similarity_function": self.similarity_function.to_dict(),
                "is_symmetric": self.is_symmetric,
                "references": [reference.to_dict() for reference in self.references],
                "queries": [query.to_dict() for query in self.queries] if not self.is_symmetric else None,
                "scores": self.scores.tolist()}

    @property
    def scores(self) -> np.ndarray:
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


class ScoresBuilder:
    """
    Builder class for :class:`~matchms.Scores`.
    """

    def __init__(self):
        self.references = None
        self.queries = None
        self.similarity_function = None
        self.is_symmetric = None
        self.scores = None

    def build(self) -> Scores:
        """
        Build scores object
        """
        scores = Scores(references=self.references,
                        queries=self.queries,
                        similarity_function=self.similarity_function,
                        is_symmetric=self.is_symmetric)
        scores._scores = self.scores  # pylint: disable=protected-access
        return scores

    def from_json(self, file_path: str):
        """
        Import scores data from a JSON file.

        Parameters
        ----------
        file_path
            Path to the scores file.
        """
        with open(file_path, "rb") as f:
            scores_dict = json.load(f, object_hook=scores_json_decoder)

        self._validate_json_input(scores_dict)

        self.is_symmetric = scores_dict["is_symmetric"]
        self.similarity_function = self._construct_similarity_function(scores_dict["similarity_function"])
        self.references = scores_dict["references"]
        self.queries = scores_dict["queries"] if not self.is_symmetric else self.references
        self.scores = self._restructure_scores(scores_dict["scores"])

        return self

    def _restructure_scores(self, scores: dict) -> np.ndarray:
        """
        Restructure scores from a nested list to a numpy array. If scores were stored as an array of tuples, restores
        their original form.
        """
        scores = np.array(scores)

        if len(scores.shape) > 2:
            dt = np.dtype(self.similarity_function.score_datatype)
            return numpy.lib.recfunctions.unstructured_to_structured(scores, dtype=dt)
        return scores

    @staticmethod
    def _construct_similarity_function(similarity_function_dict: dict) -> BaseSimilarity:
        """
        Construct similarity function from its serialized form.
        """
        similarity_function_class = get_similarity_function_by_name(similarity_function_dict.pop("__Similarity__"))
        return similarity_function_class(**similarity_function_dict)

    @staticmethod
    def _validate_json_input(scores_dict: dict):
        if {"__Scores__", "similarity_function", "is_symmetric", "references", "queries", "scores"} != scores_dict.keys():
            raise ValueError("Scores JSON file does not match the expected schema.\n\
                             Make sure the file contains the following keys:\n\
                             ['__Scores__', 'similarity_function', 'is_symmetric', 'references', 'queries', 'scores']")
