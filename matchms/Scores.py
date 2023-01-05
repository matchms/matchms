from __future__ import annotations
import json
import pickle
import numpy as np
from numpy.lib.recfunctions import unstructured_to_structured
from scipy.sparse import coo_matrix
from sparsestack import StackedSparseArray
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
                  f" is {score[0]:.2f} with {score[1]} matched peaks")

    Should output

    .. testoutput::

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
        self._scores = StackedSparseArray(self.n_rows, self.n_cols)
        self._index = 0

    def __eq__(self, other):
        if isinstance(other, Scores):
            if self.n_rows != other.n_rows or self.n_cols != other.n_cols:
                return False
            if not np.array_equal(self.references, other.references):
                return False
            if not np.array_equal(self.queries, other.queries):
                return False
            if  self._scores != other._scores:
                return False
            return True
        return NotImplemented

    def __iter__(self):
        return self

    def __next__(self):
        if self._index < len(self._scores.col):
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
            "Expected input argument 'references' to be list or tuple or np.ndarray."

        assert isinstance(queries, (list, tuple, np.ndarray)),\
            "Expected input argument 'queries' to be list or tuple or np.ndarray."

    def calculate(self, similarity_function: BaseSimilarity,
                  name: str = None,
                  array_type: str = "numpy",
                  join_type="left") -> Scores:
        """
        Calculate the similarity between all reference objects vs all query objects using
        the most suitable available implementation of the given similarity_function.
        If Scores object already contains similarity scores, the newly computed measures
        will be added to a new layer (name --> layer name).
        Additional scores will be added as specified with join_type, the default being 'left'.

        Parameters
        ----------
        similarity_function
            Function which accepts a reference + query object and returns a score or tuple of scores
        is_symmetric
            Set to True when *references* and *queries* are identical (as for instance for an all-vs-all
            comparison). By using the fact that score[i,j] = score[j,i] the calculation will be about
            2x faster. Default is False.
        array_type
            Specify the type of array to store and compute the scores. Choose from "numpy" or "sparse".
        join_mode
            Choose from left, right, outer, inner to specify the merge type.
        """
        def is_sparse_advisable():
            return (
                (len(self._scores.score_names) > 0)  # already scores in Scores
                and (join_type in ["inner", "left"])  # inner/left join
                and (len(self._scores.row) < (self.n_rows * self.n_cols)/2)  # fewer than half of scores have entries
                )

        if name is None:
            name = similarity_function.__class__.__name__
        if (self.n_rows == 0) or (self.n_cols == 0):
            raise ValueError("Number of elements must be >= 1")
        if self.n_rows == self.n_cols == 1:
            score = similarity_function.pair(self.references[0],
                                             self.queries[0])
            self._scores.add_dense_matrix(np.array([score]), name)
        elif is_sparse_advisable():
            new_scores = similarity_function.sparse_array(references=self.references,
                                                          queries=self.queries,
                                                          idx_row=self._scores.row,
                                                          idx_col=self._scores.col,
                                                          is_symmetric=self.is_symmetric)
            self._scores.add_sparse_data(self._scores.row,
                                         self._scores.col,
                                         new_scores,
                                         name)
        else:
            new_scores = similarity_function.matrix(self.references,
                                                    self.queries,
                                                    array_type=array_type,
                                                    is_symmetric=self.is_symmetric)
            if isinstance(new_scores, np.ndarray):
                self._scores.add_dense_matrix(new_scores, name, join_type=join_type)
            elif len(new_scores.score_names) == 1:
                new_scores.data.dtype.names = [name]
                self._scores.add_sparse_data(new_scores.row,
                                             new_scores.col,
                                             new_scores.data, "", join_type=join_type)
            else:
                self._scores.add_sparse_data(new_scores.row,
                                             new_scores.col,
                                             new_scores.data, name, join_type=join_type)
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
        if name is None and len(self.score_names) > 1 and sort is True:
            raise IndexError("For sorting, score must be specified")
        assert reference in self.references, "Given input not found in references."
        selected_idx = int(np.where(self.references == reference)[0])
        _, r, scores_for_ref = self._scores[selected_idx, :]
        if sort:
            if name is None:
                name = self._scores.guess_score_name()
            if scores_for_ref.dtype.type == np.void:
                query_idx_sorted = np.argsort(scores_for_ref[name])[::-1]
            else:
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
            selected_scores = scores.scores_by_query(spectrum_4, 'CosineGreedy_score', sort=True)
            print([x[1][0].round(3) for x in selected_scores])

        Should output

        .. testoutput::

            [0.796, 0.613]

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
        if name is None and len(self.score_names) > 1 and sort is True:
            raise IndexError("For sorting, score must be specified")
        assert query in self.queries, "Given input not found in queries."
        selected_idx = int(np.where(self.queries == query)[0])
        c, _, scores_for_query = self._scores[:, selected_idx]
        if sort:
            if name is None:
                name = self._scores.guess_score_name()
            # TODO: add option to use other sorting algorithm
            if scores_for_query.dtype.type == np.void:
                references_idx_sorted = np.argsort(scores_for_query[name])[::-1]
            else:
                references_idx_sorted = np.argsort(scores_for_query)[::-1]
            return list(zip(self.references[c[references_idx_sorted]],
                            scores_for_query[references_idx_sorted].copy()))
        return list(zip(self.references[c], scores_for_query.copy()))

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
        scores_dict = {"__Scores__": True,
                       "is_symmetric": self.is_symmetric,
                       "references": [reference.to_dict() for reference in self.references],
                       "queries": [query.to_dict() for query in self.queries] if not self.is_symmetric else None}
        scores_dict.update(self.scores.to_dict())
        return scores_dict

    @property
    def shape(self):
        return self._scores.shape

    @property
    def score_names(self):
        return self._scores.score_names

    @property
    def scores(self):
        return self._scores

    def filter_by_range(self, inplace=False, **kwargs):
        """Remove all scores for which the score `name` is outside the given range.

        Parameters
        ----------
        inplace
            Default is False in which case a filtered scores object will be returned.
            Set to True to change the scores array in-place.
        name
            Name of the score which is used for filtering. Run `.score_names` to
            see all scores scored in the sparse array.
        low
            Lower threshold below which all scores will be removed.
        high
            Upper threshold above of which all scores will be removed.
        above_operator
            Define operator to be used to compare against `low`. Default is '>'.
            Possible choices are '>', '<', '>=', '<='.
        below_operator
            Define operator to be used to compare against `high`. Default is '<'.
            Possible choices are '>', '<', '>=', '<='.
        """
        if inplace is True:
            self._scores = self._scores.filter_by_range(**kwargs)
            return None
        return self._scores.filter_by_range(**kwargs)

    def to_array(self, name=None) -> np.ndarray:
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

            scores = calculate_scores(spectrums, spectrums, IntersectMz()).to_array()

            print(scores.shape)
            print(scores)

        Should output

        .. testoutput::

             (2, 2)
             [[1.  0.2]
              [0.2 1. ]]

        Parameters
        ----------
        name
            Name of the score that should be returned (if multiple scores are stored).
        """
        return self._scores.to_array(name)

    def to_coo(self, name=None) -> coo_matrix:
        """Scores as scipy sparse COO matrix

        Parameters
        ----------
        name
            Name of the score that should be returned (if multiple scores are stored).
        """
        return self._scores.to_coo(name)


class ScoresBuilder:
    """
    Builder class for :class:`~matchms.Scores`.
    """

    def __init__(self):
        self.references = None
        self.queries = None
        self.is_symmetric = None
        self.scores = None

    def build(self) -> Scores:
        """
        Build scores object
        """
        scores = Scores(references=self.references,
                        queries=self.queries,
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
        self.references = scores_dict["references"]
        self.queries = scores_dict["queries"] if not self.is_symmetric else self.references
        self.scores = self._restructure_scores(scores_dict)

        return self

    def _restructure_scores(self, scores_dict: dict) -> np.ndarray:
        """
        Restructure scores from a nested list to a numpy array. If scores were stored as an array of tuples, restores
        their original form.
        """
        sparsestack = StackedSparseArray(scores_dict.get("n_row"), scores_dict.get("n_col"))
        sparsestack.row = np.array(scores_dict.get("row"))
        sparsestack.col = np.array(scores_dict.get("col"))
        dtype = scores_dict.get("dtype")
        if len(dtype[0]) > 1:
            dtype = [(x[0], x[1]) for x in dtype]
        sparsestack.data = unstructured_to_structured(np.array(scores_dict.get("data")),
                                                      dtype=np.dtype(dtype))
        return sparsestack

    @staticmethod
    def _construct_similarity_functions(similarity_function_dict: dict) -> BaseSimilarity:
        """
        Construct similarity function from its serialized form.
        """
        similarity_function_class = get_similarity_function_by_name(similarity_function_dict.pop("__Similarity__"))
        return similarity_function_class(**similarity_function_dict)

    @staticmethod
    def _validate_json_input(scores_dict: dict):
        if {"__Scores__", "is_symmetric", "references", "queries", "row",
                "col", "data", "dtype", "n_row", "n_col"} != scores_dict.keys():
            raise ValueError("Scores JSON file does not match the expected schema.\n\
                             Make sure the file contains the following keys:\n\
                             ['__Scores__', 'is_symmetric', 'references', 'queries', 'scores_row',\
                             'scores_col', 'scores_data', 'scores_dtype']")
