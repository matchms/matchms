import gensim
import numpy
import scipy
from .calc_vector import calc_vector


class Spec2VecParallel:

    def __init__(self, model=None, documents=None, intensity_weighting_power=0):
        self.model = model
        self.dictionary = gensim.corpora.Dictionary([d.words for d in documents])
        self.intensity_weighting_power = intensity_weighting_power
        self.vector_size = model.wv.vector_size

    def __call__(self, references, queries):

        n_rows = len(references)
        reference_vectors = numpy.empty((n_rows, self.vector_size), dtype="float")
        for index_reference, reference in enumerate(references):
            reference_vectors[index_reference, 0:self.vector_size] = calc_vector(self.model,
                                                                                 reference,
                                                                                 self.intensity_weighting_power)
        n_cols = len(queries)
        query_vectors = numpy.empty((n_cols, self.vector_size), dtype="float")
        for index_query, query in enumerate(queries):
            query_vectors[index_query, 0:self.vector_size] = calc_vector(self.model,
                                                                         query,
                                                                         self.intensity_weighting_power)

        cdist = scipy.spatial.distance.cdist(reference_vectors, query_vectors, "cosine")

        return 1 - cdist
