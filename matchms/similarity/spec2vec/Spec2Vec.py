import gensim
import numpy
import scipy


class Spec2Vec:

    def __init__(self, model=None, documents=None):
        self.model = model
        self.dictionary = gensim.corpora.Dictionary([d.words for d in documents])

    def __call__(self, query, reference):
        query_vector = self.calc_vector(query)
        reference_vector = self.calc_vector(reference)
        cdist = scipy.spatial.distance.cosine(query_vector, reference_vector)

        return 1 - cdist

    def calc_vector(self, document, intensity_weighting_power=None):
        """Derive latent space vector for entire document."""
        word_vectors = self.model.wv[document.words]
        if intensity_weighting_power:
            vector_size = self.model.wv.vector_size
            word_weights = numpy.power(document.weights, intensity_weighting_power)
            # word_weights = word_weights/numpy.sum(word_weights)  # normalize weights? better not
            return numpy.mean(word_vectors * numpy.tile(word_weights, (vector_size, 1)).T, 0)

        return numpy.mean(word_vectors, 0)
