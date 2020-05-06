import gensim
import numpy
import scipy


class Spec2Vec:

    def __init__(self, model=None, documents=None, intensity_weighting_power=0):
        self.model = model
        self.dictionary = gensim.corpora.Dictionary([d.words for d in documents])
        self.intensity_weighting_power = intensity_weighting_power
        self.vector_size = model.wv.vector_size

    def __call__(self, query, reference):

        def calc_vector(document):
            """Derive latent space vector for entire document."""
            word_vectors = self.model.wv[document.words]
            weights = numpy.asarray(document.weights).reshape(len(document), 1)
            weights_raised = numpy.power(weights, self.intensity_weighting_power)
            weights_raised_tiled = numpy.tile(weights_raised, (1, self.vector_size))
            vector = sum(word_vectors * weights_raised_tiled, 0)
            return vector

        query_vector = calc_vector(query)
        reference_vector = calc_vector(reference)
        cdist = scipy.spatial.distance.cosine(query_vector, reference_vector)

        return 1 - cdist
