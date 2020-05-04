import gensim
from numpy import asarray, power, tile, sum
import scipy


class Spec2Vec:

    def __init__(self, model=None, documents=None, weigh_by_intensity=False, intensity_weighting_power=1):
        self.model = model
        self.dictionary = gensim.corpora.Dictionary([d.words for d in documents])
        self.weigh_by_intensity = weigh_by_intensity
        self.intensity_weighting_power = intensity_weighting_power
        self.vector_size = model.wv.vector_size

    def __call__(self, query, reference):

        def calc_vector(document):
            """Derive latent space vector for entire document."""
            word_vectors = self.model.wv[document.words]
            if self.weigh_by_intensity is True:
                weights = asarray(document.weights).reshape(len(document), 1)
                weights_raised = power(weights, self.intensity_weighting_power)
                weights_raised_tiled = tile(weights_raised, (1, self.vector_size))
                vector = sum(word_vectors * weights_raised_tiled, 0)
            else:
                vector = sum(word_vectors, 0)
            return vector

        query_vector = calc_vector(query)
        reference_vector = calc_vector(reference)
        cdist = scipy.spatial.distance.cosine(query_vector, reference_vector)

        return 1 - cdist
