import gensim
import scipy
from .calc_vector import calc_vector


class Spec2Vec:

    def __init__(self, model=None, documents=None, intensity_weighting_power=0):
        self.model = model
        self.dictionary = gensim.corpora.Dictionary([d.words for d in documents])
        self.intensity_weighting_power = intensity_weighting_power
        self.vector_size = model.wv.vector_size

    def __call__(self, reference, query):

        reference_vector = calc_vector(self.model, reference, self.intensity_weighting_power)
        query_vector = calc_vector(self.model, query, self.intensity_weighting_power)
        cdist = scipy.spatial.distance.cosine(reference_vector, query_vector)

        return 1 - cdist
