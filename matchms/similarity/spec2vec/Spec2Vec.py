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

    def calc_vector(self, document):
        """Derive latent space vector for entire document."""
        bag_of_words = self.dictionary.doc2bow(document.words)
        words = [self.dictionary[item[0]] for item in bag_of_words]
        word_vectors = self.model.wv[words]

        return numpy.mean(word_vectors, 0)
