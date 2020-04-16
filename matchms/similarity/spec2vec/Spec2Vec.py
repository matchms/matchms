import gensim


class Spec2Vec:

    def __init__(self, model=None, documents=None):
        self.model = model
        self.dictionary = gensim.corpora.Dictionary([d.words for d in documents])

    def __call__(self, query, reference):

        def calc_vector(document):
            bag_of_words = self.dictionary.doc2bow(document.words)
            words = [self.dictionary[item[0]] for item in bag_of_words]
            return self.model.wv[words]

        query_vector = calc_vector(query)
        reference_vector = calc_vector(reference)

        return query_vector * reference_vector.T
