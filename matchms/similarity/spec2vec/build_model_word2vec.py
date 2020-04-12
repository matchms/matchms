# https://github.com/matchms/old-iomega-spec2vec/blob/216b8f8b5e4ffd320b4575326a05fb6c7cd28223/matchms/old/similarity_measure.py#L193-L280
import gensim


def build_model_word2vec(corpus):
    return gensim.models.Word2Vec(corpus)
