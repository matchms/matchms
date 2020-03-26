# test functions
import os
import numpy as np
import pytest
import unittest

from matchms.MS_functions import Spectrum, load_MGF_data
from matchms.similarity_measure import SimilarityMeasures

# Use test data from following folder
PATH_TESTDATA = os.path.join(os.path.dirname(__file__), 'testdata')

class ModelGenerationSuite(unittest.TestCase):
    """Basic test cases."""

    def test_SimilarityMeasures(self):
        """ Test importing spectra, calculating cosine score matrix and modified
        cosine matrix. """
        # Import spectra
        test_mgf_file = os.path.join(PATH_TESTDATA, 'GNPS-COLLECTIONS-PESTICIDES-NEGATIVE.mgf')
        spectra, spec_dict, MS_docs, MS_docs_intensity, metadata = load_MGF_data(test_mgf_file,
                                            file_json = None,
                                            num_decimals = 1,
                                            min_frag = 0.0, max_frag = 1000.0,
                                            min_loss = 5.0, max_loss = 500.0,
                                            min_intensity_perc = 0,
                                            exp_intensity_filter = 0.8,
                                            min_keep_peaks_0 = 10,
                                            min_keep_peaks_per_mz = 20/200,
                                            min_peaks = 5,
                                            max_peaks = None,
                                            peak_loss_words = ['peak_', 'loss_'])

        # Create SimilarityMeasures object
        MS_measure = SimilarityMeasures(MS_docs)
        MS_measure.preprocess_documents(0.2, min_frequency = 2, create_stopwords = False)

        assert len(MS_measure.dictionary) == 1237, 'expected different number of words in dictionary'
        assert MS_measure.corpus[0][-5:] == ['loss_70.1', 'loss_88.1', 'loss_88.1', 'loss_88.1', 'loss_108.1']

        # Train a word2vec model
        # -----------------------------------------------------------------------------
        file_model = os.path.join(PATH_TESTDATA, 'Spec2Vec_model.model')
        vector_dimension = 100

        MS_measure.build_model_word2vec(file_model, size=vector_dimension, window=500,
                                     min_count=1, workers=4, iterations=20,
                                     use_stored_model=False)

        assert os.path.isfile(file_model[:-6] + '_iter_20.model') == True, 'Word2Vec model was not saved as expected.'
        os.remove(file_model[:-6] + '_iter_20.model')

        # Test loading pre-trained word2vec model
        file_model_test = os.path.join(PATH_TESTDATA, 'Spec2Vec_model_test.model')
        MS_measure.build_model_word2vec(file_model_test,
                                        size=vector_dimension, window=500,
                                     min_count=1, workers=4, iterations=20,
                                     use_stored_model=True)

        assert MS_measure.model_word2vec.wv.vectors.shape == (1237, 100), 'unexpected number or shape of word2vec vectors'
        assert np.sum(MS_measure.model_word2vec.wv.vectors[0]) == pytest.approx(20.86065, 0.0001), 'unexpected values for word2vec vectors'


        # Calculate Spec2Vec vectors for all spectra
        # -----------------------------------------------------------------------------
        MS_measure.get_vectors_centroid(method = 'ignore',
                                     tfidf_weighted=True,
                                     weighting_power = 0.5,
                                     tfidf_model = None)

        assert MS_measure.vectors_centroid.shape == (76, vector_dimension)


        # Calculate matrix of all-vs-all similarity scores:
        # -----------------------------------------------------------------------------
        from scipy import spatial

        M_sim = 1 - spatial.distance.cdist(MS_measure.vectors_centroid,
                                           MS_measure.vectors_centroid, 'cosine')

        assert np.mean(M_sim.diagonal()) == 1.0, 'diagonal values of all-vs-all similarity matrix should be 1'
        assert np.max(M_sim) <= 1.0, 'similarity matrix cannot contain values > 1'

if __name__ == '__main__':
    unittest.main()




