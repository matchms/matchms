# test functions

import numpy as np
from helper_functions import calculate_similarities, preprocess_document

def test_preprocess_document():
    # Test with test-vectors and known outcome
    corpus = [['aa', 'BB', 'cc'], ['aA', 'Bb', 'cC', 'DD']] 
    corpus_weights = [[1, 2, 3], [5, 5, 5, 5]] 
    
    # Run function:
    corpus_lowered, corpus_weights_new = preprocess_document(corpus, 
                                                            corpus_weights = None, 
                                                            stopwords = [], 
                                                            min_frequency = 1)
    
    assert corpus_lowered[1] == ['aa', 'bb', 'cc', 'dd']     
    assert corpus_weights_new is None
    
    corpus_lowered, corpus_weights_new = preprocess_document(corpus, 
                                                            corpus_weights = corpus_weights, 
                                                            stopwords = [], 
                                                            min_frequency = 2)    
    
    assert corpus_lowered[1] == ['aa', 'bb', 'cc']
    assert corpus_weights_new[1] == [5, 5, 5]
    

def test_calculate_similarities():
    # Test with test-vectors and known outcome
    testvectors = np.array([[0,0,0,0,1], [1,0,0,0,1], [1,1,1,0,0], [0.5,0.5,0.5,0,0]])
    
    # Run function:
    list_similars_ids, list_similars, mean_similarity = calculate_similarities(testvectors, num_hits=4, method='cosine')
    
    assert list_similars[0][1] == list_similars[1][1] > 0.7     
    assert list_similars[2][1] == list_similars[3][1] == 1  
    assert list_similars[0][3] == list_similars[2][3] == 0
    assert np.min(list_similars_ids[1,:] == np.array([1, 0, 2, 3]))
    assert np.min(list_similars_ids[2,:] == list_similars_ids[3,:] == np.array([2, 3, 1, 0]))




  