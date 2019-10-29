# test functions

import numpy as np
from helper_functions import calculate_similarities

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




  