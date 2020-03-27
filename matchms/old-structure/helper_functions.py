#
# Spec2Vec
#
# Copyright 2019 Netherlands eScience Center
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

from __future__ import print_function
import numpy as np
from scipy import spatial
import json
import math
import pandas as pd

## ----------------------------------------------------------------------------
## ---------------- Document processing functions -----------------------------
## ----------------------------------------------------------------------------

def preprocess_document(corpus, 
                        corpus_weights = None, 
                        stopwords = [], 
                        min_frequency = 2):
    """ Basic preprocessing of document words
    
    - Remove common words from stopwords and tokenize
    - Only include words that appear at least *min_frequency* times. 
    - Set words to lower case.
    
    Args:
    -------
    corpus: list
        Corpus of documents.
    corpus_weights: list, optional
        Weights for all words in all documents of corpus. Default = None
    stopwords: list, optional
        List of stopwords to exclude from documents. Default = []
    min_frequency: int
        Minimum total occurence of a word necessary to be included in processed corpus. Default = 2
    """
    corpus_lowered = [[word.lower() for word in document if word not in stopwords] for document in corpus]

    # Count word occurences
    from collections import defaultdict
    
    frequency = defaultdict(int)
    for document in corpus_lowered:
        for word in list(set(document)):
            frequency[word] += 1
    
    # Remove words that appear less than min_frequency times
    corpus_lowered_new = [[word for word in document if frequency[word] >= min_frequency] for document in corpus_lowered]
    if corpus_weights is not None:
        corpus_weights = [[weights[x] for x in range(len(weights)) if frequency[corpus_lowered[i][x]] >= min_frequency] for i, weights in enumerate(corpus_weights)]
    
    return corpus_lowered_new, corpus_weights



##
## ---------------- General functions ----------------------------------------
## 

def dict_to_json(mydict, file_json): 
    # save dictionary as json file
    with open(file_json, 'w') as outfile:  
        json.dump(mydict, outfile)

        
def json_to_dict(file_json): 
    # create dictionary from json file
    with open(file_json) as infile:  
        mydict = json.load(infile)

    return mydict


def full_wv(vocab_size, word_idx, word_count):
    """ Create full word vector
    """
    one_hot = np.zeros((vocab_size)) 
    one_hot[word_idx] = word_count
    return one_hot



##
## ---------------- Clustering & metrics functions ----------------------------
## 

def ifd_scores(vocabulary, corpus):
    """ Calulate idf score (Inverse Document Frequency score) for all words in vocabulary over a given corpus 
    
    Args:
    --------
    vocabulary: gensim.corpora.dictionary
        Dictionary of all corpus words
    corpus: list of lists
        List of all documents (document = list of words)
    
    Output: 
        idf_scores: pandas DataFrame 
            contains all words and their ids, their word-count, and idf score
    """ 
    #TODO: this function is still slow! (but only needs to be calculated once)
    
    idf_scores = []
    idf_score = []
    vocabulary_size = len(vocabulary)
    corpus_size = len(corpus)
    
    for i in range(0, vocabulary_size):
        if (i+1) % 100 == 0 or i == vocabulary_size-1:  # show progress
                print('\r', ' Calculated scores for ', i+1, ' of ', vocabulary_size, ' words.', end="")
        
        word_containing = 0
        word = vocabulary[i]
        for document in corpus:
            word_containing += 1 * (document.count(word) > 0)
            idf_score = math.log(corpus_size / (max(1, word_containing)))   
            
        idf_scores.append([i, word, word_containing, idf_score])
    print("")
    return pd.DataFrame(idf_scores, columns=["id", "word", "word count", "idf score"])


def calculate_similarities(vectors, 
                           num_hits=25, 
                           method='cosine'):
    """ Calculate similarities (all-versus-all --> matrix) based on array of all vectors
    
    Args:
    -------
    num_centroid_hits: int
        Function will store the num_centroid_hits closest matches. Default is 25.      
    method: str
        See scipy spatial.distance.cdist for options. Default is 'cosine'.
        
    TODO: Check how to go from distance to similarity for methods other than cosine!!
    """
    Cdist = spatial.distance.cdist(vectors, vectors, method)
    mean_similarity = 1 - np.mean(Cdist)
    
    # Create numpy arrays to store distances
    list_similars_ids = np.zeros((Cdist.shape[0],num_hits), dtype=int)
    list_similars = np.zeros((Cdist.shape[0],num_hits))
    
    for i in range(Cdist.shape[0]):
        list_similars_ids[i,:] = Cdist[i,:].argsort()[:num_hits]
        list_similars[i,:] = 1- Cdist[i, list_similars_ids[i,:]]
    
    return list_similars_ids, list_similars, mean_similarity