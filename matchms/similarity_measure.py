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

import os
import numpy as np
import logging
from pprint import pprint
import gensim
from gensim import corpora
from gensim import models
from gensim.test.utils import get_tmpfile
from gensim.models.callbacks import CallbackAny2Vec

#from scipy import spatial
from sklearn.decomposition import PCA

# Imports from Spec2Vec functions
from . import helper_functions as functions


class EpochLogger(CallbackAny2Vec):
    '''Callback to log information about training progress.
    Used to keep track of gensim model training (word2vec, lda...)'''
    def __init__(self, num_of_epochs, iterations, filename):
        self.epoch = 0
        self.num_of_epochs = num_of_epochs
        self.iterations = iterations
        self.filename = filename
        self.loss = 0
        #self.loss_to_be_subed = 0
    def on_epoch_end(self, model):
        loss = model.get_latest_training_loss()
        #loss_now = loss - self.loss_to_be_subed
        print('\r' + ' Epoch ' + str(self.epoch+1) + ' of ' + str(self.num_of_epochs) + '.', end="")
        print('Change in loss after epoch {}: {}'.format(self.epoch+1, loss - self.loss))
        self.epoch += 1
        self.loss = loss

        # Save model during training if specified in iterations list
        if self.filename is not None:
            if self.epoch in [int(x + np.sum(self.iterations[:i])) for i, x in enumerate(self.iterations)]:
                #if self.epoch < self.num_of_epochs:
                filename = self.filename.split('.model')[0] + '_iter_' + str(self.epoch) + '.model'
                print('Saving model with name:', filename)
                model.save(filename)



## ------------------------------------------------------------------------------
## ---------------------- SimilarityMeasures class ------------------------------
## ------------------------------------------------------------------------------

class SimilarityMeasures():
    """ Class to run different similarity measure on sentence-like data.
    Words can be representing all kind of things (e.g. peaks for spectra).
    Documents lists of words.

    Similarity measuring methods:
    1) Low-dimensional document vector similarity (e.g. cosine similarity)
        a) Word2Vec based centroid vector (tfidf weighted or not weighted)
        b) PCA
    2) Topic modeling:
        a) LDA
        b) LSI
    """

    def __init__(self, initial_documents, initial_documents_weights = None):
        self.corpus = initial_documents
        self.corpus_weights = initial_documents_weights
        self.dictionary = []
        self.bow_corpus = []
        self.stopwords = []
        self.X_data = None

        # Trained models
        self.model_word2vec = None
        self.model_lda = None
        self.model_lsi = None
        self.index_lda = None
        self.index_lsi = None
        self.tfidf = None
        self.vectors_centroid = []
        self.vectors_pca = []

        # Listed similarities
        self.list_similars_ctr = None
        self.list_similars_ctr_idx = None
        self.list_similars_pca = None
        self.list_similars_pca_idx = None
        self.list_similars_lda = None
        self.list_similars_lda_idx = None
        self.list_similars_lsi = None
        self.list_similars_lsi_idx = None


    def preprocess_documents(self,
                             max_fraction,
                             min_frequency,
                             remove_stopwords = None,
                             create_stopwords = False):
        """ Preprocess 'documents'

        Obvious steps:
            --> in 'helper_functions.preprocess_document'
            - Take all words that occur at least (min_frequency =) 2 times.
            - Lower case

        Calculate word frequency
        --> Words that occur more than max_fraction will become stopwords (words with no or little discriminative power)

        Args:
        --------
        max_fraction: float
            Gives maximum fraction of documents that may contain a certain word.
        min_frequency: int
            Words that occur less frequently will be ignored.
        remove_stopwords: list, None
            Give list of stopwords if they should be removed. Default is None.
        create_stopwords: bool
            if True: Words that are more common then max_fraction will be added to stopwords.
        """

        if max_fraction <= 0 or max_fraction > 1:
            print("max_fraction should be value > 0 and <= 1.")

        # Preprocess documents (all lower letters, every word exists at least 2 times)
        print("Preprocess documents...")
        if remove_stopwords is None:
            self.corpus, self.corpus_weights = functions.preprocess_document(self.corpus,
                                                                             self.corpus_weights,
                                                                             stopwords = [],
                                                                             min_frequency = min_frequency)
        else:
            self.corpus, self.corpus_weights = functions.preprocess_document(self.corpus,
                                                                             self.corpus_weights,
                                                                             stopwords = remove_stopwords,
                                                                             min_frequency = min_frequency)

        # Create dictionary (or "vocabulary") containting all unique words from documents
        self.dictionary = corpora.Dictionary(self.corpus)

        if create_stopwords:
            # Calculate word frequency to determine stopwords
            print("Calculate inverse document frequency for entire dictionary.")
            documents_size = len(self.corpus)
            self.idf_scores = functions.ifd_scores(self.dictionary, self.corpus)

            # Words that appear too frequently (fraction>max_fration) become stopwords
            self.stopwords = self.idf_scores["word"][self.idf_scores["word count"] > documents_size*max_fraction]

            print(len(self.stopwords), " stopwords were selected from a total of ",
                  len(self.dictionary), " words in the entire corpus.")

            # Create corpus, dictionary, and BOW corpus
            self.corpus, self.corpus_weights = functions.preprocess_document(self.corpus, self.stopwords, min_frequency = min_frequency)

        self.bow_corpus = [self.dictionary.doc2bow(text) for text in self.corpus]


    ## ------------------------------------------------------------------------------
    ## ---------------------- Model building & training  ----------------------------
    ## ------------------------------------------------------------------------------

    def build_model_word2vec(self,
                             file_model_word2vec,
                             sg=0,
                             negative = 5,
                             size=100,
                             window=50,
                             min_count=1,
                             workers=4,
                             iterations=100,
                             use_stored_model=True,
                             learning_rate_initial = 0.025,
                             learning_rate_decay = 0.00025):
        """ Build Word2Vec model (using gensim)

        Args:
        --------
        file_model_word2vec: str,
            Filename to save model (or load model if it exists under this name).
        sg: int (0,1)
            For sg = 0 --> CBOW model, for sg = 1 --> skip gram model (see Gensim documentation).
        negative: int
            from Gensim:  If > 0, negative sampling will be used, the int for negative specifies how many “noise words”
            should be drawn (usually between 5-20). If set to 0, no negative sampling is used.
        size: int,
            Dimensions of word vectors (default = 100)
        window: int,
            Window size for context words (small for local context,
            larger for global context, default = 50)
        min_count: int,
            Only consider words that occur at least min_count times in the corpus (default =1).
        workers: int,
            Number of threads to run the training on (should not be more than number of cores/threads, default = 4).
        iterations: int, list
            Number of training iterations (default=100). If given as list training will loop through
            all given iterations [iter0, iter1, ...] in the list and save the model after each completed
            cycle. Temporary models will be saved using the name: file_model_word2ve + '_TEMP_#epoch.model'
        use_stored_model: bool,
            Load stored model if True, else train new model.
        """

        # Check if model already exists and should be loaded
        if file_model_word2vec is None or not use_stored_model:
            train_new_model = True
        else:
            train_new_model = False

        if not train_new_model:
            if os.path.isfile(file_model_word2vec):
                print("Load stored word2vec model ...")
                self.model_word2vec = gensim.models.Word2Vec.load(file_model_word2vec)
            else:
                print("No saved word2vec model found with given filename!")
        else:
            print("Calculating new word2vec model...")

            # Set up GENSIM logging
            logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.WARNING)

            if not isinstance(iterations, list):
                iterations = [iterations]

            epoch_logger = EpochLogger(np.sum(iterations), iterations, file_model_word2vec)
            iter_sum = np.sum(iterations)
            min_alpha = learning_rate_initial - iter_sum * learning_rate_decay
            if min_alpha < 0:
                print("Warning! Number of iterations is too high for specified learning_rate decay.")
                print("Learning_rate_decay will be set from", learning_rate_decay, "to", learning_rate_initial/iter)
                min_alpha = 0
            self.model_word2vec = gensim.models.Word2Vec(self.corpus,
                                                         sg=sg,
                                                         negative = negative,
                                                         size=size,
                                                         window=window,
                                                         min_count=min_count,
                                                         workers=workers,
                                                         iter=iter_sum,
                                                         alpha = learning_rate_initial,
                                                         min_alpha = min_alpha,
                                                         seed=42,
                                                         compute_loss=True,
                                                         callbacks=[epoch_logger])



    def build_model_lda(self,
                        file_model_lda,
                        num_of_topics=100,
                        num_pass=4,
                        num_iter=100,
                        use_stored_model=True):
        """ Build LDA model (using gensim).

        Args:
        --------
        file_model_lda: str,
            Filename to save model (or load model if it exists under this name).
        num_of_topics: int,
            Number of topics to sort feature into (default = 100).
        num_pass: int,
            Number of passes through the corpus during training.
       num_iter: int,
            Number of training iterations (default=100).
        use_stored_model: bool,
            Load stored model if True, else train new model.
        """

        # Check if model already exists and should be loaded
        if os.path.isfile(file_model_lda) and use_stored_model:
            print("Load stored LDA model ...")
            self.model_lda = gensim.models.LdaModel.load(file_model_lda)
        else:
            if use_stored_model:
                print("Stored LDA model not found!")
            print("Calculating new LDA model...")
            self.model_lda = gensim.models.LdaModel(self.bow_corpus, id2word=self.dictionary,
                                               num_topics=num_of_topics, passes=num_pass, iterations=num_iter)

            # Save model
            self.model_lda.save(file_model_lda)

            # Output the Keyword in the 10 topics
            pprint("Keyword in the 10 topics")
            pprint(self.model_lda.print_topics())


    def build_model_lsi(self,
                        file_model_lsi,
                        num_of_topics=100,
                        num_iter=10,
                        use_stored_model=True):
        """ Build LSI model (using gensim).

        Args:
        --------
        file_model_lsi: str,
            Filename to save model (or load model if it exists under this name).
        num_of_topics: int,
            Number of topics to sort feature into (default = 100).
        num_iter: int,
            Number of training iterations (default=100).
        use_stored_model: bool,
            Load stored model if True, else train new model.
        """

        # Check if model already exists and should be loaded
        if os.path.isfile(file_model_lsi) and use_stored_model:
            print("Load stored LSI model ...")
            self.model_lsi = gensim.models.LsiModel.load(file_model_lsi)
        else:
            if use_stored_model:
                print("Stored LSI model not found!")
            print("Calculating new LSI model...")
            self.model_lsi = gensim.models.LsiModel(self.bow_corpus,
                                                    id2word=self.dictionary,
                                                    power_iters=num_iter,
                                                    num_topics=num_of_topics)

            # Save model
            self.model_lsi.save(file_model_lsi)


    ## ------------------------------------------------------------------------------
    ## -------------------- Calculate document vectors ------------------------------
    ## ------------------------------------------------------------------------------

    def get_vectors_centroid(self, method = 'update',
                             tfidf_weighted=True,
                             weighting_power = 0.5,
                             tfidf_model = None,
                             extra_epochs = 10):
        """ Calculate centroid vectors for all documents of the library.

        Individual word vectors are weighted using tfidf (unless weighted=False).

        Args:
        --------
        method: str
            Which method to use if not all words are present in trained model.
            'update': word2vec model will be updated by additional training of the model.
            'ignore': will ignore all 'words' not present in the pre-trained model.
            TODO 'substitute": will look to replace missing words with closest matches?
        tfidf_weighted: bool
            True, False
        weighting_power: float
            If weights are present (self.corpus_weights), than those weights will be
            used to the power of 'weighting_power'.
            Set to 0 to ignore.
        tfidf_model: str
            Give filename if pre-defined tfidf model should be used. Otherwise set to None.
        extra_epochs: int
            Number of extra epochs to train IF method is 'update' and missing words are detected.
        """
        # TODO  maybe move the update section to the build_model function?

        # Check if everything is there:
        # 1) Check if model and bow-corpus are present
        if self.model_word2vec is None:
            print("Word2vec model first needs to be load or made (self.build_model_word2vec).")
        if len(self.bow_corpus) == 0:
            print("BOW corpus has not been calculated yet (bow_corpus).")

        # 2) Check if all words are included in trained word2vec model
        dictionary = [self.dictionary[x] for x in self.dictionary]
        test_vocab = []
        for i, word in enumerate(dictionary):
            if word not in self.model_word2vec.wv.vocab:
                test_vocab.append((i, word))

        if len(test_vocab) > 0:
            print("Not all 'words' of the given documents are present in the trained word2vec model!")
            print(len(test_vocab), " out of ", len(self.dictionary), " 'words' were not found in the word2vec model.")
            if method == 'update':
                print("The word2vec model will hence be updated by additional training.")
                self.model_word2vec.build_vocab(self.corpus, update=True)
                self.model_word2vec.train(self.corpus, total_examples=len(self.corpus), epochs = extra_epochs)
                self.model_word2vec.save('newmodel')

            elif method == 'ignore':
                print("'Words'missing in the pretrained word2vec model will be ignored.")

                _, missing_vocab = zip(*test_vocab)
                print("Removing missing 'words' from corpus...")
                # Update corpus and BOW-corpus
                self.corpus = [[word for word in document if word not in missing_vocab] for document in self.corpus]
                self.bow_corpus = [self.dictionary.doc2bow(text) for text in self.corpus]
                # TODO: add check with word intensities
            else:
                print("Given method how do deal with missing words could not be found.")
        else:
            print("All 'words' of the given documents were found in the trained word2vec model.")

        if tfidf_weighted is True:
            if tfidf_model is not None:
                self.tfidf = models.TfidfModel.load(tfidf_model)
                print("Tfidf model found and loaded.")
            else:
                if self.tfidf is None:
                    self.tfidf = models.TfidfModel(self.bow_corpus)
                    print("No tfidf model found.")
                else:
                    print("Using present tfidf model.")


        vector_size = self.model_word2vec.wv.vector_size
        vectors_centroid = []

        for i in range(len(self.bow_corpus)):
            if (i+1) % 10 == 0 or i == len(self.bow_corpus)-1:  # show progress
                print('\r', ' Calculated centroid vectors for ', i+1, ' of ', len(self.bow_corpus), ' documents.', end="")

            document = [self.dictionary[x[0]] for x in self.bow_corpus[i]]
            if self.corpus_weights is not None:
                # TODO: maybe next line can be skipped?
                document_weight = [self.corpus_weights[i][self.corpus[i].index(self.dictionary[x[0]])] for x in self.bow_corpus[i]]
                if len(document_weight) > 0:
                    document_weight = np.array(document_weight)**weighting_power/np.max(document_weight)  # normalize
            else:
                document_weight = np.ones((len(document)))
            if len(document) > 0:
                term1 = self.model_word2vec.wv[document]
                if tfidf_weighted:
                    term2 = np.array(list(zip(*self.tfidf[self.bow_corpus[i]]))[1])
                else:
                    term2 = np.ones((len(document)))

                term1 = term1 * np.tile(document_weight, (vector_size,1)).T
                weighted_docvector = np.sum((term1.T * term2).T, axis=0)
            else:
                weighted_docvector = np.zeros((self.model_word2vec.vector_size))
            vectors_centroid.append(weighted_docvector)

        self.vectors_centroid = np.array(vectors_centroid)


    def get_vectors_pca(self, dimension=100):
        """ Calculate PCA vectors for all documents.

        Args:
        -------
        dimension: int
            Dimension of reduced PCA vectors. Default is 100.
        """
        pca = PCA(n_components=dimension)

        input_dim = len(self.dictionary)
        corpus_dim = len(self.corpus)

        # See if there is one-hot encoded vectors (X_data)
        if self.X_data is None:
            # Transform data to be used as input for Keras model
            self.X_data = np.zeros((corpus_dim, input_dim))

            for i, bow_doc in enumerate(self.bow_corpus[:corpus_dim]):
                word_vector_bow = np.array([x[0] for x in bow_doc]).astype(int)
                word_vector_count = np.array([x[1] for x in bow_doc]).astype(int)
                self.X_data[i,:] = functions.full_wv(input_dim, word_vector_bow, word_vector_count)

        self.vectors_pca = pca.fit_transform(self.X_data)


    ## ------------------------------------------------------------------------------
    ## -------------------- Calculate similarities ----------------------------------
    ## ------------------------------------------------------------------------------

    def get_centroid_similarity(self, num_hits=25, method='cosine'):
        """ Calculate centroid similarities(all-versus-all --> matrix)

        Args:
        -------
        num_centroid_hits: int
            Function will store the num_centroid_hits closest matches. Default is 25.
        method: str
            See scipy spatial.distance.cdist for options. Default is 'cosine'.

        """
        list_similars_idx, list_similars, mean_similarity = functions.calculate_similarities(self.vectors_centroid,
                                                                   num_hits, method = method)
        print("Calculated distances between ", list_similars.shape[0], " documents.")
        self.list_similars_ctr_idx = list_similars_idx
        self.list_similars_ctr = list_similars


    def get_pca_similarity(self, num_hits=25, method='cosine'):
        """ Calculate PCA similarities(all-versus-all --> matrix)

        Args:
        -------
        num_centroid_hits: int
            Function will store the num_centroid_hits closest matches. Default is 25.
        method: str
            See scipy spatial.distance.cdist for options. Default is 'cosine'.

        """
        list_similars_idx, list_similars, mean_similarity = functions.calculate_similarities(self.vectors_pca,
                                                                   num_hits, method = method)

        self.list_similars_pca_idx = list_similars_idx
        self.list_similars_pca = list_similars


    def get_lda_similarity(self, num_hits=25):
        """ Calculate LDA topic based similarities (all-versus-all)

        Args:
        -------
        num_centroid_hits: int
            Function will store the num_centroid_hits closest matches. Default is 25.

        """

        # Now using faster gensim way (also not requiering to load everything into memory at once)
        index_tmpfile = get_tmpfile("index")
        index = gensim.similarities.Similarity(index_tmpfile, self.model_lda[self.bow_corpus],
                                               num_features=len(self.dictionary))  # build the index
        Cdist = np.zeros((len(self.corpus), len(self.corpus)))
        for i, similarities in enumerate(index):  # yield similarities of all indexed documents
            Cdist[:,i] = similarities

#        Cdist = 1 - Cdist  # switch from similarity to distance

        # Create numpy arrays to store similarities
        list_similars_idx = np.zeros((Cdist.shape[0],num_hits), dtype=int)
        list_similars = np.zeros((Cdist.shape[0],num_hits))

        for i in range(Cdist.shape[0]):
            list_similars_idx[i,:] = Cdist[i,:].argsort()[-num_hits:][::-1]
            list_similars[i,:] = Cdist[i, list_similars_idx[i,:]]

        self.list_similars_lda_idx = list_similars_idx
        self.list_similars_lda = list_similars


    def get_lsi_similarity(self, num_hits=25):
        """ Calculate LSI based similarities (all-versus-all)

        Args:
        -------
        num_centroid_hits: int
            Function will store the num_centroid_hits closest matches. Default is 25.

        """

        # Now using faster gensim way (also not requiering to load everything into memory at once)
        index_tmpfile = get_tmpfile("index")
        index = gensim.similarities.Similarity(index_tmpfile, self.model_lsi[self.bow_corpus],
                                               num_features=len(self.dictionary))  # build the index
        Cdist = np.zeros((len(self.corpus), len(self.corpus)))
        for i, similarities in enumerate(index):  # yield similarities of all indexed documents
            Cdist[:,i] = similarities

#        Cdist = 1 - Cdist  # switch from similarity to distance

        # Create numpy arrays to store distances
        list_similars_idx = np.zeros((Cdist.shape[0],num_hits), dtype=int)
        list_similars = np.zeros((Cdist.shape[0],num_hits))

        for i in range(Cdist.shape[0]):
            list_similars_idx[i,:] = Cdist[i,:].argsort()[-num_hits:][::-1]
            list_similars[i,:] = Cdist[i, list_similars_idx[i,:]]

        self.list_similars_lsi_idx = list_similars_idx
        self.list_similars_lsi = list_similars


    def save(self, filename):
        """ Save entire SimilarityMeasures() object to file.
        Uses pickle. Not ideal, but fine for now.

        Args:
        -------
        filename: str
            Filename to save object to.
        """
        import pickle
        f = open(filename, 'wb')
        pickle.dump(self.__dict__, f)
        f.close()


    def load(self, filename):
        """ Load SimilarityMeasures() object from file.
        Uses pickle. Not ideal, but fine for now.

        Args:
        -------
        filename: str
            Filename to load object from.
        """
        import pickle
        f = open(filename, 'rb')
        tmp_dict = pickle.load(f)
        self.__dict__.update(tmp_dict)
        f.close()
