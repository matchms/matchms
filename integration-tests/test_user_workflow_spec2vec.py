import os
import gensim
from matchms.importing import load_from_mgf
from matchms.filtering import default_filters
from matchms.similarity.spec2vec import convert_spectrum_to_document, Spec2Vec
from matchms import calculate_scores


def test_user_workflow_spec2vec():

    def evaluate_assert_set_1():
        r, q, _ = scores.__next__()
        scores.reset_iterator()
        assert r == references[0]
        assert q == queries[0]

    def evaluate_assert_set_2():
        filtered = [triplet for triplet in scores if triplet[2] > 0.9999]
        assert len(filtered) > 1, "Expected some really good scores."

        sorted_by_score = sorted(scores, key=lambda elem: elem[2], reverse=True)
        rd = sorted_by_score[0][0]
        qd = sorted_by_score[0][1]

        assert rd == references[-1], "The best match should be between two copies of documents[69]"
        assert qd == queries[0], "The best match should be between two copies of documents[69]"

    module_root = os.path.join(os.path.dirname(__file__), '..')
    references_file = os.path.join(module_root, 'tests', 'pesticides.mgf')

    spectrums = [default_filters(s) for s in load_from_mgf(references_file)]
    documents = [convert_spectrum_to_document(s) for s in spectrums]

    # create and train model
    model = gensim.models.Word2Vec([d.words for d in documents], size=5, min_count=1)
    model.train([d.words for d in documents], total_examples=len(documents), epochs=20)

    # define similarity_function
    spec2vec = Spec2Vec(model=model, documents=documents)

    references = documents[:70]
    queries = documents[69:]

    # calculate scores on all combinations of references and queries
    scores = calculate_scores(references, queries, spec2vec)

    evaluate_assert_set_1()
    evaluate_assert_set_2()
