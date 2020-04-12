import os
from matchms.importing import load_from_mgf
from matchms.filtering import default_filters
from matchms.similarity.spec2vec import convert_spectrum_to_document, build_model_word2vec, Spec2Vec
from matchms import calculate_scores


def test_user_workflow_spec2vec():

    module_root = os.path.join(os.path.dirname(__file__), '..')
    references_file = os.path.join(module_root, 'tests', 'pesticides.mgf')

    spectrums = [default_filters(s) for s in load_from_mgf(references_file)]
    documents = [convert_spectrum_to_document(s) for s in spectrums]

    # train model
    model = build_model_word2vec(documents)

    # define similarity_function
    spec2vec = Spec2Vec(model=model)

    queries = documents[:7]
    references = documents[6:]

    # call calc_scores
    queries_top3, references_top3, scores_top3 = \
        calculate_scores(queries, references, spec2vec).top(3, omit_self_comparisons=True)

    print(queries_top3)
    print(references_top3)
    print(scores_top3)

    assert True


if __name__ == '__main__':
    test_user_workflow_spec2vec()
