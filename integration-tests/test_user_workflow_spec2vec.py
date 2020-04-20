import os
import gensim
from matchms.importing import load_from_mgf
from matchms.filtering import default_filters
from matchms.similarity.spec2vec import convert_spectrum_to_document, Spec2Vec
from matchms import calculate_scores


def test_user_workflow_spec2vec():

    module_root = os.path.join(os.path.dirname(__file__), '..')
    references_file = os.path.join(module_root, 'tests', 'pesticides.mgf')

    spectrums = [default_filters(s) for s in load_from_mgf(references_file)]
    documents = [convert_spectrum_to_document(s) for s in spectrums]

    # create and train model
    model = gensim.models.Word2Vec([d.words for d in documents], size=5, min_count=1)
    model.train([d.words for d in documents], total_examples=len(documents), epochs=20)

    # define similarity_function
    spec2vec = Spec2Vec(model=model, documents=documents)

    queries = documents[:7]
    references = documents[6:]

    # calculate scores on all combinations of references and queries
    queries_top3, references_top3, scores_top3 = \
        calculate_scores(queries, references, spec2vec).top(3, include_self_comparisons=False)

    assert scores_top3[0][0] > 0.99, "Expected some really good scores."


if __name__ == '__main__':
    test_user_workflow_spec2vec()
