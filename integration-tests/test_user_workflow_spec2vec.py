import os
import gensim
from matchms.importing import load_from_mgf
from matchms.filtering import default_filters, require_minimum_number_of_peaks, add_parent_mass, normalize_intensities
from matchms.filtering import select_by_relative_intensity, select_by_mz, add_losses
from matchms.similarity.spec2vec import SpectrumDocument, Spec2Vec
from matchms import calculate_scores


def test_user_workflow_spec2vec():

    def apply_my_filters(s):
        s = default_filters(s)
        s = add_parent_mass(s)
        s = add_losses(s)
        s = normalize_intensities(s)
        s = select_by_relative_intensity(s, intensity_from=0.01, intensity_to=1.0)
        s = select_by_mz(s, mz_from=0, mz_to=1000)
        s = require_minimum_number_of_peaks(s, n_required=5)
        return s

    module_root = os.path.join(os.path.dirname(__file__), '..')
    spectrums_file = os.path.join(module_root, 'tests', 'pesticides.mgf')

    # apply my filters to the data
    spectrums = [apply_my_filters(s) for s in load_from_mgf(spectrums_file)]

    # omit spectrums that didn't qualify for analysis
    spectrums = [s for s in spectrums if s is not None]

    documents = [SpectrumDocument(s) for s in spectrums]

    # create and train model
    model = gensim.models.Word2Vec([d.words for d in documents], size=5, min_count=1)
    model.train([d.words for d in documents], total_examples=len(documents), epochs=20)

    # define similarity_function
    spec2vec = Spec2Vec(model=model, documents=documents, intensity_weighting_power=0.5)

    references = documents[:26]
    queries = documents[25:]

    # calculate scores on all combinations of references and queries
    scores = list(calculate_scores(references, queries, spec2vec))

    # filter out self-comparisons
    filtered = [(reference, query, score) for (reference, query, score) in scores if reference != query]

    sorted_by_score = sorted(filtered, key=lambda elem: elem[2], reverse=True)

    actual_top10 = sorted_by_score[:10]

    actual_scores = [score for (reference, query, score) in actual_top10]

    assert max(actual_scores) > 0.99
