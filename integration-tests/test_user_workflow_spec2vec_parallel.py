import os
import gensim
import pytest
from matchms import calculate_scores_parallel
from matchms.filtering import add_losses
from matchms.filtering import add_parent_mass
from matchms.filtering import default_filters
from matchms.filtering import normalize_intensities
from matchms.filtering import require_minimum_number_of_peaks
from matchms.filtering import select_by_mz
from matchms.filtering import select_by_relative_intensity
from matchms.importing import load_from_mgf
from matchms.similarity.spec2vec import Spec2VecParallel
from matchms.similarity.spec2vec import SpectrumDocument


def test_user_workflow_spec2vec_parallel():

    def apply_my_filters(s):
        s = default_filters(s)
        s = add_parent_mass(s)
        s = add_losses(s)
        s = normalize_intensities(s)
        s = select_by_relative_intensity(s, intensity_from=0.01, intensity_to=1.0)
        s = select_by_mz(s, mz_from=0, mz_to=1000)
        s = require_minimum_number_of_peaks(s, n_required=5)
        return s

    repository_root = os.path.join(os.path.dirname(__file__), "..")
    spectrums_file = os.path.join(repository_root, "tests", "pesticides.mgf")

    # apply my filters to the data
    spectrums = [apply_my_filters(s) for s in load_from_mgf(spectrums_file)]

    # omit spectrums that didn't qualify for analysis
    spectrums = [s for s in spectrums if s is not None]

    documents = [SpectrumDocument(s) for s in spectrums]

    model_file = os.path.join(repository_root, "integration-tests", "test_user_workflow_spec2vec.model")
    if os.path.isfile(model_file):
        model = gensim.models.Word2Vec.load(model_file)
    else:
        # create and train model
        model = gensim.models.Word2Vec([d.words for d in documents], size=5, min_count=1)
        model.train([d.words for d in documents], total_examples=len(documents), epochs=20)
        model.save(model_file)

    # define similarity_function
    spec2vec = Spec2VecParallel(model=model, documents=documents, intensity_weighting_power=0.5)

    references = documents[:26]
    queries = documents[25:]

    # calculate scores on all combinations of references and queries
    scores = list(calculate_scores_parallel(references, queries, spec2vec))

    # filter out self-comparisons
    filtered = [(reference, query, score) for (reference, query, score) in scores if reference != query]

    sorted_by_score = sorted(filtered, key=lambda elem: elem[2], reverse=True)

    actual_top10 = sorted_by_score[:10]

    expected_top10 = [
        (documents[25], documents[61], pytest.approx(0.9961967430032844, rel=1e-9)),
        (documents[7], documents[25], pytest.approx(0.9928597742721408, rel=1e-9)),
        (documents[23], documents[38], pytest.approx(0.9924218396259692, rel=1e-9)),
        (documents[25], documents[56], pytest.approx(0.9920959275602975, rel=1e-9)),
        (documents[21], documents[38], pytest.approx(0.9917814395457049, rel=1e-9)),
        (documents[10], documents[38], pytest.approx(0.9915107610836775, rel=1e-9)),
        (documents[15], documents[53], pytest.approx(0.9914723281404583, rel=1e-9)),
        (documents[7], documents[61], pytest.approx(0.9911414671886221, rel=1e-9)),
        (documents[25], documents[49], pytest.approx(0.9909861225199654, rel=1e-9)),
        (documents[11], documents[62], pytest.approx(0.99092120438651, rel=1e-9))
    ]

    assert actual_top10 == expected_top10
