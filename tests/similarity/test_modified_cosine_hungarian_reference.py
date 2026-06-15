import pytest
from matchms.reference_spectra import aspirin, cocaine, glucose, hydroxy_cholesterol, phenylalanine, salicin
from matchms.similarity import ModifiedCosineHungarian


SCORE_ABS_TOL = 3e-8

REFERENCE_PAIRS = [
    {"left": "aspirin", "right": "aspirin", "score": 1.000000000000000, "matches": 29},
    {"left": "aspirin", "right": "cocaine", "score": 0.002140425769044, "matches": 2},
    {"left": "aspirin", "right": "glucose", "score": 0.191756421092755, "matches": 3},
    {"left": "aspirin", "right": "hydroxy_cholesterol", "score": 0.443841901000275, "matches": 16},
    {"left": "aspirin", "right": "phenylalanine", "score": 0.038780856719341, "matches": 4},
    {"left": "aspirin", "right": "salicin", "score": 0.001709472985324, "matches": 2},
    {"left": "cocaine", "right": "cocaine", "score": 1.000000000000000, "matches": 9},
    {"left": "cocaine", "right": "glucose", "score": 0.000000000000000, "matches": 0},
    {"left": "cocaine", "right": "hydroxy_cholesterol", "score": 0.017526287755408, "matches": 3},
    {"left": "cocaine", "right": "phenylalanine", "score": 0.000000000000000, "matches": 0},
    {"left": "cocaine", "right": "salicin", "score": 0.302562193618597, "matches": 1},
    {"left": "glucose", "right": "glucose", "score": 1.000000000000000, "matches": 14},
    {"left": "glucose", "right": "hydroxy_cholesterol", "score": 0.011152865111783, "matches": 8},
    {"left": "glucose", "right": "phenylalanine", "score": 0.000136790042176, "matches": 3},
    {"left": "glucose", "right": "salicin", "score": 0.000000000000000, "matches": 0},
    {"left": "hydroxy_cholesterol", "right": "hydroxy_cholesterol", "score": 1.000000000000000, "matches": 131},
    {"left": "hydroxy_cholesterol", "right": "phenylalanine", "score": 0.007063595459364, "matches": 7},
    {"left": "hydroxy_cholesterol", "right": "salicin", "score": 0.018647310888727, "matches": 4},
    {"left": "phenylalanine", "right": "phenylalanine", "score": 1.000000000000000, "matches": 23},
    {"left": "phenylalanine", "right": "salicin", "score": 0.000004640873395, "matches": 1},
    {"left": "salicin", "right": "salicin", "score": 1.000000000000000, "matches": 21},
]


def _reference_spectra():
    return {
        "aspirin": aspirin(),
        "cocaine": cocaine(),
        "glucose": glucose(),
        "hydroxy_cholesterol": hydroxy_cholesterol(),
        "phenylalanine": phenylalanine(),
        "salicin": salicin(),
    }


@pytest.mark.parametrize("pair", REFERENCE_PAIRS)
def test_modified_cosine_hungarian_reference_pairs(pair):
    similarity = ModifiedCosineHungarian(tolerance=0.1, mz_power=0.0, intensity_power=1.0)
    spectra = _reference_spectra()

    left = spectra[pair["left"]]
    right = spectra[pair["right"]]

    score = similarity.pair(left, right)

    assert float(score["score"]) == pytest.approx(pair["score"], abs=SCORE_ABS_TOL)
    assert int(score["matches"]) == pair["matches"]



def test_modified_cosine_hungarian_reference_matrix():
    similarity = ModifiedCosineHungarian(tolerance=0.1, mz_power=0.0, intensity_power=1.0)
    spectra = _reference_spectra()

    names = ["aspirin", "cocaine", "glucose", "hydroxy_cholesterol", "phenylalanine", "salicin"]
    ordered_spectra = [spectra[name] for name in names]

    matrix = similarity.matrix(ordered_spectra, ordered_spectra)

    for pair in REFERENCE_PAIRS:
        i = names.index(pair["left"])
        j = names.index(pair["right"])

        assert float(matrix[i, j]["score"]) == pytest.approx(pair["score"], abs=SCORE_ABS_TOL)
        assert int(matrix[i, j]["matches"]) == pair["matches"]

        assert float(matrix[j, i]["score"]) == pytest.approx(pair["score"], abs=SCORE_ABS_TOL)
        assert int(matrix[j, i]["matches"]) == pair["matches"]
