import matchms.similarity as mssim
from matchms.similarity.default_parameters import DEFAULT_MZ_TOLERANCE


def test_peak_similarity_default_tolerances_are_consistent():
    """Test that default tolerances are consistent across all similarity implementations.
    Adjust if any of the default tolerances are changed in the future.
    """
    similarities = [
        mssim.Cosine(),
        mssim.CosineGreedy(),
        mssim.CosineHungarian(),
        mssim.CosineLinear(),
        mssim.CosineFlash(),
        mssim.ModifiedCosine(),
        mssim.ModifiedCosineGreedy(),
        mssim.ModifiedCosineHungarian(),
        mssim.Entropy(),
        mssim.EntropyGreedy(),
        mssim.FlashEntropy(),
    ]

    assert all(
        similarity.tolerance == DEFAULT_MZ_TOLERANCE
        for similarity in similarities
    )