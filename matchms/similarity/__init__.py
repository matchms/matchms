"""
Functions for computing spectra similarities
############################################

Matchms provides a number of frequently used similarity scores to compare mass
spectra. This includes

* scores based on comparing peak positions and intensities
  (:class:`~matchms.similarity.CosineGreedy`,
  :class:`~matchms.similarity.ModifiedCosineGreedy`,
  :class:`~matchms.similarity.ModifiedCosineHungarian`)
* simple scores that only assess precursor m/z or parent mass matches
  (:class:`~matchms.similarity.PrecursorMzMatch` or: :class:`~matchms.similarity.ParentMassMatch`)
* scores assessing molecular similarity if structures (SMILES, InchiKey) are given as metadata
  (:class:`~matchms.similarity.FingerprintSimilarity`)
* score for assessing matches in user-defined metadata fields which can be used to find equal
  entries (e.g. instrument_type) or numerical values within a specified tolerance
  (for instance: retention_time, collision energy...) (:class:`~matchms.similarity.MetadataMatch`)

It is also easily possible to add own custom similarity measures or import external ones
(such as `Spec2Vec <https://github.com/iomega/spec2vec>`_).
"""

from .BinnedEmbeddingSimilarity import BinnedEmbeddingSimilarity
from .BlinkCosine import BlinkCosine
from .CosineLinear import CosineLinear, LinearCosine
from .CosineGreedy import CosineGreedy
from .CosineHungarian import CosineHungarian
from .FingerprintSimilarity import FingerprintSimilarity
from .FlashSimilarity import FlashSimilarity
from .IntersectMz import IntersectMz
from .MetadataMatch import MetadataMatch
from .ModifiedCosineGreedy import ModifiedCosineGreedy
from .ModifiedCosineHungarian import ModifiedCosineHungarian
from .NeutralLossesCosine import NeutralLossesCosine
from .ParentMassMatch import ParentMassMatch
from .PrecursorMzMatch import PrecursorMzMatch


__all__ = [
    "BinnedEmbeddingSimilarity",
    "BlinkCosine",
    "CosineLinear",
    "ModifiedCosineGreedy",
    "CosineGreedy",
    "CosineHungarian",
    "FingerprintSimilarity",
    "FlashSimilarity",
    "IntersectMz",
    "MetadataMatch",
    "ModifiedCosineHungarian",
    "NeutralLossesCosine",
    "ParentMassMatch",
    "PrecursorMzMatch",
]


def get_similarity_function_by_name(similarity_function_name: str):
    """
    Get a similarity function by the name of its class.

    Parameters
    ----------
    similarity_function_name : str
        Name of the similarity function.
    """
    mapper = {
        "BinnedEmbeddingSimilarity": BinnedEmbeddingSimilarity,
        "BlinkCosine": BlinkCosine,
        "CosineLinear": CosineLinear,
        "CosineGreedy": CosineGreedy,
        "CosineHungarian": CosineHungarian,
        "FingerprintSimilarity": FingerprintSimilarity,
        "FlashSimilarity": FlashSimilarity,
        "IntersectMz": IntersectMz,
        "LinearCosine": CosineLinear,
        "MetadataMatch": MetadataMatch,
        "ModifiedCosineGreedy": ModifiedCosineGreedy,
        "ModifiedCosineHungarian": ModifiedCosineHungarian,
        "NeutralLossesCosine": NeutralLossesCosine,
        "ParentMassMatch": ParentMassMatch,
        "PrecursorMzMatch": PrecursorMzMatch,
    }

    assert similarity_function_name in mapper, f"Unknown similarity function: {similarity_function_name}"
    return mapper[similarity_function_name]
