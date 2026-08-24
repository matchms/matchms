"""
Functions for computing spectra similarities
############################################

Matchms provides similarity measures for comparing mass spectra and their
metadata. The recommended high-level entry points for peak-based cosine scoring
are :class:`~matchms.similarity.Cosine` and
:class:`~matchms.similarity.ModifiedCosine`.

These classes choose an appropriate implementation internally and are intended
as the default choice for most workflows. Users who need a specific algorithmic
variant can select one of the explicit implementations directly, for example
:class:`~matchms.similarity.CosineLinear`,
:class:`~matchms.similarity.CosineFlash`,
:class:`~matchms.similarity.CosineGreedy`, or
:class:`~matchms.similarity.CosineHungarian`.

Available similarity functions include:

* cosine-based peak similarity
  (:class:`~matchms.similarity.Cosine`,
  :class:`~matchms.similarity.CosineLinear`,
  :class:`~matchms.similarity.CosineFlash`,
  :class:`~matchms.similarity.CosineGreedy`,
  :class:`~matchms.similarity.CosineHungarian`)
* modified cosine similarity for spectra with shifted fragment peaks
  (:class:`~matchms.similarity.ModifiedCosine`,
  :class:`~matchms.similarity.CosineFlash` with matching_mode="hybrid",
  :class:`~matchms.similarity.ModifiedCosineGreedy`,
  :class:`~matchms.similarity.ModifiedCosineHungarian`)
* neutral-loss-based peak similarity
  (:class:`~matchms.similarity.NeutralLossesCosine`)
* fast embedding-based or approximate similarity methods
  (:class:`~matchms.similarity.BinnedEmbeddingSimilarity`,
  :class:`~matchms.similarity.CosineBlink`,
  :class:`~matchms.similarity.FlashEntropy`)
* simple precursor or parent-mass matching
  (:class:`~matchms.similarity.PrecursorMzMatch`,
  :class:`~matchms.similarity.ParentMassMatch`)
* molecular-structure similarity based on metadata such as SMILES or InChIKey
  (:class:`~matchms.similarity.FingerprintSimilarity`)
* metadata-based matching for user-defined fields, for example exact matches in
  ``instrument_type`` or numerical matches within a tolerance for fields such as
  ``retention_time`` or ``collision_energy``
  (:class:`~matchms.similarity.MetadataMatch`)

Custom similarity measures can be added by subclassing
:class:`~matchms.similarity.BaseSimilarity`. Similarities that also provide
sparse score computation should subclass
:class:`~matchms.similarity.BaseSimilarityWithSparse`.

External similarity measures, such as
`Spec2Vec <https://github.com/iomega/spec2vec>`_, can also be used together with
matchms workflows.
"""

from .binned_embedding_similarity import BinnedEmbeddingSimilarity
from .cosine import Cosine
from .cosine_blink import CosineBlink
from .cosine_greedy import CosineGreedy
from .cosine_hungarian import CosineHungarian
from .cosine_linear import CosineLinear
from .fingerprint_similarity import FingerprintSimilarity
from .flash_similarity import CosineFlash, FlashEntropy
from .metadata_match import MetadataMatch
from .modified_cosine import ModifiedCosine
from .modified_cosine_greedy import ModifiedCosineGreedy
from .modified_cosine_hungarian import ModifiedCosineHungarian
from .neutral_losses_cosine import NeutralLossesCosine
from .parent_mass_match import ParentMassMatch
from .precursor_mz_match import PrecursorMzMatch


__all__ = [
    "BinnedEmbeddingSimilarity",
    "Cosine",
    "CosineBlink",
    "CosineFlash",
    "CosineGreedy",
    "CosineHungarian",
    "CosineLinear",
    "FingerprintSimilarity",
    "FlashEntropy",
    "MetadataMatch",
    "ModifiedCosine",
    "ModifiedCosineGreedy",
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
        "BinnedEmbeddingSimilarity": binned_embedding_similarity,
        "CosineBlink": CosineBlink,
        "Cosine": Cosine,
        "CosineLinear": CosineLinear,
        "CosineGreedy": CosineGreedy,
        "CosineHungarian": CosineHungarian,
        "FingerprintSimilarity": FingerprintSimilarity,
        "CosineFlash": CosineFlash,
        "FlashEntropy": FlashEntropy,
        "MetadataMatch": MetadataMatch,
        "ModifiedCosine": ModifiedCosine,
        "ModifiedCosineGreedy": ModifiedCosineGreedy,
        "ModifiedCosineHungarian": ModifiedCosineHungarian,
        "NeutralLossesCosine": NeutralLossesCosine,
        "ParentMassMatch": ParentMassMatch,
        "PrecursorMzMatch": PrecursorMzMatch,
    }

    assert similarity_function_name in mapper, f"Unknown similarity function: {similarity_function_name}"
    return mapper[similarity_function_name]
