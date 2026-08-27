"""
Functions for computing spectrum similarities
##############################################

Matchms provides similarity measures for comparing mass spectra, spectrum
metadata, and molecular structures.

For peak-based spectral similarity, the recommended high-level entry points are:

* :class:`~matchms.similarity.Cosine` for standard cosine similarity,
* :class:`~matchms.similarity.ModifiedCosine` when precursor-mass shifts should
  be considered, and
* :class:`~matchms.similarity.Entropy` for spectral entropy similarity.

These classes select suitable implementations internally and are intended to
be the default choice for most workflows using :meth:`pair` or :meth:`matrix`.

Specialized implementations
---------------------------

For applications that require explicit control over the scoring algorithm,
matchms also exposes the underlying implementations.

Cosine similarity
~~~~~~~~~~~~~~~~~

* :class:`~matchms.similarity.CosineGreedy`
* :class:`~matchms.similarity.CosineHungarian`
* :class:`~matchms.similarity.CosineLinear`
* :class:`~matchms.similarity.CosineFlash`
* :class:`~matchms.similarity.CosineBlink`

Modified cosine similarity
~~~~~~~~~~~~~~~~~~~~~~~~~~

* :class:`~matchms.similarity.ModifiedCosineGreedy`
* :class:`~matchms.similarity.ModifiedCosineHungarian`
* :class:`~matchms.similarity.CosineFlash` with ``matching_mode="hybrid"``

Spectral entropy similarity
~~~~~~~~~~~~~~~~~~~~~~~~~~~

* :class:`~matchms.similarity.EntropyGreedy`
* :class:`~matchms.similarity.FlashEntropy`

Other similarity measures
-------------------------

Additional similarity measures include:

* :class:`~matchms.similarity.NeutralLossesCosine` for neutral-loss-based
  spectral similarity,
* :class:`~matchms.similarity.BinnedEmbeddingSimilarity` for comparison of
  binned spectrum representations,
* :class:`~matchms.similarity.FingerprintSimilarity` for molecular fingerprint
  similarity,
* :class:`~matchms.similarity.MetadataMatch` for user-defined metadata
  comparisons,
* :class:`~matchms.similarity.PrecursorMzMatch` and
  :class:`~matchms.similarity.ParentMassMatch` for mass-based matching.

Custom similarities
-------------------

Custom similarity measures can be implemented by subclassing
:class:`~matchms.similarity.BaseSimilarity`. Similarities that also support
sparse score computation can subclass
:class:`~matchms.similarity.BaseSimilarityWithSparse`.

External similarity measures, such as
`Spec2Vec <https://github.com/iomega/spec2vec>`_, can also be integrated into
matchms workflows.
"""

from .binned_embedding_similarity import BinnedEmbeddingSimilarity
from .cosine import Cosine
from .cosine_blink import CosineBlink
from .cosine_greedy import CosineGreedy
from .cosine_hungarian import CosineHungarian
from .cosine_linear import CosineLinear
from .entropy import Entropy
from .entropy_greedy import EntropyGreedy
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
    "Entropy",
    "EntropyGreedy",
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
        "BinnedEmbeddingSimilarity": BinnedEmbeddingSimilarity,
        "CosineBlink": CosineBlink,
        "Cosine": Cosine,
        "CosineLinear": CosineLinear,
        "CosineGreedy": CosineGreedy,
        "CosineHungarian": CosineHungarian,
        "Entropy": Entropy,
        "EntropyGreedy": EntropyGreedy,
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
