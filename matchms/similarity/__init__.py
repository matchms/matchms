"""Similarity measures for spectra, metadata, and molecular structures.

Choosing a peak-based spectral similarity
-----------------------------------------
For most workflows, start with one of the public high-level classes:

* :class:`~matchms.similarity.Cosine` for standard cosine similarity,
* :class:`~matchms.similarity.ModifiedCosine` when precursor-mass shifts should
  contribute to matching,
* :class:`~matchms.similarity.Entropy` for general spectral entropy similarity,
  and
* :class:`~matchms.similarity.EntropySearch` for high-throughput fragment-only
  entropy searches against large reference libraries.

``Cosine``, ``ModifiedCosine``, ``Entropy``, and ``EntropySearch`` all support
``pair`` and ``matrix`` workflows. Their indexed implementations also provide
``build_index`` and ``search`` for repeated queries against a fixed library.

Choosing between Entropy and EntropySearch
------------------------------------------
Both classes use the spectral entropy score, but they make different assumptions
about peak matching.

:class:`~matchms.similarity.Entropy` is the general-purpose choice. It explicitly
handles competing one-to-one peak matches and supports fragment, neutral-loss,
and hybrid matching with Da or ppm tolerances. Use it when preserving the general
matching semantics is more important than maximum library-search throughput, or
when spectra may contain peaks whose tolerance windows overlap.

:class:`~matchms.similarity.EntropySearch` is optimized for repeated fragment-only
library searches with absolute Da tolerances. It requires peaks within each
spectrum to be sufficiently separated that candidate matches cannot compete. The
class can either merge close peaks during preparation or raise an error when this
requirement is violated. Merging changes the prepared spectrum and can therefore
change scores relative to ``Entropy``. Use ``EntropySearch`` when this separation
assumption is acceptable and search throughput is the primary concern.

Specialized implementations
---------------------------
For applications that require explicit control over the scoring algorithm,
matchms also exposes lower-level implementations.

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
* :class:`~matchms.similarity.ModifiedCosineLinear`
* :class:`~matchms.similarity.CosineFlash` with ``matching_mode="hybrid"``

Spectral entropy similarity
~~~~~~~~~~~~~~~~~~~~~~~~~~~

* :class:`~matchms.similarity.EntropyGreedy` for a direct pair-oriented
  implementation,
* :class:`~matchms.similarity.FlashEntropy` for the general indexed entropy
  backend used by :class:`~matchms.similarity.Entropy`.

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
`Spec2Vec <https://github.com/iomega/spec2vec>`_ or
`MS2DeepScore <https://github.com/matchms/ms2deepscore>`_, can also be integrated into
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
from .entropy_search import EntropySearch
from .fingerprint_similarity import FingerprintSimilarity
from .flash_similarity import CosineFlash, FlashEntropy
from .metadata_match import MetadataMatch
from .modified_cosine import ModifiedCosine
from .modified_cosine_greedy import ModifiedCosineGreedy
from .modified_cosine_hungarian import ModifiedCosineHungarian
from .modified_cosine_linear import ModifiedCosineLinear
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
    "EntropySearch",
    "FingerprintSimilarity",
    "FlashEntropy",
    "MetadataMatch",
    "ModifiedCosine",
    "ModifiedCosineGreedy",
    "ModifiedCosineHungarian",
    "ModifiedCosineLinear",
    "NeutralLossesCosine",
    "ParentMassMatch",
    "PrecursorMzMatch",
]


def get_similarity_function_by_name(similarity_function_name: str):
    """Return a similarity class by its public class name.

    Parameters
    ----------
    similarity_function_name
        Name of the similarity class.

    Returns
    -------
    type
        Matching similarity class.

    Raises
    ------
    ValueError
        If ``similarity_function_name`` is not a known public similarity class.
    """
    mapper = {
        "BinnedEmbeddingSimilarity": BinnedEmbeddingSimilarity,
        "Cosine": Cosine,
        "CosineBlink": CosineBlink,
        "CosineFlash": CosineFlash,
        "CosineGreedy": CosineGreedy,
        "CosineHungarian": CosineHungarian,
        "CosineLinear": CosineLinear,
        "Entropy": Entropy,
        "EntropyGreedy": EntropyGreedy,
        "EntropySearch": EntropySearch,
        "FingerprintSimilarity": FingerprintSimilarity,
        "FlashEntropy": FlashEntropy,
        "MetadataMatch": MetadataMatch,
        "ModifiedCosine": ModifiedCosine,
        "ModifiedCosineGreedy": ModifiedCosineGreedy,
        "ModifiedCosineHungarian": ModifiedCosineHungarian,
        "ModifiedCosineLinear": ModifiedCosineLinear,
        "NeutralLossesCosine": NeutralLossesCosine,
        "ParentMassMatch": ParentMassMatch,
        "PrecursorMzMatch": PrecursorMzMatch,
    }

    try:
        return mapper[similarity_function_name]
    except KeyError as exc:
        raise ValueError(
            f"Unknown similarity function: {similarity_function_name}"
        ) from exc
