"""
Functions for computing spectra similarities
############################################

Matchms provides a number of frequently used similarity scores to compare mass
spectra. This includes

* scores based on comparing peak positions and intensities
  (:class:`~matchms.similarity.CosineGreedy` or :class:`~matchms.similarity.ModifiedCosine`)
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
from .CosineGreedy import CosineGreedy
from .CosineHungarian import CosineHungarian
from .FingerprintSimilarity import FingerprintSimilarity
from .IntersectMz import IntersectMz
from .MetadataMatch import MetadataMatch
from .ModifiedCosine import ModifiedCosine
from .NeutralLossesCosine import NeutralLossesCosine
from .ParentMassMatch import ParentMassMatch
from .PrecursorMzMatch import PrecursorMzMatch


__all__ = [
    "CosineGreedy",
    "CosineHungarian",
    "FingerprintSimilarity",
    "IntersectMz",
    "MetadataMatch",
    "ModifiedCosine",
    "NeutralLossesCosine",
    "ParentMassMatch",
    "PrecursorMzMatch",
]
