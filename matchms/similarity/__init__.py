"""
Functions for computing spectra similarities
############################################

Matchms provides a number of frequently used similarity scores to compare mass
spectra. This includes

* scores based on comparing peak positions and intensities
  (:class:`~matchms.similarity.CosineGreedy` or :class:`~matchms.similarity.ModifiedCosine`)
* simple scores that only assess precursor m/z or parent mass matches
  (:class:`~matchms.similarity.PrecursormzMatch` or: :class:`~matchms.similarity.ParentmassMatch`)
* scores assessing molecular similarity if structures (SMILES, InchiKey) are given as metadata
  (:class:`~matchms.similarity.FingerprintSimilarity`)

It is also easily possible to add own custom similarity measures or import external ones
(such as `Spec2Vec <https://github.com/iomega/spec2vec>`_).
"""
from .CosineGreedy import CosineGreedy
from .CosineHungarian import CosineHungarian
from .FingerprintSimilarity import FingerprintSimilarity
from .IntersectMz import IntersectMz
from .ModifiedCosine import ModifiedCosine
from .ParentmassMatch import ParentmassMatch
from .PrecursormzMatch import PrecursormzMatch


__all__ = [
    "CosineGreedy",
    "CosineHungarian",
    "FingerprintSimilarity",
    "IntersectMz",
    "ModifiedCosine",
    "ParentmassMatch",
    "PrecursormzMatch",
]
