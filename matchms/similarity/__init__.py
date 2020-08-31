"""
Functions for computing spectra similarities
############################################

Matchms provides a number of frequently used similarity scores to compare mass
spectra. This includes

* scores based on comparing peak positions and intensities
  (:meth:`~matchms.similarity.CosineGreedy` or :meth:`~matchms.similarity.ModifiedCosine`)
* simple scores that only assess parent mass matches
  (:meth:`~matchms.similarity.ParentmassMatchParallel`)
* scores assessing molecular similarity if structures (SMILES, InchiKey) are given as metadata
  (:meth:`~matchms.similarity.FingerprintSimilarityParallel`)

It is also easily possible to add own custom similarity measures or import external ones
(such as `Spec2Vec <https://github.com/iomega/spec2vec>`_).
"""
from .CosineGreedy import CosineGreedy
from .CosineHungarian import CosineHungarian
from .FingerprintSimilarity import FingerprintSimilarity
from .IntersectMz import IntersectMz
from .ModifiedCosine import ModifiedCosine
from .ParentmassMatch import ParentmassMatch


__all__ = [
    "CosineGreedy",
    "CosineHungarian",
    "FingerprintSimilarity",
    "IntersectMz",
    "ModifiedCosine",
    "ParentmassMatch",
]
