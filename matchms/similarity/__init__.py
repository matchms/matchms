"""Functions for computing spectra similarities.

Matchms provides a number of frequently used similarity scores to compare mass
spectra. This includes scores based on comparing peak positions and intensities
(:meth:`~matchms.similarity.CosineGreedy` or :meth:`~matchms.similarity.ModifiedCosine`),
simple scores that only assess parent mass matches
(:meth:`~matchms.similarity.ParentmassMatchParallel`), or similarity scores that
assess the underlying molecular similarity if structures are given as metadata
(:meth:`~matchms.similarity.FingerprintSimilarityParallel`). It is also easily
possible to add own custom similarity measures or import external ones (e.g
Spec2Vec similarity, see https://github.com/iomega/spec2vec).
"""
from .CosineGreedy import CosineGreedy
from .CosineHungarian import CosineHungarian
from .FingerprintSimilarityParallel import FingerprintSimilarityParallel
from .IntersectMz import IntersectMz
from .ModifiedCosine import ModifiedCosine
from .ParentmassMatch import ParentmassMatch
from .ParentmassMatchParallel import ParentmassMatchParallel


__all__ = [
    "CosineGreedy",
    "CosineHungarian",
    "FingerprintSimilarityParallel",
    "IntersectMz",
    "ModifiedCosine",
    "ParentmassMatch",
    "ParentmassMatchParallel",
]
