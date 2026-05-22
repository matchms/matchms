from collections.abc import Callable
from typing import NewType, Union
import numpy as np


SpectrumType = NewType("Spectrum", object)
ScoresType = NewType("Scores", object)
SpectraCollectionType = NewType("SpectraCollection", object)
FragmentCollectionType = NewType("FragmentCollection", object)
ReferencesType = QueriesType = Union[list[object], tuple[object], np.ndarray]
ScoreFilter = Callable[[np.ndarray], bool]

"""Result of a similarity function"""
Score = Union[float, tuple[float, int]]
