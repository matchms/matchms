from collections.abc import Callable
from typing import NewType
import numpy as np


SpectrumType = NewType("Spectrum", object)
ScoresType = NewType("Scores", object)
SpectraCollectionType = NewType("SpectraCollection", object)
FragmentCollectionType = NewType("FragmentCollection", object)
ReferencesType = QueriesType = list[object] | tuple[object] | np.ndarray
ScoreFilter = Callable[[np.ndarray], bool]

"""Result of a similarity function"""
Score = float | tuple[float, int]
