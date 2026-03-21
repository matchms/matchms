from typing import Callable, List, NewType, Tuple, Union
import numpy as np


SpectrumType = NewType("Spectrum", object)
ScoresType = NewType("Scores", object)
SpectraCollectionType = NewType("SpectraCollection", object)
FragmentCollectionType = NewType("FragmentCollection", object)
ReferencesType = QueriesType = Union[List[object], Tuple[object], np.ndarray]
ScoreFilter = Callable[[np.ndarray], bool]

"""Result of a similarity function"""
Score = Union[float, Tuple[float, int]]
