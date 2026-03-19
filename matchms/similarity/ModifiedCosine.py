import logging
from typing import Optional, Sequence, Tuple
import numpy as np
from matchms.typing import SpectrumType
from .BaseSimilarity import BaseSimilarity
from .FlashSimilarity import FlashCosine
from .ModifiedCosineGreedy import ModifiedCosineGreedy
from .ModifiedCosineHungarian import ModifiedCosineHungarian


logger = logging.getLogger("matchms")


class ModifiedCosine(BaseSimilarity):
    """Calculate an approximate modified cosine score between mass spectra.

    This is matchms central Modified Cosine class. 
    The Modified Cosine score aims at quantifying the similarity between two
    mass spectra. Two peaks are considered a potential match if their m/z ratios
    lie within the given ``tolerance``, or if their m/z ratios lie within the
    tolerance once a mass-shift is applied. The mass shift is the difference in
    precursor m/z between the two spectra.

    Matchms provides various implementations of the Modified Cosine score which
    are combined here in what we believe to be the typical best choice for most users.

    By default, the parameter ``use_hungarian`` is set to False, which means that
    the greedy algorithm is used to find the best matches. This is typically faster
    than the Hungarian algorithm, and for most applications the results are very similar.
    If you need the exact optimal solution, you can set ``use_hungarian`` to True,
    which will use the Hungarian algorithm to find the best matches.

    For more conceptual context, see Watrous et al. 
    [PNAS, 2012, https://www.pnas.org/content/109/26/E1743].
    """

    is_commutative = True
    score_datatype = [("score", np.float64), ("matches", "int")]
    score_fields = ("score", "matches")

    def __init__(
            self,
            tolerance: float = 0.1,
            intensity_power: float = 1.0,
            use_hungarian: bool = False,
            ):
        """Initialize the modified cosine score class.

        Parameters
        ----------
        tolerance:
            Peaks will be considered a match when <= tolerance apart. Default is 0.1.
        intensity_power:
            The power to raise intensity to in the cosine function. The default is 1.
        use_hungarian:
            Whether to use the Hungarian algorithm to find the best matches. The default is False,
            which means that the greedy algorithm is used to find the best matches.
            The greedy algorithm is typically faster than the Hungarian algorithm, and for most
            applications the results are very similar.
        """
        self.tolerance = tolerance
        self.intensity_power = intensity_power
        self.use_hungarian = use_hungarian

    def pair(self, spectrum_1: SpectrumType, spectrum_2: SpectrumType) -> Tuple[float, int]:
        """Calculate approximate modified cosine score between two spectra."""

        if self.use_hungarian:
            modcos = ModifiedCosineHungarian(
                tolerance=self.tolerance,
                intensity_power=self.intensity_power,
            )
        else:
            modcos = ModifiedCosineGreedy(
                tolerance=self.tolerance,
                intensity_power=self.intensity_power,
            )
        return modcos.pair(spectrum_1, spectrum_2)


    def matrix(
            self,
            spectra_1: Sequence[SpectrumType],
            spectra_2: Optional[Sequence[SpectrumType]] = None,
            score_fields: Optional[Sequence[str]] = None,
            progress_bar: bool = True,
            n_jobs: int = -1,
        ):
        """
        Calculate matrix of Modified Cosine scores.

        Parameters
        ----------
        spectra_1
            First collection of input spectra.
        spectra_2
            Second collection of input spectra. If None, compare `spectra_1`
            against itself.
        score_fields
            Requested score fields. Only ``("score",)`` is supported.
        progress_bar
            When True, show a progress bar.
        n_jobs
            Number of parallel jobs to run.
            Default is -1, which means that all available CPUs minus one will be used.

        Returns
        -------
        Scores
            Dense score matrix as a ``Scores`` object.
        """
        if self.use_hungarian:
            modcos = ModifiedCosineHungarian(
                tolerance=self.tolerance,
                intensity_power=self.intensity_power,
            )

            return modcos.matrix(
                spectra_1=spectra_1,
                spectra_2=spectra_2,
                score_fields=score_fields,
                progress_bar=progress_bar,
            )
        modcos = FlashCosine(
            matching_mode="hybrid",
            tolerance=self.tolerance,
            intensity_power=self.intensity_power,
            )

        return modcos.matrix(
            spectra_1=spectra_1,
            spectra_2=spectra_2,
            score_fields=score_fields,
            progress_bar=progress_bar,
            n_jobs=n_jobs,
        )
