"""Submodule providing a reference spectrum for salicin."""

import numpy as np
from ..Spectrum import Spectrum


SALICIN_PRECURSOR_MZ: float = 321.0750

SALICIN_MZ: np.ndarray = np.array(
    [
        52.27001,
        55.74109,
        57.76688,
        60.32213,
        64.12009,
        82.39278,
        87.02966,
        91.02013,
        92.37563,
        93.13435,
        112.07272,
        116.92837,
        123.04468,
        138.61075,
        140.95815,
        146.96143,
        174.95708,
        184.95059,
        238.55797,
        305.44812,
        321.07443,
    ]
)

SALICIN_INTENSITIES: np.ndarray = np.array(
    [
        2309.0,
        1977.0,
        2003.0,
        2102.0,
        2177.0,
        2127.0,
        2376.0,
        2380.0,
        2703.0,
        2200.0,
        2232.0,
        2923.0,
        2173.0,
        2257.0,
        2367.0,
        4363.0,
        31526.0,
        5119.0,
        2252.0,
        2233.0,
        22755.0,
    ]
)


def salicin() -> Spectrum:
    """Return a reference spectrum for salicin."""
    return Spectrum(
        mz=SALICIN_MZ,
        intensities=SALICIN_INTENSITIES,
        metadata={"precursor_mz": SALICIN_PRECURSOR_MZ},
    )
