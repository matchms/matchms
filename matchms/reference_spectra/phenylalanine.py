"""Submodule providing a reference spectrum for phenylalanine."""

import numpy as np
from ..Spectrum import Spectrum


PHENYLANINE_PRECURSOR_MZ: float = 166.086

PHENYLANINE_MZ: np.ndarray = np.array(
    [
        84.329651,
        95.261452,
        104.255539,
        107.151161,
        108.493408,
        120.354172,
        121.332741,
        122.174088,
        123.061447,
        123.876312,
        125.26976,
        127.950974,
        131.030273,
        131.898819,
        132.987488,
        133.966949,
        135.217804,
        136.683533,
        138.050446,
        148.049011,
        149.007416,
        149.634232,
        150.754486,
    ]
)

PHENYLANINE_INTENSITIES: np.ndarray = np.array(
    [
        802.63324,
        527.747314,
        232.556427,
        297.472168,
        113.401199,
        8457614.0,
        5223.298828,
        1784.401855,
        733.603821,
        514.92334,
        680.641602,
        341.421326,
        44541.625,
        596.891602,
        781.120422,
        2167.02002,
        130.054764,
        264.539215,
        1620.135986,
        55389.789062,
        364218.59375,
        4676.80957,
        590.623291,
    ]
)


def phenylalanine() -> Spectrum:
    """Return a reference spectrum for phenylalanine."""
    return Spectrum(
        mz=PHENYLANINE_MZ,
        intensities=PHENYLANINE_INTENSITIES,
        metadata={"precursor_mz": PHENYLANINE_PRECURSOR_MZ},
    )
