import os
import numpy
from matchms import Spectrum
from matchms.importing import load_from_msp


def assert_matching_inchikey(molecule, expected_inchikey):
    assert molecule.get("inchikey").lower() == expected_inchikey.lower(), "Expected different InChIKey."


def test_load_from_msp_spaces_1():
    """
    Test parse of msp file to spectrum objects.
    Check if InChiKey is loaded correctly.
    """

    module_root = os.path.join(os.path.dirname(__file__), "..")
    spectrums_file = os.path.join(module_root, "tests", "MoNA-export-GC-MS-first10.msp")
    spectrum = load_from_msp(spectrums_file)

    expected_inchikey = numpy.array([
        "ALRLPDGCPYIVHP-UHFFFAOYSA-N", "UFBJCMHMOXMLKC-UHFFFAOYSA-N", "WDNBURPWRNALGP-UHFFFAOYSA-N",
        "RANCECPPZPIPNO-UHFFFAOYSA-N", "HOLHYSJJBXSLMV-UHFFFAOYSA-N", "UMPSXRYVXUPCOS-UHFFFAOYSA-N",
        "HFZWRUODUSTPEG-UHFFFAOYSA-N", "VPOMSPZBQMDLTM-UHFFFAOYSA-N", "LHJGJYXLEPZJPM-UHFFFAOYSA-N",
        "LINPIYWFGCPVIE-UHFFFAOYSA-N"
    ])

    for k, n in enumerate(spectrum):
        assert_matching_inchikey(n, expected_inchikey[k])


def test_load_from_msp_spaces_2():
    """
    Test parse of msp file to spectrum objects.
    Check if peak m/z and intensity are loaded correctly if separated by spaces.
    """
    module_root = os.path.join(os.path.dirname(__file__), "..")
    spectrums_file = os.path.join(module_root, "tests", "MoNA-export-GC-MS-first10.msp")
    spectrum = next(iter(load_from_msp(spectrums_file)))

    expected_mz = numpy.array([
        51, 55, 57, 58, 59, 60, 61, 62, 63, 66, 68, 70, 72, 73, 74, 75, 76, 78,
        80, 81, 82, 83, 86, 87, 92, 93, 94, 98, 99, 100, 104, 107, 108, 110,
        112, 113, 115, 116, 120, 122, 123, 124, 125, 126, 134, 135, 137, 147,
        149, 150, 151, 159, 162, 163, 173, 174, 175, 177, 187, 188, 189, 190,
        191, 198, 199, 200, 201, 202, 203, 207, 214, 217, 218, 247, 248
    ])

    expected_intensities = numpy.array([
        2.66, 8, 7.33, 1.33, 1.33, 14, 1.33, 3.33, 3.33, 1.33, 8.66, 2, 5.33,
        7.33, 3.33, 2.66, 2, 1.33, 4, 2, 1.33, 3.33, 12.66, 8.66, 2, 10, 6,
        14.66, 83.33, 60.66, 4, 1.33, 1.33, 3.33, 1.33, 1.33, 1.33, 1.33, 1.33,
        4, 2.66, 2.66, 2, 1.33, 1.33, 2, 1.33, 1.33, 2, 4.66, 3.33, 2, 2, 2.66,
        2, 8.66, 4.66, 2, 5.33, 4.66, 56.66, 12, 16.66, 10.66, 9.33, 72.66,
        99.99, 16, 1.33, 1.33, 1.33, 25.33, 5.33, 52.66, 10.16
    ])

    numpy.testing.assert_array_almost_equal(spectrum.peaks.mz, expected_mz)
    numpy.testing.assert_array_almost_equal(spectrum.peaks.intensities, expected_intensities)


def test_load_from_msp_tabs():
    """Test parse of msp file to spectrum objects with tabstop separator."""

    module_root = os.path.join(os.path.dirname(__file__), "..")
    spectrums_file = os.path.join(module_root, "tests", "rcx_gc-ei_ms_20201028_perylene.msp")
    spectra = load_from_msp(spectrums_file)

    expected_inchikey = numpy.array([
        "CSHWQDPOILHKBI-UHFFFAOYSA-N"
    ])

    expected_mz = numpy.array([[
        112.03071, 113.03854, 124.03076, 124.53242, 125.03855, 125.54019,
        126.04636, 126.54804, 222.04645, 224.06192, 226.04175, 246.04646,
        248.06204, 249.07072, 250.07765, 251.07967, 252.09323, 253.09656,
        254.09985
    ]])

    expected_intensities = numpy.array([[
        49892, 87510, 100146, 24923, 179254, 49039, 131679, 36313, 28905,
        55632, 37413, 23286, 140007, 62236, 641789, 137600, 1955166, 402252,
        39987
    ]])

    for idx, spectrum in enumerate(spectra):
        assert_matching_inchikey(spectrum, expected_inchikey[idx])
        numpy.testing.assert_array_almost_equal(spectrum.peaks.mz, expected_mz[idx])
        numpy.testing.assert_array_almost_equal(spectrum.peaks.intensities, expected_intensities[idx])


def test_load_from_msp_multiline():
    """Test parse of msp file to spectrum objects with ';' separator and multiple peaks in one line."""

    module_root = os.path.join(os.path.dirname(__file__), "..")
    spectrums_file = os.path.join(module_root, "tests", "multiline_semicolon.msp")

    actual = list(load_from_msp(spectrums_file))
    expected = [
        Spectrum(
            mz=numpy.array([
                12, 24, 25, 26, 27, 35, 36, 37, 38, 39, 47, 48, 49, 50, 51
            ]).astype(float),
            intensities=numpy.array([
                0, 0, 2, 4, 0, 19, 26, 120, 49, 5, 25, 11, 60, 104, 13
            ]).astype(float),
            metadata={
                "name": "Compound A",
                "num peaks": '15'
            }),
        Spectrum(
            mz=numpy.array([
                40, 41, 42, 43, 44, 46, 50, 51, 52, 53
            ]).astype(float),
            intensities=numpy.array([
                147, 57, 13, 52, 30, 1, 1, 8, 6, 1
            ]).astype(float),
            metadata={
                "name": "JWH 081",
                "num peaks": '10'
            })
    ]

    assert actual == expected
