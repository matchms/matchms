import os
import numpy
from matchms.importing import load_from_msp

def assert_matching_inchikey(molecule, expected_inchikey):
    assert molecule.get("inchikey").lower() == expected_inchikey.lower(), "Expected different InChIKey."

def test_load_from_msp_spaces():
    """Test parse of msp file to sprectum objects"""

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

def test_load_from_msp_tabs():
    """Test parse of msp file to sprectum objects"""

    module_root = os.path.join(os.path.dirname(__file__), "..")
    spectrums_file = os.path.join(module_root, "tests", "rcx_gc-ei_ms_20201028_perylene.msp")
    spectra = load_from_msp(spectrums_file)

    expected_inchikey = numpy.array([
        "CSHWQDPOILHKBI-UHFFFAOYSA-N"
    ])

    expected_mz = numpy.array([
        [112.03071, 113.03854, 124.03076, 124.53242, 125.03855, 125.54019,
        126.04636, 126.54804, 222.04645, 224.06192, 226.04175, 246.04646,
        248.06204, 249.07072, 250.07765, 251.07967, 252.09323, 253.09656,
        254.09985]
    ])

    expected_intensities = numpy.array([
        [49892, 87510, 100146, 24923, 179254, 49039, 131679, 36313, 28905,
        55632, 37413, 23286, 140007, 62236, 641789, 137600, 1955166, 402252,
        39987]
    ])

    for idx, spectrum in enumerate(spectra):
        assert_matching_inchikey(spectrum, expected_inchikey[idx])
        numpy.testing.assert_array_almost_equal(spectrum.peaks.mz, expected_mz[idx])
        numpy.testing.assert_array_almost_equal(spectrum.peaks.intensities, expected_intensities[idx])


    