import os
import numpy as np
import pytest
from matchms import Spectrum
from matchms.importing.load_from_msp import load_from_msp, parse_metadata
from tests.builder_Spectrum import SpectrumBuilder


def assert_matching_inchikey(molecule, expected_inchikey):
    assert molecule.get("inchikey").lower(
    ) == expected_inchikey.lower(), "Expected different InChIKey."


def assert_matching_mass(molecule, expected_mass):
    assert np.isclose(molecule.get("parent_mass"), expected_mass, rtol=1e-5), \
        "Expected different InChIKey."


def assert_matching_metadata_string(molecule, expected_entry, field):
    assert molecule.get(
        field) == expected_entry, f"Expected different entry for {field}."


def test_load_from_msp_spaces_mona_1():
    """
    Test parse of msp file to spectrum objects using MoNA msp file.
    Check if InChiKey is loaded correctly.
    """

    module_root = os.path.join(os.path.dirname(__file__), "..")
    spectra_file = os.path.join(
        module_root, "testdata", "MoNA-export-GC-MS-first10.msp")
    spectrum = list(load_from_msp(spectra_file))

    expected_inchikey = np.array([
        "ALRLPDGCPYIVHP-UHFFFAOYSA-N", "UFBJCMHMOXMLKC-UHFFFAOYSA-N", "WDNBURPWRNALGP-UHFFFAOYSA-N",
        "RANCECPPZPIPNO-UHFFFAOYSA-N", "HOLHYSJJBXSLMV-UHFFFAOYSA-N", "UMPSXRYVXUPCOS-UHFFFAOYSA-N",
        "HFZWRUODUSTPEG-UHFFFAOYSA-N", "VPOMSPZBQMDLTM-UHFFFAOYSA-N", "LHJGJYXLEPZJPM-UHFFFAOYSA-N",
        "LINPIYWFGCPVIE-UHFFFAOYSA-N"
    ])

    assert len(spectrum) > 0
    for k, n in enumerate(spectrum):
        assert_matching_inchikey(n, expected_inchikey[k])


def test_load_from_msp_spaces_mona_2():
    """
    Test parse of msp file to spectrum objects using MoNA msp file.
    Check if peak m/z and intensity are loaded correctly if separated by spaces.
    """
    module_root = os.path.join(os.path.dirname(__file__), "..")
    spectra_file = os.path.join(
        module_root, "testdata", "MoNA-export-GC-MS-first10.msp")
    spectrum = next(iter(load_from_msp(spectra_file)))

    expected_mz = np.array([
        51, 55, 57, 58, 59, 60, 61, 62, 63, 66, 68, 70, 72, 73, 74, 75, 76, 78,
        80, 81, 82, 83, 86, 87, 92, 93, 94, 98, 99, 100, 104, 107, 108, 110,
        112, 113, 115, 116, 120, 122, 123, 124, 125, 126, 134, 135, 137, 147,
        149, 150, 151, 159, 162, 163, 173, 174, 175, 177, 187, 188, 189, 190,
        191, 198, 199, 200, 201, 202, 203, 207, 214, 217, 218, 247, 248
    ])

    expected_intensities = np.array([
        2.66, 8, 7.33, 1.33, 1.33, 14, 1.33, 3.33, 3.33, 1.33, 8.66, 2, 5.33,
        7.33, 3.33, 2.66, 2, 1.33, 4, 2, 1.33, 3.33, 12.66, 8.66, 2, 10, 6,
        14.66, 83.33, 60.66, 4, 1.33, 1.33, 3.33, 1.33, 1.33, 1.33, 1.33, 1.33,
        4, 2.66, 2.66, 2, 1.33, 1.33, 2, 1.33, 1.33, 2, 4.66, 3.33, 2, 2, 2.66,
        2, 8.66, 4.66, 2, 5.33, 4.66, 56.66, 12, 16.66, 10.66, 9.33, 72.66,
        99.99, 16, 1.33, 1.33, 1.33, 25.33, 5.33, 52.66, 10.16
    ])

    np.testing.assert_array_almost_equal(spectrum.peaks.mz, expected_mz)
    np.testing.assert_array_almost_equal(
        spectrum.peaks.intensities, expected_intensities)


def test_load_from_msp_spaces_massbank_1():
    """
    Test parse of msp file to spectrum objects using MassBank msp file.
    Check if InChiKey is loaded correctly and if comment field is handled correctly.
    """

    module_root = os.path.join(os.path.dirname(__file__), "..")
    spectra_file = os.path.join(
        module_root, "testdata", "massbank_five_spectra.msp")
    spectrum = list(load_from_msp(spectra_file))

    expected_inchikey = [
        "XTWYTFMLZFPYCI-UHFFFAOYSA-N",
        "BEJNERDRQOWKJM-UHFFFAOYSA-N",
        "UVKZSORBKUEBAZ-UHFFFAOYSA-N",
        "TTWJBBZEZQICBI-UHFFFAOYSA-N",
        "SIIRBDOFKDACOK-UHFFFAOYSA-N",
    ]

    for k, n in enumerate(spectrum):
        assert_matching_inchikey(n, expected_inchikey[k])

    expected_parent_comment = [
        428.31, 141.0193, 267.1856, 300.1473, 415.234
    ]

    assert len(spectrum) > 0
    for k, n in enumerate(spectrum):
        assert np.isclose(n.get("parent"), expected_parent_comment[k])


def test_load_from_msp_tabs():
    """Test parse of msp file to spectrum objects with tabstop separator."""

    module_root = os.path.join(os.path.dirname(__file__), "..")
    spectra_file = os.path.join(
        module_root, "testdata", "rcx_gc-ei_ms_20201028_perylene.msp")
    spectra = list(load_from_msp(spectra_file))

    expected_inchikey = np.array([
        "CSHWQDPOILHKBI-UHFFFAOYSA-N"
    ])

    expected_mz = np.array([[
        112.03071, 113.03854, 124.03076, 124.53242, 125.03855, 125.54019,
        126.04636, 126.54804, 222.04645, 224.06192, 226.04175, 246.04646,
        248.06204, 249.07072, 250.07765, 251.07967, 252.09323, 253.09656,
        254.09985
    ]])

    expected_intensities = np.array([[
        49892, 87510, 100146, 24923, 179254, 49039, 131679, 36313, 28905,
        55632, 37413, 23286, 140007, 62236, 641789, 137600, 1955166, 402252,
        39987
    ]])

    expected_peak_comments = {113.03854: "Theoretical m/z 113.039125, Mass diff 0 (0 ppm), Formula C9H5",
                              125.03855: "Theoretical m/z 125.039125, Mass diff 0 (0 ppm), Formula C10H5",
                              249.07072: "Theoretical m/z 249.070425, Mass diff -0.001 (0 ppm), Formula C20H9",
                              252.09323: "Theoretical m/z 252.093354, Mass diff 0 (0.49 ppm), SMILES "
                                         "C1=CC=2C=CC=C3C4=CC=CC5=CC=CC(C(=C1)C23)=C54, Annotation [C20H12]+, "
                                         "Rule of HR False"}

    assert len(spectra) > 0

    for idx, spectrum in enumerate(spectra):
        assert_matching_inchikey(spectrum, expected_inchikey[idx])
        np.testing.assert_array_almost_equal(
            spectrum.peaks.mz, expected_mz[idx])
        np.testing.assert_array_almost_equal(
            spectrum.peaks.intensities, expected_intensities[idx])
        assert spectrum.peak_comments == expected_peak_comments


def test_load_from_msp_multiline():
    """Test parse of msp file to spectrum objects with ';' separator and multiple peaks in one line."""

    module_root = os.path.join(os.path.dirname(__file__), "..")
    spectra_file = os.path.join(
        module_root, "testdata", "multiline_semicolon.msp")

    actual = list(load_from_msp(spectra_file))
    expected = [
        Spectrum(
            mz=np.array([
                12, 24, 25, 26, 27, 35, 36, 37, 38, 39, 47, 48, 49, 50, 51
            ]).astype(float),
            intensities=np.array([
                0, 0, 2, 4, 0, 19, 26, 120, 49, 5, 25, 11, 60, 104, 13
            ]).astype(float),
            metadata={
                "name": "Compound A",
                "num peaks": '15'
            }),
        Spectrum(
            mz=np.array([
                40, 41, 42, 43, 44, 46, 50, 51, 52, 53
            ]).astype(float),
            intensities=np.array([
                147, 57, 13, 52, 30, 1, 1, 8, 6, 1
            ]).astype(float),
            metadata={
                "name": "JWH 081",
                "num peaks": '10',
            })
    ]

    assert actual == expected


def test_load_from_msp_diverse_spectrum_collection():
    """
    Test parse of msp file to spectrum objects using msp file containing various
    spectra. Some will contain duplicate entries (e.g. ExactMass field and exact_mass in comments).
    Check if InChiKey is loaded correctly.
    """

    module_root = os.path.join(os.path.dirname(__file__), "..")
    spectra_file = os.path.join(
        module_root, "testdata", "test_spectra_collection.msp")
    spectrum = load_from_msp(spectra_file)

    expected_inchikey = np.array([
        "UDOOPSJCRMKSGL-ZHACJKMWSA-N", "QQVDJLLNRSOCEL-UHFFFAOYSA-N", "KPZYYKDXZKFBQU-UHFFFAOYSA-N"
    ])
    for k, n in enumerate(spectrum):
        assert_matching_inchikey(n, expected_inchikey[k])

    expected_parent_mass = np.array([
        224.083729624, 125.0241797459999
    ])
    for k, n in enumerate(spectrum):
        assert_matching_mass(n, expected_parent_mass[k])

    expected_adducts = np.array([
        "M-H", "[M-H]-", "M+H"
    ])
    for k, n in enumerate(spectrum):
        assert_matching_metadata_string(n, expected_adducts[k], "adduct")


def test_load_msl():
    module_root = os.path.join(os.path.dirname(__file__), "..")
    spectra_file = os.path.join(module_root, "testdata", "JL_2021_V2.msl")
    actual = list(load_from_msp(spectra_file))[0]

    metadata = {
        "COMPOUND_NAME": "G3P",
        "CASNO": "TUBE2_~1-N1010",
        "RETENTION_INDEX": 1586.2,
        "RETENTION_TIME": 9.648,
        "COMMENT": "RI=1586.2,   9.6481 min TUBE2_28-01-2020_18-48-00|RI:1586.2",
        "SOURCE": "C:\\USERS\\UTILISATEUR\\DESKTOP\\METABOLOME\\calib\\standard.msl",
        "NUM PEAKS": '30'
    }

    mz = [52,55,56,59,61,62,63,66,68,70,73,75,76,77,80,82,84,85,86,87,89,90,91,92,95,96,98,100,101,102 ]
    intensities = [3, 136, 14, 96, 31, 2,1,1,4,11,1000,303,22,12,4,19,31,7,7,3,84,8,8,1,1,1,4,6,4,4]
    expected = SpectrumBuilder().with_metadata(metadata).with_mz(mz).with_intensities(intensities).build()

    assert actual == expected


def test_load_golm_style_msp():
    module_root = os.path.join(os.path.dirname(__file__), "..")
    spectra_file = os.path.join(module_root, "testdata", "golm.msp")
    edge_case_spectra_file = os.path.join(module_root, "testdata", "edge_golm.msp")
    actual = list(load_from_msp(spectra_file))
    edge_case_actual = list(load_from_msp(edge_case_spectra_file))

    assert len(actual) == 3
    assert len(actual[0].mz) == 50
    assert len(edge_case_actual[0].metadata["synonyms"]) == 11
    assert edge_case_actual[0].metadata["retention_index"] == 986.12
    assert "inchi" in edge_case_actual[0].metadata.keys()


def test_load_msp_with_comments_including_quotes():
    module_root = os.path.join(os.path.dirname(__file__), "..")
    spectra_file = os.path.join(module_root, "testdata", "comments_with_quotes.msp")
    actual = next(load_from_msp(spectra_file))

    assert len(actual.mz) == 248
    assert actual.get("columntype") == "Semi-standard non-polar, TG-5SILMSwith10mGuard, 30mx0.25mmx0.25um"
    assert actual.get("carriergasflowrate") == "1.0 mL/min"
    assert actual.get("category") == "Amino Acid"


def test_load_msp_with_scientific_notation():
    module_root = os.path.join(os.path.dirname(__file__), "..")
    spectra_file = os.path.join(module_root, "testdata", "test_spectra_collection.msp")
    actual = list(load_from_msp(spectra_file))

    assert len(actual) == 3


@pytest.mark.parametrize("input_line, expected_output", [
    ['comments: "SMILES="', {}],
    ['comments: SMILES="CC(O)C(O)=O"', {"smiles": "CC(O)C(O)=O"}],
    ['comments: mass=12.0', {"mass": '12.0'}],
    ['name: 3,4-DICHLOROPHENOL', {'name': '3,4-DICHLOROPHENOL'}],
    ['comments: "SMILES=CC(O)C(O)=O"', {"smiles": "CC(O)C(O)=O"}],
    ['comments: "DB#=JP000001"', {"db#":"JP000001"}],
])
def test_parse_metadata(input_line, expected_output):
    """tests if metadata is correctly parsed"""
    params = {}
    parse_metadata(input_line, params)
    assert params == expected_output
