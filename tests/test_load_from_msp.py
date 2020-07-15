import os
import numpy
from matchms.importing import load_from_msp


def test_load_from_msp():
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
        assert n.get("inchikey").lower() == expected_inchikey[k].lower(), "Expected different InChIKey."
