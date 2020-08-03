import os
from matchms.importing import load_from_mzxml


def test_load_from_mzxml():
    """Test parsing of mzxml file to spectrum objects"""

    module_root = os.path.dirname(__file__)
    mzxml_file = os.path.join(module_root, "testdata.mzxml")
    spectrum = list(load_from_mzxml(mzxml_file))[0]

    assert int(spectrum.get("precursor_mz")) == 343, "Expected different precursor m/z"
    assert spectrum.get("charge") == -1, "Expected different charge."
    assert len(spectrum.peaks) == 50
