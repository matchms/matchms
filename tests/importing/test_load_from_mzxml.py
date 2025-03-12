import os
from pathlib import Path
import pytest
from matchms.importing import load_from_mzxml


@pytest.mark.parametrize("mzxml_file", [
    (os.path.join(os.path.dirname(__file__), "..", "testdata", "testdata.mzXML")),
    (Path(os.path.join(os.path.dirname(__file__), "..", "testdata", "testdata.mzXML")))
])
def test_load_from_mzxml(mzxml_file):
    """Test parsing of mzxml file to spectrum objects"""

    spectrum = next(load_from_mzxml(mzxml_file))

    assert len(list(load_from_mzxml(mzxml_file))) == 1, "Expected single spectrum"
    assert int(spectrum.get("precursor_mz")) == 343, "Expected different precursor m/z"
    assert spectrum.get("charge") == -1, "Expected different charge."
    assert len(spectrum.peaks) == 50


def test_load_from_mzxml_ms_levels():
    """Test parsing of mzxml file to spectrum objects using different ms levels."""

    module_root = os.path.join(os.path.dirname(__file__), "..")
    mzxml_file = os.path.join(module_root, "testdata", "testdata.mzXML")

    expected_num_spectra = [1, 1, 3, 0]
    for i in range(4):
        ms_level = i + 1
        spectra = list(load_from_mzxml(mzxml_file, ms_level))
        assert len(spectra) == expected_num_spectra[i], (
            f"Expected different number of spectra for ms_level={ms_level}")
