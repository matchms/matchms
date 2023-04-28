import os
from matchms.importing import load_from_mzml


def test_load_from_mzml():
    """Test parsing of mzml file to spectrum objects"""

    module_root = os.path.join(os.path.dirname(__file__), "..")
    mzml_file = os.path.join(module_root, "testdata", "testdata.mzml")
    spectrums = list(load_from_mzml(mzml_file))

    assert len(spectrums) == 10, "Expected 10 spectrums."
    assert int(spectrums[5].get("precursor_mz")) == 177, "Expected different precursor m/z"
    assert float(spectrums[0].get("scan_start_time")[0]) == 0.616304, "Expected different time."
    assert [len(x.peaks) for x in spectrums] == [30, 28, 21, 70, 28, 20, 22, 27, 11, 25]
