import os
from pathlib import Path
import pytest
from matchms.importing import load_from_mzml


@pytest.mark.parametrize("mzml_file", [
    (os.path.join(os.path.dirname(__file__), "..", "testdata", "testdata.mzml")),
    (Path(os.path.join(os.path.dirname(__file__), "..", "testdata", "testdata.mzml")))
])
def test_load_from_mzml(mzml_file):
    """Test parsing of mzml file to spectrum objects"""

    spectra = list(load_from_mzml(mzml_file))

    assert len(spectra) == 10, "Expected 10 spectra."
    assert int(spectra[5].get("precursor_mz")) == 177, "Expected different precursor m/z"
    assert float(spectra[0].get("scan_start_time")[0]) == 0.616304, "Expected different time."
    assert [len(x.peaks) for x in spectra] == [30, 28, 21, 70, 28, 20, 22, 27, 11, 25]
