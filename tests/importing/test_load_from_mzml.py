import os
from pathlib import Path
import numpy as np
import pytest
from matchms.importing import load_from_mzml


@pytest.mark.parametrize(
    "mzml_file",
    [
        (os.path.join(os.path.dirname(__file__), "..", "testdata", "testdata.mzml")),
        (Path(os.path.join(os.path.dirname(__file__), "..", "testdata", "testdata.mzml"))),
    ],
)
def test_load_from_mzml(mzml_file):
    """Test parsing of mzml file to spectrum objects"""

    spectra = list(load_from_mzml(mzml_file))
    assert len(spectra) == 10, "Expected 10 spectra."
    assert int(spectra[5].get("precursor_mz")) == 177, "Expected different precursor m/z"
    assert float(spectra[0].get("scan_start_time")) == 0.616304, "Expected different time."
    assert [len(x.peaks) for x in spectra] == [30, 28, 21, 70, 28, 20, 22, 27, 11, 25]


def test_load_from_mzml_2():
    """Test parsing of mzml file to spectrum objects"""
    mzml_file = os.path.join(os.path.dirname(__file__), "..", "testdata", "tiny.pwiz.1.1.mzML")
    spectra = list(load_from_mzml(mzml_file))
    assert len(spectra) == 1, "Expected 10 spectra."
    assert spectra[0].metadata == {
        "index": 1,
        "id": "scan=20",
        "count": 2,
        "defaultarraylength": 10,
        "instrumentconfigurationref": "LCQ_x0020_Deca",
        "spectrumref": "scan=19",
        "scan_window_lower_limit": 110.0,
        "scan_window_upper_limit": 905.0,
        "scan_start_time": 5.9905,
        "filter_string": "+ c d Full ms2  445.35@cid35.00 [ 110.00-905.00]",
        "preset_scan_configuration": 4.0,
        "peak_intensity": 120053.0,
        "activation-collision_energy": 35.0,
        "ms_level": 2,
        "lowest_observed_m/z": 320.39,
        "highest_observed_m/z": 1003.56,
        "base_peak_m/z": 456.347,
        "base_peak_intensity": 23433.0,
        "total_ion_current": 16675500.0,
        "isolationwindow-isolation_window_target_m/z": 445.3,
        "isolationwindow-isolation_window_lower_offset": 0.5,
        "isolationwindow-isolation_window_upper_offset": 0.5,
        "precursor_mz": 445.34,
        "charge": 2,
    }
    print(spectra[0].mz)
    print(spectra[0].intensities)

    np.testing.assert_array_equal(spectra[0].mz, np.array([0, 2, 4, 6, 8, 10, 12, 14, 16, 18]))
    np.testing.assert_array_equal(spectra[0].intensities, np.array([20, 18, 16, 14, 12, 10, 8, 6, 4, 2]))
