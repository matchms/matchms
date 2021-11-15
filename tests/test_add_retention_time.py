import numpy
import pytest
from matchms import Spectrum
from matchms.filtering import add_retention_time


@pytest.mark.parametrize("metadata, expected", [
    [{"retention_time": 100.0}, 100.0],
    [{"retention_time": "NA"}, None],
    [{"retention_time": "100.0"}, 100.0],
    [{"retentiontime": 200}, 200.0],
    [{"retentiontime": -1}, None],
    [{"retentiontime": "-1"}, None],
    [{"rt": 200}, 200.0],
    [{"RT": 200}, 200.0],
    [{"nothing": "200"}, None],
    [{'scan_start_time': 0.629566}, 0.629566]
])
def test_add_retention_time(metadata, expected):
    spectrum_in = Spectrum(mz=numpy.array(
        [], "float"), intensities=numpy.array([], "float"), metadata=metadata)

    spectrum = add_retention_time(spectrum_in)
    actual = spectrum.get("retention_time")

    if expected is None:
        assert actual is None
    else:
        assert actual == expected and isinstance(actual, float)
