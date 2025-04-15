import pytest
from matchms.filtering import add_retention_index, add_retention_time
from ..builder_Spectrum import SpectrumBuilder


@pytest.mark.parametrize(
    "metadata, expected",
    [
        [{"retention_time": 100.0}, 100.0],
        [{"retention_time": "NA"}, None],
        [{"retention_time": "100.0"}, 100.0],
        [{"retentiontime": 200}, 200.0],
        [{"retentiontime": -1}, None],
        [{"retentiontime": "-1"}, None],
        [{"rt": "4.810467 min"}, 288.62802],
        [{"rt": "no retention time in min available"}, None],
        [{"rt": 200}, 200.0],
        [{"RT": 200}, 200.0],
        [{"RT_Query": 200}, 200.0],
        [{"nothing": "200"}, None],
        [{"scan_start_time": 0.629566}, 0.629566],
        [{"scan_start_time": [0.629566]}, 0.629566],
        [{"rt": None, "retentiontime": 12.17}, 12.17],
        [{"retention_time": "100.0 sec"}, 100.0],
    ],
)
def test_add_retention_time(metadata, expected):
    spectrum_in = SpectrumBuilder().with_metadata(metadata).build()

    spectrum = add_retention_time(spectrum_in)
    actual = spectrum.get("retention_time")

    if expected is None:
        assert actual is None
    else:
        assert actual == expected and isinstance(actual, float)


@pytest.mark.parametrize(
    "metadata, expected",
    [
        [{"retention_index": 100.0}, 100.0],
        [{"retention_index": "NA"}, None],
        [{"retention_index": "100.0"}, 100.0],
        [{"retentionindex": 200}, 200.0],
        [{"retentionindex": -1}, None],
        [{"retentionindex": "-1"}, None],
        [{"ri": 200}, 200.0],
        [{"RI": 200}, 200.0],
        [{"nothing": "200"}, None],
    ],
)
def test_add_retention_index(metadata, expected):
    spectrum_in = SpectrumBuilder().with_metadata(metadata).build()

    spectrum = add_retention_index(spectrum_in)
    actual = spectrum.get("retention_index")

    if expected is None:
        assert actual is None
    else:
        assert actual == expected and isinstance(actual, float)


def test_empty_spectrum():
    spectrum_in = None
    spectrum = add_retention_time(spectrum_in)
    assert spectrum is None, "Expected different handling of None spectrum."

    spectrum = add_retention_index(spectrum_in)
    assert spectrum is None, "Expected different handling of None spectrum."
