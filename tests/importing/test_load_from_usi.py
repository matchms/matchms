from unittest.mock import Mock, patch
import numpy as np
from matchms.importing import load_from_usi
from ..builder_Spectrum import SpectrumBuilder


@patch("requests.get")
def test_normal(mock_get):
    mock_get.return_value = Mock(ok=True)
    mock_get.return_value.json.return_value = {"peaks": [[1., 2.], [3., 4.]]}
    mock_get.return_value.headers.get = Mock(return_value="application/json")

    spec = load_from_usi("something")
    expected_metadata = {"usi": "something", "server": "https://metabolomics-usi.gnps2.org", "precursor_mz": None}
    expected = SpectrumBuilder().with_mz(np.array([1., 3.])).with_intensities(
        np.array([2., 4.])).with_metadata(expected_metadata,
                                          metadata_harmonization=True).build()
    assert spec == expected


@patch("requests.get")
def test_404(mock_get):
    mock_get.return_value = Mock(ok=True)
    mock_get.return_value.status_code = 404
    mock_get.return_value.json.return_value = None
    mock_get.return_value.headers.get = Mock(return_value="application/json")

    spec = load_from_usi("something")
    expected = None
    assert spec == expected


@patch("requests.get")
def test_no_peaks(mock_get):
    mock_get.return_value = Mock(ok=True)
    mock_get.return_value.json.return_value = {"peaks": []}
    mock_get.return_value.headers.get = Mock(return_value="application/json")

    spec = load_from_usi("something")
    expected = None
    assert spec == expected


def test_api_call():
    usi = "mzspec:MASSBANK::accession:SM858102" # taken from load_from_usi docstring

    spec = load_from_usi(usi)
    
    assert spec is not None
    assert hasattr(spec, "peaks")
    assert len(spec.peaks.mz) > 0
    assert "usi" in spec.metadata
    assert spec.metadata["usi"] == usi
