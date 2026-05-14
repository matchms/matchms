import warnings
import numpy as np
import pytest
from matchms import Spectrum
from matchms.filtering._dispatch import (
    apply_spectrum_filter_to_collection,
    collection_filter,
)
from matchms.SpectraCollection import SpectraCollection


def _make_spectrum(identifier="spectrum", mz=None, intensities=None, metadata=None):
    if mz is None:
        mz = np.array([100.0, 200.0, 300.0])
    if intensities is None:
        intensities = np.array([0.1, 0.5, 1.0])
    if metadata is None:
        metadata = {"id": identifier}

    return Spectrum(
        mz=np.asarray(mz, dtype=float),
        intensities=np.asarray(intensities, dtype=float),
        metadata=metadata,
    )


def _make_collection():
    spectra = [
        _make_spectrum("s1", mz=[100.0, 200.0], intensities=[1.0, 2.0]),
        _make_spectrum("s2", mz=[150.0, 250.0], intensities=[3.0, 4.0]),
        _make_spectrum("s3", mz=[175.0, 275.0], intensities=[5.0, 6.0]),
    ]
    return SpectraCollection(spectra)


def test_collection_filter_returns_none_for_none_input():
    def _dummy_spectrum_filter(spectrum, clone=True):
        return spectrum

    public_filter = collection_filter(_dummy_spectrum_filter)

    assert public_filter(None) is None


def test_collection_filter_uses_spectrum_implementation_for_spectrum_input():
    spectrum = _make_spectrum()

    def _mark_spectrum_filter(spectrum, clone=True):
        spectrum = spectrum.clone() if clone else spectrum
        spectrum.set("processed", True)
        return spectrum

    public_filter = collection_filter(_mark_spectrum_filter)

    result = public_filter(spectrum)

    assert isinstance(result, Spectrum)
    assert result.get("processed") is True
    assert spectrum.get("processed") is None


def test_collection_filter_passes_non_collection_input_to_spectrum_implementation():
    def _identity_spectrum_filter(obj, clone=True):
        return {"received": obj, "clone": clone}

    public_filter = collection_filter(_identity_spectrum_filter)

    result = public_filter("not-a-spectrum", clone=False)

    assert result == {"received": "not-a-spectrum", "clone": False}


def test_collection_filter_prefers_native_collection_implementation():
    collection = _make_collection()

    def _spectrum_impl(spectrum, clone=True):
        raise AssertionError("Spectrum fallback should not be called.")

    def _collection_impl(collection, clone=True):
        return "native collection result"

    public_filter = collection_filter(
        _spectrum_impl,
        collection_impl=_collection_impl,
    )

    assert public_filter(collection) == "native collection result"


def test_collection_filter_raises_for_collection_without_fallback():
    collection = _make_collection()

    def _dummy_spectrum_filter(spectrum, clone=True):
        return spectrum

    public_filter = collection_filter(
        _dummy_spectrum_filter,
        collection_impl=None,
        allow_spectrum_fallback=False,
    )

    with pytest.raises(
        NotImplementedError,
        match="does not support SpectraCollection",
    ):
        public_filter(collection)


def test_collection_filter_warns_when_using_fallback_if_enabled():
    collection = _make_collection()

    def _dummy_spectrum_filter(spectrum, clone=True):
        return spectrum

    public_filter = collection_filter(
        _dummy_spectrum_filter,
        collection_impl=None,
        allow_spectrum_fallback=True,
        warn_on_fallback=True,
    )

    with pytest.warns(
        RuntimeWarning,
        match="no native SpectraCollection implementation",
    ):
        result = public_filter(collection)

    assert isinstance(result, SpectraCollection)
    assert len(result) == len(collection)


def test_collection_filter_does_not_warn_when_fallback_warning_disabled():
    collection = _make_collection()

    def _dummy_spectrum_filter(spectrum, clone=True):
        return spectrum

    public_filter = collection_filter(
        _dummy_spectrum_filter,
        collection_impl=None,
        allow_spectrum_fallback=True,
        warn_on_fallback=False,
    )

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        result = public_filter(collection)

    assert isinstance(result, SpectraCollection)
    assert len(result) == len(collection)


def test_apply_spectrum_filter_to_collection_returns_collection():
    collection = _make_collection()

    def _add_metadata(spectrum, clone=True):
        spectrum = spectrum.clone() if clone else spectrum
        spectrum.set("processed", True)
        return spectrum

    result = apply_spectrum_filter_to_collection(collection, _add_metadata)

    assert isinstance(result, SpectraCollection)
    assert len(result) == len(collection)
    assert result[0].get("processed") is True
    assert result[1].get("processed") is True
    assert result[2].get("processed") is True


def test_apply_spectrum_filter_to_collection_drops_none_results():
    collection = _make_collection()

    def _keep_only_s2(spectrum, clone=True):
        if spectrum.get("id") == "s2":
            return spectrum
        return None

    result = apply_spectrum_filter_to_collection(collection, _keep_only_s2)

    assert isinstance(result, SpectraCollection)
    assert len(result) == 1
    assert result[0].get("id") == "s2"


def test_apply_spectrum_filter_to_collection_returns_none_if_all_spectra_are_dropped():
    collection = _make_collection()

    def _drop_all(spectrum, clone=True):
        return None

    result = apply_spectrum_filter_to_collection(collection, _drop_all)

    assert result is None


def test_apply_spectrum_filter_to_collection_calls_spectrum_filter_with_clone_false():
    collection = _make_collection()
    observed_clone_values = []

    def _record_clone_argument(spectrum, clone=True):
        observed_clone_values.append(clone)
        return spectrum

    result = apply_spectrum_filter_to_collection(
        collection,
        _record_clone_argument,
        clone=True,
    )

    assert isinstance(result, SpectraCollection)
    assert observed_clone_values == [False, False, False]


def test_apply_spectrum_filter_to_collection_warns_when_clone_false_is_requested():
    collection = _make_collection()

    def _dummy_spectrum_filter(spectrum, clone=True):
        return spectrum

    with pytest.warns(
        RuntimeWarning,
        match="does not support true in-place modification",
    ):
        result = apply_spectrum_filter_to_collection(
            collection,
            _dummy_spectrum_filter,
            clone=False,
        )

    assert isinstance(result, SpectraCollection)
    assert result is not collection


def test_collection_filter_preserves_public_name_without_private_prefix_and_suffix():
    def _example_filter_spectrum(spectrum, clone=True):
        """Example docstring."""
        return spectrum

    public_filter = collection_filter(_example_filter_spectrum)

    assert public_filter.__name__ == "example_filter"
    assert public_filter.__doc__ == "Example docstring."


def test_collection_filter_forwards_args_and_kwargs_to_native_collection_impl():
    collection = _make_collection()

    def _spectrum_impl(spectrum, value, *, option=False, clone=True):
        return spectrum

    def _collection_impl(collection, value, *, option=False, clone=True):
        return {
            "collection": collection,
            "value": value,
            "option": option,
            "clone": clone,
        }

    public_filter = collection_filter(
        _spectrum_impl,
        collection_impl=_collection_impl,
    )

    result = public_filter(collection, 123, option=True, clone=False)

    assert result["collection"] is collection
    assert result["value"] == 123
    assert result["option"] is True
    assert result["clone"] is False


def test_collection_filter_forwards_args_and_kwargs_to_spectrum_impl():
    spectrum = _make_spectrum()

    def _spectrum_impl(spectrum, value, *, option=False, clone=True):
        return {
            "spectrum": spectrum,
            "value": value,
            "option": option,
            "clone": clone,
        }

    public_filter = collection_filter(_spectrum_impl)

    result = public_filter(spectrum, 123, option=True, clone=False)

    assert result["spectrum"] is spectrum
    assert result["value"] == 123
    assert result["option"] is True
    assert result["clone"] is False


def test_collection_filter_accepts_spectrum_in_keyword():
    spectrum = _make_spectrum()

    def _dummy_filter_spectrum(spectrum_in, mass_tolerance, clone=True):
        spectrum = spectrum_in.clone() if clone else spectrum_in
        spectrum.set("mass_tolerance", mass_tolerance)
        return spectrum

    public_filter = collection_filter(_dummy_filter_spectrum)

    result = public_filter(spectrum_in=spectrum, mass_tolerance=0.1)

    assert result is not spectrum
    assert result.get("mass_tolerance") == 0.1


def test_collection_filter_accepts_collection_as_spectrum_in_keyword():
    collection = _make_collection()

    def _dummy_filter_spectrum(spectrum_in, clone=True):
        return spectrum_in

    def _dummy_filter_collection(spectrum_in, clone=True):
        return "collection implementation"

    public_filter = collection_filter(
        _dummy_filter_spectrum,
        collection_impl=_dummy_filter_collection,
    )

    result = public_filter(spectrum_in=collection)

    assert result == "collection implementation"
