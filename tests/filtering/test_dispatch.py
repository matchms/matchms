import inspect
import warnings
import numpy as np
import pandas as pd
import pytest
from matchms import Spectrum
from matchms.filtering._dispatch import (
    apply_spectrum_filter_to_collection,
    collection_filter,
    metadata_update_filter,
)
from matchms.SpectraCollection import SpectraCollection
from tests.builder_Spectrum import SpectrumBuilder


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


# --------------------------------------------
# Tests for medata_update_filter
# --------------------------------------------


def test_metadata_update_filter_preserves_public_name_docstring_and_signature():
    def _example_metadata_filter(metadata, mass_tolerance: float = 0.1) -> dict:
        """Example metadata filter docstring."""
        return {}

    public_filter = metadata_update_filter(_example_metadata_filter)

    signature = inspect.signature(public_filter)

    assert public_filter.__name__ == "example_metadata_filter"
    assert public_filter.__doc__ == "Example metadata filter docstring."

    assert list(signature.parameters) == [
        "spectrum_in",
        "mass_tolerance",
        "clone",
    ]
    assert signature.parameters["spectrum_in"].default is inspect.Parameter.empty
    assert signature.parameters["mass_tolerance"].default == 0.1
    assert signature.parameters["clone"].default is True


def test_metadata_update_filter_signature_works_for_spectrum_processor_parameter_check():
    def _example_metadata_filter(metadata, mass_tolerance: float = 0.1) -> dict:
        return {}

    public_filter = metadata_update_filter(_example_metadata_filter)

    signature = inspect.signature(public_filter)
    parameters_without_default = [
        name
        for name, parameter in signature.parameters.items()
        if parameter.default is inspect.Parameter.empty
    ]

    assert parameters_without_default == ["spectrum_in"]


def test_metadata_update_filter_clone_false_updates_spectrum_in_place():
    spectrum = _make_spectrum(metadata={"id": "s1"})

    def _add_metadata(metadata) -> dict:
        return {"processed": True}

    public_filter = metadata_update_filter(_add_metadata)

    result = public_filter(spectrum, clone=False)

    assert result is spectrum
    assert spectrum.get("processed") is True


def test_metadata_update_filter_updates_collection_metadata():
    collection = _make_collection()

    def _add_metadata(metadata) -> dict:
        return {"processed": True}

    public_filter = metadata_update_filter(_add_metadata)

    result = public_filter(collection)

    assert isinstance(result, SpectraCollection)
    assert result is not collection
    assert result.metadata["processed"].tolist() == [True, True, True]
    assert "processed" not in collection.metadata.columns


def test_metadata_update_filter_clone_false_updates_collection_in_place():
    collection = _make_collection()

    def _add_metadata(metadata) -> dict:
        return {"processed": True}

    public_filter = metadata_update_filter(_add_metadata)

    result = public_filter(collection, clone=False)

    assert result is collection
    assert collection.metadata["processed"].tolist() == [True, True, True]


def test_metadata_update_filter_applies_sparse_collection_updates():
    collection = _make_collection()

    def _mark_only_s2(metadata) -> dict:
        if metadata.get("id") == "s2":
            return {"processed": True}
        return {}

    public_filter = metadata_update_filter(_mark_only_s2)

    result = public_filter(collection)

    assert pd.isna(result.metadata.loc[0, "processed"])
    assert result.metadata.loc[1, "processed"] is True
    assert pd.isna(result.metadata.loc[2, "processed"])


def test_metadata_update_filter_forwards_args_and_kwargs_to_metadata_impl():
    spectrum = _make_spectrum(metadata={"id": "s1"})

    def _add_metadata(metadata, value, *, suffix="", clone_marker=False) -> dict:
        return {
            "processed": f"{metadata.get('id')}-{value}{suffix}",
            "clone_marker": clone_marker,
        }

    public_filter = metadata_update_filter(_add_metadata)

    result = public_filter(
        spectrum,
        123,
        suffix="-x",
        clone_marker=True,
    )

    assert result.get("processed") == "s1-123-x"
    assert result.get("clone_marker") is True


def test_metadata_update_filter_returns_none_for_none_input():
    def _add_metadata(metadata) -> dict:
        return {"processed": True}

    public_filter = metadata_update_filter(_add_metadata)

    assert public_filter(None) is None


def test_metadata_update_filter_rejects_metadata_impl_with_args():
    def _bad_metadata_filter(metadata, *args) -> dict:
        return {}

    with pytest.raises(ValueError, match="must not define \\*args or \\*\\*kwargs"):
        metadata_update_filter(_bad_metadata_filter)


def test_metadata_update_filter_rejects_metadata_impl_with_kwargs():
    def _bad_metadata_filter(metadata, **kwargs) -> dict:
        return {}

    with pytest.raises(ValueError, match="must not define \\*args or \\*\\*kwargs"):
        metadata_update_filter(_bad_metadata_filter)


def test_metadata_update_filter_rejects_metadata_impl_with_clone_parameter():
    def _bad_metadata_filter(metadata, clone=True) -> dict:
        return {}

    with pytest.raises(ValueError, match="must not define a 'clone' parameter"):
        metadata_update_filter(_bad_metadata_filter)


def test_metadata_update_filter_uses_custom_collection_impl():
    collection = _make_collection()

    def _metadata_impl(metadata) -> dict:
        raise AssertionError("Default metadata collection path should not be used.")

    def _collection_impl(spectrum_in, clone=True):
        return "custom collection result"

    public_filter = metadata_update_filter(
        _metadata_impl,
        collection_impl=_collection_impl,
    )

    assert public_filter(collection) == "custom collection result"


def test_metadata_update_filter_can_write_none_updates_to_collection():
    def _clear_field(metadata):
        return {"value": None}

    filter_function = metadata_update_filter(
        _clear_field,
        drop_missing_updates=False,
    )

    collection = SpectraCollection(
        [
            SpectrumBuilder().with_metadata({"value": "invalid"}).build(),
        ]
    )

    processed = filter_function(collection)

    assert processed.metadata.loc[0, "value"] is None

