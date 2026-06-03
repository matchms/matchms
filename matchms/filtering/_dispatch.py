import inspect
import warnings
from collections.abc import Callable
from functools import wraps
from tqdm.auto import tqdm
from matchms.filtering.filter_utils.metadata_conversions import (
    apply_metadata_row_filter,
    apply_metadata_updates_to_spectrum,
)
from matchms.typing import SpectraCollectionType


def _get_spectra_collection_type():
    """Import SpectraCollection lazily to avoid circular imports."""
    from matchms.SpectraCollection import SpectraCollection

    return SpectraCollection


def _is_spectra_collection(obj) -> bool:
    """Return True if obj is a SpectraCollection."""
    return isinstance(obj, _get_spectra_collection_type())


# ----------------------------------
# Public filter factories
# 1. General purpose collection_filter for filters with separate spectrum and collection implementations.
# ----------------------------------
def collection_filter(
    spectrum_impl: Callable,
    collection_impl: Callable | None = None,
    *,
    allow_spectrum_fallback: bool = True,
    warn_on_fallback: bool = False,
):
    """Create a public filter supporting Spectrum and SpectraCollection.

    Parameters
    ----------
    spectrum_impl
        Implementation for a single Spectrum.
    collection_impl
        Optional native implementation for SpectraCollection.
    allow_spectrum_fallback
        If True, SpectraCollection inputs without a native implementation are
        processed spectrum-by-spectrum and converted back to SpectraCollection.
    warn_on_fallback
        If True, warn when falling back to spectrum-wise processing.
    """
    @wraps(spectrum_impl)
    def wrapper(spectrum_in=None, *args, **kwargs):
        if spectrum_in is None:
            return None

        # Option (1) --> Handle SpectraCollection
        if _is_spectra_collection(spectrum_in):
            if collection_impl is not None:
                return collection_impl(
                    spectrum_in,
                    *args,
                    **kwargs
                    )

            if allow_spectrum_fallback:
                if warn_on_fallback:
                    warnings.warn(
                        f"{wrapper.__name__} has no native "
                        "SpectraCollection implementation yet. Falling back to "
                        "spectrum-wise processing, which may be slow.",
                        RuntimeWarning,
                        stacklevel=2,
                    )

                return apply_spectrum_filter_to_collection(
                    spectrum_in,
                    spectrum_impl,
                    *args,
                    **kwargs,
                )

            raise NotImplementedError(
                f"{wrapper.__name__} does not support SpectraCollection."
            )
        # Option (2) --> Handle Spectrum
        return spectrum_impl(spectrum_in, *args, **kwargs)

    wrapper.__name__ = (
        spectrum_impl.__name__
        .removeprefix("_")
        .removesuffix("_spectrum")
    )

    return wrapper


# ----------------------------------
# 2. metadata_update_filter for filters that only update metadata and do not drop spectra.
# ----------------------------------

def _metadata_filter_signature(metadata_impl: Callable) -> inspect.Signature:
    """Build the public Spectrum-style signature for a metadata-update filter.

    The metadata implementation has a signature like:

        _filter(metadata, mass_tolerance=0.1)

    The public filter should have a signature like:

        filter(spectrum_in, mass_tolerance=0.1, clone=True)
    """
    metadata_signature = inspect.signature(metadata_impl)
    metadata_parameters = list(metadata_signature.parameters.values())

    if not metadata_parameters:
        raise ValueError(
            f"Metadata filter {metadata_impl.__name__} must accept metadata as "
            "its first parameter."
        )

    public_parameters = [
        inspect.Parameter(
            "spectrum_in",
            kind=inspect.Parameter.POSITIONAL_OR_KEYWORD,
        )
    ]

    # Skip first parameter, which is the internal metadata argument.
    for parameter in metadata_parameters[1:]:
        if parameter.kind in (
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        ):
            raise ValueError(
                f"Metadata filter {metadata_impl.__name__} must not define "
                "*args or **kwargs. Use explicit parameters instead."
            )

        if parameter.name == "clone":
            raise ValueError(
                f"Metadata filter {metadata_impl.__name__} must not define a "
                "'clone' parameter. Cloning is handled by metadata_update_filter."
            )

        public_parameters.append(parameter)

    public_parameters.append(
        inspect.Parameter(
            "clone",
            kind=inspect.Parameter.KEYWORD_ONLY,
            default=True,
            annotation=bool | None,
        )
    )

    return inspect.Signature(
        parameters=public_parameters,
        return_annotation=metadata_signature.return_annotation,
    )


def metadata_update_filter(
    metadata_impl: Callable,
    *,
    collection_impl: Callable | None = None,
    drop_missing_updates: bool = True,
):
    """Create a Spectrum/SpectraCollection filter from row-wise metadata logic.

    ``metadata_impl`` receives one metadata mapping and returns a dict with
    metadata updates. This factory is intended for metadata-only filters that do
    not modify peaks/fragments and do not drop spectra.
    """

    def spectrum_impl(spectrum_in, *args, clone: bool | None = True, **kwargs):
        if spectrum_in is None:
            return None

        spectrum = spectrum_in.clone() if clone else spectrum_in
        updates = metadata_impl(spectrum.metadata, *args, **kwargs)
        return apply_metadata_updates_to_spectrum(spectrum, updates or {})

    def default_collection_impl(
        spectrum_in: SpectraCollectionType,
        *args,
        clone: bool | None = True,
        **kwargs,
    ):
        target = spectrum_in.copy() if clone else spectrum_in

        target.apply_to_metadata_rows(
            apply_metadata_row_filter,
            *args,
            row_filter=metadata_impl,
            inplace=True,
            drop_missing_row_updates=drop_missing_updates,
            drop_missing_updates=drop_missing_updates,
            **kwargs,
        )

        return target

    public_filter = collection_filter(
        spectrum_impl,
        collection_impl=collection_impl or default_collection_impl,
    )

    public_name = metadata_impl.__name__.removeprefix("_")
    public_signature = _metadata_filter_signature(metadata_impl)

    # Set metadata on both functions. The final public_filter is what users and
    # SpectrumProcessor normally see, but setting spectrum_impl as well makes
    # debugging and potential internal checks less surprising.
    spectrum_impl.__name__ = public_name
    spectrum_impl.__doc__ = metadata_impl.__doc__
    spectrum_impl.__signature__ = public_signature

    public_filter.__name__ = public_name
    public_filter.__doc__ = metadata_impl.__doc__
    public_filter.__signature__ = public_signature

    return public_filter


#----------------------------------
# 3. Factory for require-type filters that drop spectra/rows based on metadata requirements.
#----------------------------------
def _metadata_requirement_signature(metadata_impl: Callable) -> inspect.Signature:
    """Build the public Spectrum-style signature for a metadata requirement filter."""
    metadata_signature = inspect.signature(metadata_impl)
    metadata_parameters = list(metadata_signature.parameters.values())

    if not metadata_parameters:
        raise ValueError(
            f"Metadata requirement filter {metadata_impl.__name__} must accept "
            "metadata as its first parameter."
        )

    public_parameters = [
        inspect.Parameter(
            "spectrum_in",
            kind=inspect.Parameter.POSITIONAL_OR_KEYWORD,
        )
    ]

    for parameter in metadata_parameters[1:]:
        if parameter.kind in (
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        ):
            raise ValueError(
                f"Metadata requirement filter {metadata_impl.__name__} must not "
                "define *args or **kwargs. Use explicit parameters instead."
            )

        if parameter.name == "clone":
            raise ValueError(
                f"Metadata requirement filter {metadata_impl.__name__} must not "
                "define a 'clone' parameter. Cloning is handled by the factory."
            )

        public_parameters.append(parameter)


    return inspect.Signature(
        parameters=public_parameters,
        return_annotation=metadata_signature.return_annotation,
    )


def metadata_requirement_filter(metadata_impl: Callable):
    """Create a Spectrum/SpectraCollection requirement filter from metadata logic.

    The wrapped metadata function receives one metadata mapping and returns
    ``True`` if the spectrum/row should be kept and ``False`` if it should be
    removed.

    For Spectrum input, failed requirements return ``None``.
    For SpectraCollection input, failed rows are dropped from metadata and
    fragments.
    """

    def spectrum_impl(spectrum_in, *args, clone: bool | None = True, **kwargs):
        if spectrum_in is None:
            return None

        keep = metadata_impl(spectrum_in.metadata, *args, **kwargs)
        if not keep:
            return None

        return spectrum_in.clone() if clone else spectrum_in

    def collection_impl(collection, *args, clone: bool | None = True, **kwargs):
        target = collection.copy() if clone else collection

        metadata = target.metadata

        keep_mask = metadata.apply(
            lambda row: bool(metadata_impl(row, *args, **kwargs)),
            axis=1,
        ).values

        target.filter(keep_mask, inplace=True)
        return target

    public_filter = collection_filter(
        spectrum_impl,
        collection_impl=collection_impl,
    )

    public_name = metadata_impl.__name__.removeprefix("_")
    public_signature = _metadata_requirement_signature(metadata_impl)

    spectrum_impl.__name__ = public_name
    spectrum_impl.__doc__ = metadata_impl.__doc__
    spectrum_impl.__signature__ = public_signature

    public_filter.__name__ = public_name
    public_filter.__doc__ = metadata_impl.__doc__
    public_filter.__signature__ = public_signature

    return public_filter


def apply_spectrum_filter_to_collection(
    collection: SpectraCollectionType,
    spectrum_filter: Callable,
    *args,
    clone: bool = True,
    progress_bar: bool = False,
    **kwargs,
) -> SpectraCollectionType | None:
    """Apply a spectrum-level filter to all spectra in a collection.

    This is a compatibility fallback for filters without native
    SpectraCollection implementation.

    Notes
    -----
    This fallback always returns a new SpectraCollection. It does not mutate the
    input collection in place. The reconstructed spectra are already fresh
    Spectrum objects, so the wrapped spectrum filter is called with
    ``clone=False`` to avoid an unnecessary second copy.
    """
    if clone is False:
        warnings.warn(
            "Spectrum-wise SpectraCollection fallback does not support true "
            "in-place modification. A new SpectraCollection will be returned.",
            RuntimeWarning,
            stacklevel=2,
        )

    spectra_out = []

    for spectrum in tqdm(collection, disable=not progress_bar):
        filtered = spectrum_filter(spectrum, *args, clone=False, **kwargs)
        if filtered is not None:
            spectra_out.append(filtered)

    if len(spectra_out) == 0:
        return None

    return collection.__class__(spectra_out, bin_size=collection.bin_size)
