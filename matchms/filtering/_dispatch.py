import warnings
from collections.abc import Callable
from functools import wraps
from matchms.SpectraCollection import SpectraCollection


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

    Notes
    -----
    The spectrum-wise fallback always creates a new SpectraCollection. It does
    not mutate the input collection in place, even if ``clone=False`` is passed
    to the public filter.
    """

    @wraps(spectrum_impl)
    def wrapper(obj, *args, **kwargs):
        if obj is None:
            return None

        if isinstance(obj, SpectraCollection):
            if collection_impl is not None:
                return collection_impl(obj, *args, **kwargs)

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
                    obj,
                    spectrum_impl,
                    *args,
                    **kwargs,
                )

            raise NotImplementedError(
                f"{wrapper.__name__} does not support SpectraCollection."
            )

        # Preserve backward compatibility:
        # let the original spectrum-level implementation decide whether the
        # object is acceptable.
        return spectrum_impl(obj, *args, **kwargs)

    wrapper.__name__ = (
        spectrum_impl.__name__
        .removeprefix("_")
        .removesuffix("_spectrum")
    )
    return wrapper


def apply_spectrum_filter_to_collection(
    collection: SpectraCollection,
    spectrum_filter: Callable,
    *args,
    clone: bool = True,
    **kwargs,
) -> SpectraCollection | None:
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

    for spectrum in collection:
        filtered = spectrum_filter(spectrum, *args, clone=False, **kwargs)
        if filtered is not None:
            spectra_out.append(filtered)

    if len(spectra_out) == 0:
        return None

    return collection.__class__(spectra_out, bin_size=collection.bin_size)
