from matchms import SpectraCollection


def run_filter_as_spectrum_or_collection(filter_function, spectrum_in, as_collection, **kwargs):
    if as_collection:
        collection_out = filter_function(SpectraCollection([spectrum_in]), **kwargs)
        assert isinstance(collection_out, SpectraCollection)
        assert len(collection_out) == 1
        return collection_out[0]

    return filter_function(spectrum_in, **kwargs)
