Quick start: collection-first workflow
======================================

A typical matchms workflow starts by loading an MS/MS dataset directly as a
``SpectraCollection``:

.. code-block:: python

    from matchms.importing import load_ms2_dataset

    collection = load_ms2_dataset("my_spectra.mgf")

    print(collection)
    print(collection.metadata.head())
    print(collection.n_spectra)

Filters can then be applied directly to the collection:

.. code-block:: python

    from matchms.filtering import (
        harmonize_missing_entries,
        select_by_relative_intensity,
        require_minimum_number_of_peaks,
    )

    collection = harmonize_missing_entries(collection)
    collection = select_by_relative_intensity(
        collection,
        intensity_from=0.01,
        intensity_to=1.0,
    )
    collection = require_minimum_number_of_peaks(collection, n_required=5)

The processed collection can be exported again:

.. code-block:: python

    collection.to_mgf("processed_spectra.mgf")
    collection.to_msp("processed_spectra.msp")
    collection.to_json("processed_spectra.json")

Similarity scores can be computed directly from the processed collection:

.. code-block:: python

    from matchms.similarity import ModifiedCosine

    similarity = ModifiedCosine(tolerance=0.01)
    scores = similarity.matrix(collection)

For repeated searches against a fixed reference library, build the library
index once and reuse it. For large libraries, request only the best hits:

.. code-block:: python

    query_collection = load_ms2_dataset("my_queries.mgf")
    library_index = similarity.build_index(collection)

    top_hits = similarity.search_topk(
        query_collection, library_index, k=10, min_score=0.1, n_jobs=1
    )
    reference_indices, similarities = top_hits.get_hits(0)

Use ``similarity.search(query_collection, library_index)`` when the complete
query-by-reference score matrix is needed.

The index can also be saved and loaded again later:

.. code-block:: python

    similarity.save_index(
        library_index,
        "modified_cosine_index.npz",
    )

    library_index = similarity.load_index(
        "modified_cosine_index.npz"
    )
