Similarity measures
===================

Matchms provides several similarity measures in ``matchms.similarity`` for
comparing mass spectra, spectrum metadata, and molecular structures.

For most peak-based spectral comparisons, start with one of the high-level
classes ``Cosine``, ``ModifiedCosine``, ``Entropy``, or ``EntropySearch``.
These classes provide a common interface for comparing individual spectra,
computing complete similarity matrices, and, where applicable, repeatedly
searching a fixed reference library using a reusable index.

.. list-table:: Similarity measures at a glance
   :header-rows: 1
   :widths: 18 22 42 32

   * - Similarity
     - Recommended class
     - Typical use
     - Specialized implementations
   * - Cosine
     - ``Cosine``
     - Standard peak-based spectral similarity with one-to-one peak matching.
       Supports reusable library indices for repeated searches.
     - ``CosineGreedy``,
       ``CosineHungarian``,
       ``CosineLinear``,
       ``CosineFlash``,
       ``CosineBlink``
   * - Modified cosine
     - ``ModifiedCosine``
     - Cosine similarity allowing both direct fragment matches and matches
       shifted by the difference in precursor m/z. Supports reusable library
       indices for repeated searches.
     - ``ModifiedCosineGreedy``,
       ``ModifiedCosineHungarian``,
       ``ModifiedCosineLinear``;
       ``CosineFlash`` with ``matching_mode="hybrid"``
   * - Spectral entropy
     - ``Entropy``
     - General-purpose spectral entropy similarity with explicit one-to-one
       matching. Supports fragment, neutral-loss, and hybrid matching as well
       as reusable library indices.
     - ``EntropyGreedy``,
       ``EntropyFlash``
   * - Search-optimized spectral entropy
     - ``EntropySearch``
     - High-throughput fragment-only entropy searches against large reference
       libraries. Requires sufficiently separated peaks and can optionally
       merge close peaks during preparation.
     - -
   * - Neutral-loss cosine
     - ``NeutralLossesCosine``
     - Compare spectra based on neutral-loss rather than fragment m/z patterns.
     - -
   * - Binned spectra
     - ``BinnedEmbeddingSimilarity``
     - Compare fixed-width binned spectrum representations using cosine or
       Euclidean similarity.
     - -
   * - Molecular structure
     - ``FingerprintSimilarity``
     - Compare molecular fingerprints derived from structure metadata.
     - -
   * - Metadata
     - ``MetadataMatch``
     - Compare arbitrary metadata fields using exact or tolerance-based matching.
     - -
   * - Precursor or parent mass
     - ``PrecursorMzMatch``,
       ``ParentMassMatch``
     - Simple matching based on precursor m/z or parent mass.
     - -


Default preprocessing for spectral similarities
-----------------------------------------------

The high-level cosine and entropy implementations remove precursor-region
peaks by default (``remove_precursor=True``) and ignore peaks below a relative
intensity threshold (``noise_cutoff=0.01``). These settings are part of
*similarity preparation*; they do not modify the input collection in place.
For reproducible comparisons with older workflows, configure both explicitly:

.. code-block:: python

    from matchms.similarity import ModifiedCosine

    similarity = ModifiedCosine(
        tolerance=0.01,
        remove_precursor=False,
        noise_cutoff=None,
    )

Keep preprocessing settings consistent when constructing and reusing an index.
The exact available options can differ for specialized scoring classes.

Pair and matrix comparisons
---------------------------

Similarity matrices can be computed directly from a ``SpectraCollection``:

.. code-block:: python

    from matchms.similarity import Entropy

    similarity = Entropy(tolerance=0.02)
    scores = similarity.matrix(collection)

The same API can be used to compare two collections:

.. code-block:: python

    from matchms.similarity import ModifiedCosine

    similarity = ModifiedCosine(tolerance=0.01)
    scores = similarity.matrix(references, queries)

Rows of the resulting matrix correspond to the first input collection and
columns to the second.

Pairwise scoring of individual spectra is also supported:

.. code-block:: python

    from matchms.similarity import Cosine

    similarity = Cosine(tolerance=0.01)
    score = similarity.pair(spectrum_1, spectrum_2)


Working with Scores results
---------------------------

``pair()`` returns the score fields for one spectrum pair. ``matrix()`` and
``search()`` return a lightweight ``Scores`` container with fields such as
``"score"`` and (for cosine) ``"matches"``. It stores results but does not
perform score computations itself.

.. code-block:: python

    score_matrix = scores["score"].to_array()
    high_similarity = scores["score"] > 0.7

Some pair-oriented similarity implementations also support
``sparse_matrix()`` for calculations where only selected pairs should be
stored. ``search_topk()`` instead returns a dedicated ``TopKResults`` object
and should not be treated as a sparse representation of the full matrix.


Repeated library searches with reusable indices
-----------------------------------------------

``Cosine``, ``ModifiedCosine``, ``Entropy``, and ``EntropySearch`` support
repeated searches against a fixed reference library.

For this workflow, construct the library index once with ``build_index`` and
then use ``search`` for one or more query collections:

.. code-block:: python

    from matchms.similarity import Cosine

    similarity = Cosine(tolerance=0.01)

    library_index = similarity.build_index(reference_library)

    scores = similarity.search(
        query_collection,
        library_index,
    )

Unlike ``matrix``, ``search`` always treats the input spectra as queries.
The returned score matrix therefore has query spectra as rows and reference
library spectra as columns.

The same workflow is available for modified cosine:

.. code-block:: python

    from matchms.similarity import ModifiedCosine

    similarity = ModifiedCosine(tolerance=0.01)

    library_index = similarity.build_index(reference_library)
    scores = similarity.search(query_collection, library_index)

and for general spectral entropy similarity:

.. code-block:: python

    from matchms.similarity import Entropy

    similarity = Entropy(
        matching_mode="fragment",
        tolerance=0.01,
    )

    library_index = similarity.build_index(reference_library)
    scores = similarity.search(query_collection, library_index)

Building the index separately is especially useful when many query batches are
searched against the same reference library, because the reference spectra do
not need to be prepared and indexed again for every search.


Saving and loading similarity indices
-------------------------------------

A built index can be stored on disk and reused in a later Python session:

.. code-block:: python

    similarity.save_index(
        library_index,
        "library_index.npz",
    )

Load it again with a similarity object using the same scoring and preprocessing
configuration:

.. code-block:: python

    from matchms.similarity import Cosine

    similarity = Cosine(tolerance=0.01)

    library_index = similarity.load_index(
        "library_index.npz"
    )

    scores = similarity.search(
        query_collection,
        library_index,
    )

Index compatibility is checked when the index is loaded or used. An index
should therefore be loaded with a similarity configuration matching the one
used to create it.

The same ``build_index``, ``save_index``, ``load_index``, and ``search``
workflow is available for ``Cosine``, ``ModifiedCosine``, ``Entropy``, and
``EntropySearch``.

For ``Cosine`` and ``ModifiedCosine``, persistent indexed searching uses the
default indexed greedy implementation. It is not available when
``use_hungarian=True``.


Exact top-k library search
--------------------------

For large libraries, a dense ``search()`` result can be expensive even when
only a few high-scoring references are needed. Indexed cosine, modified
cosine, ``Entropy``, and ``EntropySearch`` therefore provide ``search_topk()``
and ``search_prepared_topk()``. These methods return the **exact best k hits**
under the selected similarity scorer, without allocating a dense
query-by-reference score matrix.

.. code-block:: python

    from matchms.similarity import ModifiedCosine

    similarity = ModifiedCosine(tolerance=0.01)
    library_index = similarity.build_index(reference_library)

    hits = similarity.search_topk(
        query_collection,
        library_index,
        k=100,
        min_score=0.1,
        n_jobs=1,
    )

    # Reference positions (zero-based) and similarity values for query 0.
    library_ids, values = hits.get_hits(0)

``TopKResults`` provides ``indices``, ``scores``, and ``valid_counts`` arrays
with shape ``(n_queries, k)`` for the first two. Each query's hits are sorted
by decreasing similarity, breaking equal scores by ascending reference ID.
Only scores *strictly greater* than ``min_score`` are kept (default: ``0``).
If fewer than ``k`` references qualify, unused entries are padded with
reference ID ``-1`` and score ``0``; ``get_hits()`` excludes the padding.
Excluded references must not be interpreted as having a zero similarity.

Cosine and modified-cosine hits can also report accepted peak-match counts:

.. code-block:: python

    library_ids, values, matched_peaks = hits.get_hits_with_matches(0)

These counts are available by default for cosine; specify
``score_fields=("score",)`` if they are not needed. Entropy top-k results do
not contain match counts. Top-k retrieval returns ``TopKResults``, **not** a
truncated ``Scores`` matrix.

The output arrays scale with ``n_queries * k`` instead of
``n_queries * n_references``. Workers still use library-sized scratch arrays;
``n_jobs=-1`` can therefore use substantially more memory than ``n_jobs=1``.
Top-k can also take longer than dense scoring, depending on the workload.
For repeated searches, ``prepare_queries()`` followed by
``search_prepared_topk()`` avoids repeating query preprocessing.

Entropy and EntropySearch
-------------------------

``Entropy`` and ``EntropySearch`` calculate spectral entropy similarity but
make different assumptions about peak matching.

``Entropy`` is the general-purpose implementation. It explicitly resolves
competing one-to-one peak matches and supports ``"fragment"``,
``"neutral_loss"``, and ``"hybrid"`` matching. Use ``Entropy`` when general
matching behavior is required, when tolerance windows may overlap, or when
neutral-loss or hybrid matching is needed.

Example:

.. code-block:: python

    from matchms.similarity import Entropy

    similarity = Entropy(
        matching_mode="hybrid",
        tolerance=0.01,
    )

    scores = similarity.matrix(collection)

``EntropySearch`` is optimized for repeated fragment-only searches against
large reference libraries. It requires peaks within each prepared spectrum to
be sufficiently separated so that candidate matches do not compete for the
same physical peak. This avoids the additional bookkeeping required by the
general ``Entropy`` implementation.

By default, close peaks are merged during preparation:

.. code-block:: python

    from matchms.similarity import EntropySearch

    similarity = EntropySearch(
        tolerance=0.01,
        peak_separation="merge",
    )

    library_index = similarity.build_index(reference_library)

    scores = similarity.search(
        query_collection,
        library_index,
    )

Merging close peaks changes the prepared spectrum and can therefore lead to
scores that differ from ``Entropy``.

If the input spectra are already sufficiently separated, use
``peak_separation="raise"`` to keep the input peak representation unchanged
and raise an error when the separation requirement is violated:

.. code-block:: python

    similarity = EntropySearch(
        tolerance=0.01,
        peak_separation="raise",
    )

``EntropySearch`` currently supports fragment matching with an absolute Da
tolerance. Use ``Entropy`` instead when ppm tolerances, neutral-loss matching,
hybrid matching, or general one-to-one matching semantics are required.

Like the other indexed similarity classes, an ``EntropySearch`` index can be
saved and reused:

.. code-block:: python

    similarity.save_index(
        library_index,
        "entropy_search_index.npz",
    )

    library_index = similarity.load_index(
        "entropy_search_index.npz"
    )


Specialized similarity implementations
--------------------------------------

The specialized classes are useful when a particular scoring implementation is
required.

``CosineGreedy`` provides a simple pair-oriented greedy cosine implementation,
while ``CosineHungarian`` uses optimal peak assignment. ``CosineLinear`` and
``CosineBlink`` provide alternative cosine implementations, and ``CosineFlash``
exposes the indexed Flash-based cosine implementation directly.

For spectral entropy, ``EntropyGreedy`` provides a compact pair-oriented
implementation and ``EntropyFlash`` exposes the general indexed implementation
used by ``Entropy``.

For most applications, the high-level ``Cosine``, ``ModifiedCosine``,
``Entropy``, and ``EntropySearch`` classes are preferred.

Molecular fingerprints are handled through the ``Fingerprints`` class, which
uses the optional ``chemap`` package for fingerprint generation. The old
``add_fingerprint`` filter was removed. Install the optional fingerprint
support with ``pip install 'matchms[fingerprints]'`` when needed.
