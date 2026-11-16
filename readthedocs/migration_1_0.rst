Migrating from matchms 0.x
==========================

Matchms 1.0 emphasizes collection-oriented MS/MS processing. Existing workflows
based on individual :class:`matchms.Spectrum` objects remain supported, but
new dataset-scale code should use :class:`matchms.SpectraCollection`.

Data representation
-------------------

* Use ``matchms.importing.load_ms2_dataset(path)`` to obtain a
  ``SpectraCollection`` instead of eagerly building lists of ``Spectrum`` objects.
* ``collection.metadata`` is a pandas-backed ``MetadataCollection`` and
  ``collection.fragments`` is a ``FragmentCollection`` (currently CSR-backed).
  Metadata rows and fragment rows remain aligned during collection operations.
* The CSR backend discretizes m/z according to ``mz_precision``. Spectrum rows
  reconstructed from collections need not reproduce input m/z values bit for bit.
* ``load_spectra()`` and the individual importers are still available for
  ``Spectrum``-oriented workflows.

Processing and filters
----------------------

* Prefer ``SpectraProcessor.process_collection(collection)`` for collection-wide
  filtering, or ``process_spectrum(spectrum)`` for a single spectrum.
* Built-in public filters typically dispatch on input type. Filters which
  reject a spectrum return ``None`` for ``Spectrum`` inputs but remove rows
  from a ``SpectraCollection``.
* Processing reports expose ``to_dataframe()`` to summarize input, output,
  removed, metadata-changed and fragments-changed counts. Do not assume an
  old ``missing_metadata`` property exists on ``ProcessingReport``.

Similarity and results
----------------------

* Prefer the high-level ``Cosine``, ``ModifiedCosine``, ``Entropy`` or
  ``EntropySearch`` classes for new spectral comparisons.
* Matrix results are ``Scores`` objects containing fields such as
  ``"score"`` and ``"matches"``; use ``scores["score"].to_array()`` when
  you explicitly need a dense NumPy matrix.
* For repeated library searches, call ``build_index`` once, then ``search``.
  ``save_index`` and ``load_index`` allow reuse in a later process.
* For large reference libraries, prefer ``search_topk(..., k=10)`` when only
  the best hits matter. It returns ``TopKResults``, not a ``Scores`` object.
  It reduces the *output* size to ``n_queries * k``; per-worker temporary arrays
  may still scale with reference-library size.
* Similarity preparation defaults and index compatibility settings matter.
  See :doc:`similarity` for details and explicit configuration examples.

Legacy APIs
-----------

The existing ``Pipeline`` workflow remains present, but its source is marked
for deprecation from 1.1.0. Prefer directly combining
``SpectraProcessor`` and similarity methods in new code. Some legacy
specialized filters have also been superseded by collection-aware filters.
Consult the release changelog for the full list of removals and breaking
changes; this page is a practical migration overview, not an exhaustive API diff.
