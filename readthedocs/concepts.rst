Core architecture
=================

.. figure:: /_static/matchms_modular_design_v1_0.png
   :alt: Modular architecture of matchms 1.0
   :width: 100%
   :align: center

   **Modular architecture of matchms 1.0.**
   Overview of the main components and their interactions,
   including spectra storage, filtering, and similarity computations.

Core concepts
-------------

SpectraCollection
~~~~~~~~~~~~~~~~~

``SpectraCollection`` is the central matchms representation for complete MS/MS
datasets. It stores many spectra in a synchronized collection layout and keeps
spectrum-level metadata aligned with fragment peak data.

A collection separates dataset storage into:

- ``MetadataCollection``: a pandas-based table where each row corresponds to one
  spectrum.
- ``FragmentCollection``: a backend for storing all fragment peaks. The default
  backend is ``CSRFragmentCollection``, which stores peaks in a sparse matrix
  with spectra as rows and binned m/z values as columns.

The central invariant is:

.. code-block:: text

    len(collection.metadata) == len(collection.fragments) == collection.n_spectra

Metadata row ``i`` and fragment row ``i`` always describe the same spectrum.
Operations such as slicing, filtering, sorting, dropping, and deduplication
preserve this alignment.

Example:

.. code-block:: python

    from matchms.importing import load_ms2_dataset

    collection = load_ms2_dataset("my_spectra.mgf")

    print(collection)
    print(collection.metadata.head())
    print(collection.describe())

    first_spectrum = collection[0]

``SpectraCollection`` supports dataset-level selection while keeping metadata
and fragment data synchronized:

.. code-block:: python

    # Select spectra by metadata
    positive = collection.filter(collection.metadata["ionmode"] == "positive")

    # Sort spectra by metadata
    sorted_collection = collection.sort("precursor_mz")

    # Select spectra and restrict m/z range
    selected = collection[:100, 50.0:500.0]

    # Drop spectra without peaks
    collection = collection.drop_empty_spectra()

    # Drop duplicate spectra
    collection = collection.drop_duplicates()

Individual rows can still be accessed as regular ``Spectrum`` objects:

.. code-block:: python

    spectrum = collection[0]

    print(spectrum.peaks.mz)
    print(spectrum.get("precursor_mz"))

It is also possible to run a simple for loop: ``for spectrum in collection:``

Spectrum
~~~~~~~~

``Spectrum`` represents one mass spectrum. It contains:

- ``Fragments``: the m/z and intensity arrays of one spectrum.
- ``Metadata``: one spectrum-level metadata dictionary.

``Spectrum`` is useful for individual spectra, custom spectrum-wise algorithms,
and backward-compatible workflows (e.g., for projects partly using matchms < 1.0).

Example:

.. code-block:: python

    import numpy as np
    from matchms import Spectrum

    spectrum = Spectrum(
        mz=np.array([100.0, 150.0, 200.0]),
        intensities=np.array([0.1, 0.5, 1.0]),
        metadata={
            "precursor_mz": 201.1,
            "ionmode": "positive",
            "smiles": "CCCO",
        },
    )

    print(spectrum.peaks.mz)
    print(spectrum.get("precursor_mz"))

Metadata handling
-----------------

For collections, metadata is stored in ``MetadataCollection``, a pandas-based
table. Each row corresponds to one spectrum, and columns correspond to metadata
fields.

.. code-block:: python

    print(collection.metadata.head())
    print(collection.metadata.columns)

Metadata column names can be harmonized using matchms key conventions:

.. code-block:: python

    collection = collection.harmonize_metadata_columns()

For example:

.. code-block:: text

    "Precursor MZ" -> "precursor_mz"
    "Compound Name" -> "compound_name"

Missing metadata values can be harmonized with:

.. code-block:: python

    from matchms.filtering import harmonize_missing_entries

    collection = harmonize_missing_entries(collection)

By default, common aliases for missing values such as ``""``, ``"N/A"``,
``"NA"``, ``"n/a"``, ``"NaN"``, ``"None"``, and ``"no data"`` are interpreted
as missing entries.

Older specialized filters such as ``harmonize_undefined_inchi``,
``harmonize_undefined_inchikey``, and ``harmonize_undefined_smiles`` are kept for
backward compatibility but are deprecated in favor of
``harmonize_missing_entries``.

For individual spectra, metadata is stored in a ``Metadata`` object and can be
accessed with:

.. code-block:: python

    spectrum.get("precursor_mz")
    spectrum.set("compound_name", "example")

Peak data handling
------------------

For collections, peak data is stored in a ``FragmentCollection`` backend. The
default backend is ``CSRFragmentCollection``, which stores peaks in a sparse
matrix.

This enables efficient dataset-level operations such as:

- counting peaks per spectrum,
- selecting peaks by intensity,
- selecting peaks by relative intensity,
- filtering spectra by number of peaks,
- slicing m/z ranges,
- computing fragment hashes,
- computing summary statistics.

Example:

.. code-block:: python

    peak_counts = collection.fragments.count(axis=1)
    intensity_sums = collection.fragments.sum(axis=1)

    selected = collection[:, 50.0:500.0]

Because the default collection backend uses binned sparse storage, spectra
reconstructed from a ``SpectraCollection`` may contain m/z values corresponding
to bin centers rather than the exact original m/z values. This is important when
testing for exact m/z equality; use numerical tolerances where appropriate.

For individual spectra, peak data is available through ``spectrum.peaks``:

.. code-block:: python

    spectrum.peaks.mz
    spectrum.peaks.intensities
