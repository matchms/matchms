Importing and exporting datasets
================================

Importing datasets
------------------

For new workflows, use ``load_ms2_dataset`` to import a complete dataset as a
``SpectraCollection``:

.. code-block:: python

    from matchms.importing import load_ms2_dataset

    collection = load_ms2_dataset("my_spectra.mgf")

The file type is detected automatically from the file extension. Supported
types include ``mzML``, ``mzXML``, ``mgf``, ``msp``, ``json``, and
``pickle``. Parquet datasets can be loaded directly with
``matchms.importing.load_from_parquet`` (requires ``pyarrow``).

If needed, the file type can be specified explicitly:

.. code-block:: python

    collection = load_ms2_dataset("my_file.txt", ftype="mgf")

Metadata harmonization is enabled by default:

.. code-block:: python

    collection = load_ms2_dataset(
        "my_spectra.mgf",
        metadata_harmonization=True,
    )

For lower-level workflows, individual importers remain available:

.. code-block:: python

    from matchms.importing import load_from_mgf

    spectra = list(load_from_mgf("my_spectra.mgf"))

The helper ``load_spectra`` returns spectra as ``Spectrum`` objects or iterables
of ``Spectrum`` objects:

.. code-block:: python

    from matchms.importing import load_spectra

    spectra = list(load_spectra("my_spectra.mgf"))

Exporting datasets
------------------

``SpectraCollection`` objects can be exported directly:

.. code-block:: python

    collection.to_mgf("processed_spectra.mgf")
    collection.to_msp("processed_spectra.msp")
    collection.to_json("processed_spectra.json")

MGF and MSP export support appending to an existing file:

.. code-block:: python

    collection.to_mgf("combined_spectra.mgf", append=True)
    collection.to_msp("combined_spectra.msp", append=True)

The export style can be selected where supported:

.. code-block:: python

    collection.to_mgf("processed_spectra.mgf", export_style="gnps")
    collection.to_msp("processed_spectra.msp", export_style="nist")

Parquet datasets can also be exported using
``matchms.exporting.save_as_parquet``. The Parquet import/export API is
separate from the collection ``to_mgf``/``to_msp``/``to_json`` methods.

The classic ``save_spectra`` wrapper remains available for workflows that work
with individual ``Spectrum`` objects or lists of spectra:

.. code-block:: python

    from matchms.exporting import save_spectra

    save_spectra(spectra, "processed_spectra.mgf")
