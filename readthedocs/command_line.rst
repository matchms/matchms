Command-line interface
======================

matchms also provides a CLI for dataset inspection, conversion, filtering,
similarity matrices, library index creation, and spectral library searches.
Run ``matchms --help`` or a subcommand with ``--help`` for all options.

.. code-block:: console

    matchms info --table
    matchms spectra describe my_spectra.mgf
    matchms spectra convert my_spectra.mgf converted_spectra.msp

For an indexed modified-cosine search from the command line:

.. code-block:: console

    matchms similarity build-index library.mgf --method ModifiedCosine -o library.index.npz
    matchms similarity search queries.mgf library.index.npz --method ModifiedCosine --top-k 10 -o hits.tsv

Use the same similarity settings when building and querying an index.
The CLI supports output as tabular text or machine-readable JSON.
