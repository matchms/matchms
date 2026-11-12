Documentation for developers
============================

Development installation
------------------------

.. code-block:: console

    git clone https://github.com/matchms/matchms.git
    cd matchms

    # Create the environment from the project's lock file with uv
    uv sync --locked --all-extras

    # Run commands in the uv-managed environment
    uv run pytest

For a pip editable install, create a separate virtual environment and use
``python -m pip install -e .``. Development dependency groups are managed
in ``pyproject.toml`` and ``uv.lock``.


Code quality
------------

Run the linter and formatter:

.. code-block:: console

    uv run ruff check --fix matchms/YOUR-MODIFIED-FILE.py
    uv run ruff format matchms/YOUR-MODIFIED-FILE.py

Install pre-commit hooks:

.. code-block:: console

    uv run pre-commit install

Run tests:

.. code-block:: console

    uv run pytest


Developer notes: collection-first design
----------------------------------------

When adding new functionality, prefer ``SpectraCollection`` support from the
start. New filters should ideally support both ``Spectrum`` and
``SpectraCollection`` inputs.

For filters with separate spectrum and collection implementations, use
``collection_filter``:

.. code-block:: python

    def _my_filter_spectrum(spectrum_in, *, clone=True):
        ...

    def _my_filter_collection(collection, *, clone=True):
        ...

    my_filter = collection_filter(
        _my_filter_spectrum,
        collection_impl=_my_filter_collection,
    )

For metadata-only filters, prefer metadata-level implementations and dispatch
helpers:

.. code-block:: python

    def _my_metadata_filter(metadata):
        ...

    my_filter = metadata_update_filter(_my_metadata_filter)

For requirement filters that keep or remove spectra, use a predicate-style
metadata implementation:

.. code-block:: python

    def _require_my_metadata(metadata) -> bool:
        ...

    require_my_metadata = metadata_requirement_filter(_require_my_metadata)

General recommendations:

- Prefer collection-native implementations for new filters.
- Metadata-only filters should operate on ``MetadataCollection`` or metadata rows.
- Peak-only filters should operate on ``FragmentCollection``.
- Filters that drop spectra should return ``None`` for failing ``Spectrum``
  inputs and should drop rows for ``SpectraCollection`` inputs.
- Filters should preserve alignment between metadata rows and fragment rows.
- Avoid introducing pandas-specific missing values into reconstructed
  ``Spectrum`` metadata.
- Prefer ``ValueError`` over ``assert`` for user-facing validation.
- Add tests for both ``Spectrum`` and ``SpectraCollection`` behavior whenever a
  public filter supports both input types.


Building the documentation
--------------------------

The documentation uses Sphinx 9 and the built-in ``sphinx.ext.apidoc`` extension.
From the repository root, use the locked environment:

.. code-block:: console

    uv lock
    uv sync --locked --group docs --extra all
    cd readthedocs
    uv run --locked --group docs --extra all make html
    uv run --locked --group docs --extra all make doctest
    uv run --locked --group docs --extra all make coverage
    python ../scripts/check_docs_coverage.py _build/coverage/python.txt 10

Run the same builders in CI with ``SPHINXOPTS=-n``. Resolve generated API
warnings in source docstrings rather than suppressing failures wholesale.
The auto-generated ``api/`` directory is created by the Sphinx extension;
keep it out of version control unless your project intentionally vendors it.
