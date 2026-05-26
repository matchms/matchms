`fair-software.nl <https://fair-software.nl/>`_ recommendations:

|GitHub Badge|
|License Badge|
|Conda Badge| |Pypi Badge| |Research Software Directory Badge|
|Zenodo Badge|
|CII Best Practices Badge| |Howfairis Badge|

Code quality checks:

|CI First Code Checks| |CI Build|
|ReadTheDocs Badge|
|Sonarcloud Quality Gate Badge| |Sonarcloud Coverage Badge|

.. image:: readthedocs/_static/matchms_header.png
   :target: readthedocs/_static/matchms.png
   :align: left
   :alt: matchms

matchms
=======

Matchms is an open-source Python package for importing, processing, cleaning,
and comparing tandem mass spectrometry data (MS/MS). It supports reproducible
workflows for transforming raw spectra from common file formats into cleaned,
harmonized, and comparable spectral datasets.

Matchms supports popular spectral data formats including mzML, mzXML, MSP, MGF,
metabolomics-USI, and JSON. It provides tools for metadata harmonization,
metadata validation, peak filtering, spectrum processing, and large-scale
spectral similarity calculations.

A central goal of matchms is to support both simple spectrum-wise workflows and
large-scale dataset workflows. For this reason, matchms now provides two
complementary data representations:

- ``Spectrum``: a single mass spectrum with metadata and fragment peaks.
- ``SpectraCollection``: a collection-level representation for complete MS/MS
  datasets, designed for more intuitive dataset inspection, metadata-table
  operations, and scalable collection-level filtering and similarity
  computation.

The traditional ``Spectrum``-centered API remains supported. Existing workflows
that process lists of ``Spectrum`` objects continue to work. New workflows can
use ``SpectraCollection`` to keep spectrum metadata and peak data synchronized
while enabling more efficient collection-level operations.

.. image:: readthedocs/_static/matchms_modular_design_v1_0.png
   :target: readthedocs/_static/matchms_modular_design_v1_0.png
   :align: left
   :alt: matchms code design


Citation
========

If you use matchms in your research, please cite the following software papers:  

F Huber, S. Verhoeven, C. Meijer, H. Spreeuw, E. M. Villanueva Castilla, C. Geng, J.J.J. van der Hooft, S. Rogers, A. Belloum, F. Diblen, J.H. Spaaks, (2020). matchms - processing and similarity evaluation of mass spectrometry data. Journal of Open Source Software, 5(52), 2411, https://doi.org/10.21105/joss.02411

de Jonge NF, Hecht H, Michael Strobel, Mingxun Wang, van der Hooft JJJ, Huber F. (2024). Reproducible MS/MS library cleaning pipeline in matchms. Journal of Cheminformatics, 2024, https://jcheminf.biomedcentral.com/articles/10.1186/s13321-024-00878-1

Core concepts
=============

Spectrum
--------

``Spectrum`` represents one mass spectrum. It contains:

- ``Fragments``: the m/z and intensity arrays of one spectrum.
- ``Metadata``: one spectrum-level metadata dictionary.

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


SpectraCollection
-----------------

``SpectraCollection`` represents many spectra in a synchronized tabular/sparse
layout. It separates dataset-level storage into:

- ``MetadataTable``: a pandas-based table where each row corresponds to one
  spectrum.
- ``FragmentCollection``: a backend for storing all fragment peaks. The default
  backend is ``CSRFragmentCollection``, which stores peaks in a sparse matrix
  with spectra as rows and binned m/z values as columns.

The central invariant is:

.. code-block:: text

    len(collection.metadata) == len(collection.fragments) == collection.n_spectra

Metadata row ``i`` and fragment row ``i`` always describe the same spectrum.

Example:

.. code-block:: python

    from matchms import SpectraCollection
    from matchms.importing import load_from_mgf

    spectra = list(load_from_mgf("my_spectra.mgf"))
    collection = SpectraCollection(spectra)

    print(collection)
    print(collection.metadata.head())
    print(collection.n_spectra)

    first_spectrum = collection[0]

``SpectraCollection`` can be sliced and filtered while preserving alignment
between metadata and fragment data:

.. code-block:: python

    # Select spectra by metadata
    positive = collection.filter(collection.metadata["ionmode"] == "positive")

    # Sort spectra by metadata
    sorted_collection = collection.sort("precursor_mz")

    # Select spectra and restrict m/z range
    selected = collection[:100, 50.0:500.0]


Spectrum and SpectraCollection together
---------------------------------------

``Spectrum`` remains the natural representation for individual spectra and for
spectrum-wise custom logic. ``SpectraCollection`` is the preferred
representation for complete datasets, especially when operations can be applied
to all spectra at once.

Many filters now support both input types:

.. code-block:: python

    from matchms.filtering import harmonize_missing_entries
    from matchms.filtering import select_by_relative_intensity

    # Works on one Spectrum
    spectrum = harmonize_missing_entries(spectrum)
    spectrum = select_by_relative_intensity(spectrum, intensity_from=0.01)

    # Works on a full SpectraCollection
    collection = harmonize_missing_entries(collection)
    collection = select_by_relative_intensity(collection, intensity_from=0.01)

For ``Spectrum`` inputs, filters return either a modified ``Spectrum`` or
``None`` if the spectrum should be removed.

For ``SpectraCollection`` inputs, filters operate on the full collection. Filters
that remove spectra drop the corresponding rows from both ``MetadataTable`` and
``FragmentCollection``. If all spectra are removed, the filter may return
``None``.


Processing workflows
====================

SpectrumProcessor
-----------------

``SpectrumProcessor`` applies filters to spectra one by one. It is useful for
existing workflows that operate on iterables of ``Spectrum`` objects.

.. code-block:: python

    from matchms import SpectrumProcessor
    from matchms.filtering.default_pipelines import BASIC_FILTERS
    from matchms.importing import load_from_mgf

    spectra = list(load_from_mgf("my_spectra.mgf"))

    processor = SpectrumProcessor(BASIC_FILTERS)
    processed_spectra, report = processor.process_spectra(spectra)

This keeps the classic matchms processing style and remains useful when filters
are inherently spectrum-wise.


SpectraCollectionProcessor
--------------------------

``SpectraCollectionProcessor`` applies filters to a complete
``SpectraCollection``. It is intended for workflows where filters can operate on
metadata tables or fragment collections directly.

.. code-block:: python

    from matchms import SpectraCollection
    from matchms import SpectraCollectionProcessor
    from matchms.importing import load_from_mgf

    spectra = list(load_from_mgf("my_spectra.mgf"))
    collection = SpectraCollection(spectra)

    processor = SpectraCollectionProcessor(
        filters=[
            "harmonize_missing_entries",
            ("select_by_relative_intensity", {"intensity_from": 0.01}),
            ("require_minimum_number_of_peaks", {"n_required": 5}),
        ]
    )

    processed_collection = processor.process_collection(collection)

``SpectraCollectionProcessor`` uses the same filter-description style as
``SpectrumProcessor``:

.. code-block:: python

    filters = [
        "harmonize_missing_entries",
        ("select_by_intensity", {"intensity_from": 10.0, "intensity_to": 1000.0}),
        custom_filter_function,
        (custom_filter_with_parameters, {"parameter": "value"}),
    ]

Known matchms filters are ordered according to the matchms filter order. Custom
filters are appended unless a specific position is supplied.


Example: collection-first workflow
==================================

.. code-block:: python

    from matchms import SpectraCollection
    from matchms.importing import load_from_mgf
    from matchms.filtering import (
        harmonize_missing_entries,
        select_by_relative_intensity,
        require_minimum_number_of_peaks,
    )
    from matchms.similarity import FlashCosine

    spectra = list(load_from_mgf("my_spectra.mgf"))

    collection = SpectraCollection(spectra)

    collection = harmonize_missing_entries(collection)
    collection = select_by_relative_intensity(
        collection,
        intensity_from=0.01,
        intensity_to=1.0,
    )
    collection = require_minimum_number_of_peaks(
        collection,
        n_required=5,
    )

    similarity = FlashCosine(matching_mode="hybrid")
    scores = similarity.matrix(collection)

This workflow avoids repeatedly converting a full dataset into lists of
``Spectrum`` objects. Filters with native collection implementations can operate
directly on ``MetadataTable`` or ``FragmentCollection``.


Metadata handling
=================

For individual spectra, metadata is stored in a ``Metadata`` object. Metadata
keys are harmonized to matchms-style names, for example:

.. code-block:: python

    "Precursor MZ" -> "precursor_mz"
    "Compound Name" -> "compound_name"

For collections, metadata is stored in ``MetadataTable``, a pandas-based table.
Column names can be harmonized using the same matchms key conventions.

.. code-block:: python

    collection = collection.harmonize_metadata_columns()

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


Peak data handling
==================

For individual spectra, peak data is stored as ``Fragments``:

.. code-block:: python

    spectrum.peaks.mz
    spectrum.peaks.intensities

For collections, peak data is stored in a ``FragmentCollection`` backend. The
default backend is ``CSRFragmentCollection``, which stores peaks in a sparse
matrix.

This enables efficient operations such as:

- counting peaks per spectrum,
- selecting peaks by intensity,
- selecting peaks by relative intensity,
- filtering spectra by number of peaks,
- slicing m/z ranges,
- computing fragment hashes.

Because the default collection backend uses binned sparse storage, spectra
reconstructed from a ``SpectraCollection`` may contain m/z values corresponding
to bin centers rather than the exact original m/z values. This is important when
testing for exact m/z equality; use numerical tolerances where appropriate.


Similarity scoring
==================

Matchms comes with several similarity measures in ``matchms.similarity``.
Common examples include cosine-based scores, modified cosine scores, neutral
loss scores, and fast approximate methods.

Example using a collection:

.. code-block:: python

    from matchms.similarity import FlashCosine

    similarity = FlashCosine(matching_mode="hybrid")
    scores = similarity.matrix(collection)

Example using individual spectra:

.. code-block:: python

    from matchms.similarity import CosineGreedy

    score = CosineGreedy(tolerance=0.1).pair(spectrum_1, spectrum_2)


Installation
============

Prerequisites:

- Python 3.11 - 3.13
- Anaconda or another virtual environment manager is recommended

Install matchms with conda:

.. code-block:: console

    conda create --name matchms python=3.12
    conda activate matchms
    conda install --channel bioconda --channel conda-forge matchms


Documentation for users
=======================

For more extensive documentation, see:

- `Read the Docs <https://matchms.readthedocs.io/en/latest/>`_
- `matchms introduction tutorial <https://blog.esciencecenter.nl/build-your-own-mass-spectrometry-analysis-pipeline-in-python-using-matchms-part-i-d96c718c68ee>`_
- `user documentation <https://matchms.github.io/matchms-docs/intro.html>`_


matchms ecosystem
=================

Additional packages can complement matchms functionality:

- `Spec2Vec <https://github.com/iomega/spec2vec>`_: machine-learning spectral
  similarity scoring.
- `MS2DeepScore <https://github.com/matchms/ms2deepscore>`_: supervised
  deep-learning-based spectral similarity scoring.
- `matchmsextras <https://github.com/matchms/matchmsextras>`_: additional tools
  for networks, PubChem search, and plotting.
- `MS2Query <https://github.com/iomega/ms2query>`_: MS/MS spectral analogue
  search.
- `memo <https://github.com/mandelbrot-project/memo>`_: retention-time agnostic
  alignment of metabolomics samples.
- `RIAssigner <https://github.com/RECETOX/RIAssigner>`_: retention index
  calculation for GC-MS data.
- `MSMetaEnhancer <https://github.com/RECETOX/MSMetaEnhancer>`_: metadata
  enrichment using web services and computational chemistry packages.
- `SimMS <https://github.com/PangeAI/SimMS>`_: GPU-based implementations of
  common similarity classes.

If you know of another package that is compatible with matchms, let us know.


Ecosystem compatibility
-----------------------

.. compatibility matrix start

.. list-table::
   :header-rows: 1

   * - NumPy Version
     - spec2vec Status
     - ms2deepscore Status
     - ms2query Status
   * - .. image:: https://img.shields.io/badge/numpy-1.25-lightgrey?logo=numpy
          :alt: numpy
     - .. image:: https://img.shields.io/badge/spec2vec-0.9.1-green
     - .. image:: https://img.shields.io/badge/ms2deepscore-2.7.2-green
     - .. image:: https://img.shields.io/badge/ms2query-1.5.4-red
   * - .. image:: https://img.shields.io/badge/numpy-2.1-lightgrey?logo=numpy
          :alt: numpy
     - .. image:: https://img.shields.io/badge/spec2vec-0.9.1-green
     - .. image:: https://img.shields.io/badge/ms2deepscore-2.7.2-green
     - .. image:: https://img.shields.io/badge/ms2query-1.5.4-red

.. compatibility matrix end


Documentation for developers
============================

Development installation
------------------------

.. code-block:: console

    git clone https://github.com/matchms/matchms.git
    cd matchms

    # Create environment using conda
    conda create --name matchms-dev python=3.13
    conda activate matchms-dev

    # Or create environment using uv
    uv venv --python 3.13
    uv sync --group dev

    # Or install with pip
    pip install -r dev-requirements.txt
    pip install --editable .


Code quality
------------

Run the linter and formatter:

.. code-block:: console

    ruff check --fix matchms/YOUR-MODIFIED-FILE.py
    ruff format matchms/YOUR-MODIFIED-FILE.py

Install pre-commit hooks:

.. code-block:: console

    pre-commit install

Run tests:

.. code-block:: console

    pytest


Developer notes: Spectrum vs. SpectraCollection
-----------------------------------------------

When adding new filters, prefer the following structure:

.. code-block:: python

    def _my_filter_spectrum(spectrum_in, ..., clone=True):
        ...

    def _my_filter_collection(collection, ..., clone=True):
        ...

    my_filter = collection_filter(
        _my_filter_spectrum,
        collection_impl=_my_filter_collection,
    )

If no efficient collection implementation exists yet, a filter can initially use
the spectrum-wise fallback. However, filters that operate only on metadata or
only on peak arrays should usually get a native collection implementation.

General recommendations:

- Metadata-only filters should operate on ``MetadataTable``.
- Peak-only filters should operate on ``FragmentCollection``.
- Filters that drop spectra should return ``None`` for failing ``Spectrum``
  inputs and should drop rows for ``SpectraCollection`` inputs.
- Filters should preserve alignment between metadata rows and fragment rows.
- Prefer ``ValueError`` over ``assert`` for user-facing validation.


Conda package
=============

The conda packaging is handled by a `recipe at Bioconda <https://github.com/bioconda/bioconda-recipes/blob/master/recipes/matchms/meta.yaml>`_.

Publishing to PyPI will trigger the creation of a pull request on the Bioconda
recipes repository. Once the pull request is merged, the new version of matchms
will appear on `Anaconda <https://anaconda.org/bioconda/matchms>`_.


Support
=======

To get support, join the public
`Slack channel <https://join.slack.com/t/matchms/shared_invite/zt-2l0t61651-Svv0d5hwl~P5jwV4ZCNFXg>`_.


Contributing
============

If you want to contribute to matchms development, see the
`contribution guidelines <CONTRIBUTING.md>`_.


License
=======

Copyright (c) 2026, Düsseldorf University of Applied Sciences &
Netherlands eScience Center

Licensed under the Apache License, Version 2.0. You may not use this file except
in compliance with the License. You may obtain a copy of the License at:

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed
under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.



.. |GitHub Badge| image:: https://img.shields.io/badge/github-repo-000.svg?logo=github&labelColor=gray&color=blue
   :target: https://github.com/matchms/matchms
   :alt: GitHub Badge

.. |License Badge| image:: https://img.shields.io/github/license/matchms/matchms
   :target: https://github.com/matchms/matchms
   :alt: License Badge

.. |Conda Badge| image:: https://anaconda.org/bioconda/matchms/badges/version.svg
   :target: https://anaconda.org/bioconda/matchms
   :alt: Conda Badge

.. |Pypi Badge| image:: https://img.shields.io/pypi/v/matchms?color=blue
   :target: https://pypi.org/project/matchms/
   :alt: Pypi Badge

.. |Research Software Directory Badge| image:: https://img.shields.io/badge/rsd-matchms-00a3e3.svg
   :target: https://www.research-software.nl/software/matchms
   :alt: Research Software Directory Badge

.. |Zenodo Badge| image:: https://zenodo.org/badge/DOI/10.5281/zenodo.3859772.svg
   :target: https://doi.org/10.5281/zenodo.3859772
   :alt: Zenodo Badge

.. |JOSS Badge| image:: https://joss.theoj.org/papers/10.21105/joss.02411/status.svg
   :target: https://doi.org/10.21105/joss.02411
   :alt: JOSS Badge

.. |CII Best Practices Badge| image:: https://bestpractices.coreinfrastructure.org/projects/3792/badge
   :target: https://bestpractices.coreinfrastructure.org/projects/3792
   :alt: CII Best Practices Badge

.. |Howfairis Badge| image:: https://img.shields.io/badge/fair--software.eu-%E2%97%8F%20%20%E2%97%8F%20%20%E2%97%8F%20%20%E2%97%8F%20%20%E2%97%8F-green
   :target: https://fair-software.eu
   :alt: Howfairis badge

.. |CI First Code Checks| image:: https://github.com/matchms/matchms/actions/workflows/CI_first_code_check.yml/badge.svg
    :alt: Continuous integration workflow
    :target: https://github.com/matchms/matchms/actions/workflows/CI_first_code_check.yml

.. |CI Build| image:: https://github.com/matchms/matchms/actions/workflows/CI_build.yml/badge.svg
    :alt: Continuous integration workflow
    :target: https://github.com/matchms/matchms/actions/workflows/CI_build.yml

.. |ReadTheDocs Badge| image:: https://readthedocs.org/projects/matchms/badge/?version=latest
    :alt: Documentation Status
    :target: https://matchms.readthedocs.io/en/latest/?badge=latest

.. |Sonarcloud Quality Gate Badge| image:: https://sonarcloud.io/api/project_badges/measure?project=matchms_matchms&metric=alert_status
   :target: https://sonarcloud.io/dashboard?id=matchms_matchms
   :alt: Sonarcloud Quality Gate

.. |Sonarcloud Coverage Badge| image:: https://sonarcloud.io/api/project_badges/measure?project=matchms_matchms&metric=coverage
   :target: https://sonarcloud.io/component_measures?id=matchms_matchms&metric=Coverage&view=list
   :alt: Sonarcloud Coverage
