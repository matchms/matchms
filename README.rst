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

Matchms is an open-source Python package to import, process, clean, and compare mass spectrometry data (MS/MS). It allows to implement and run an easy-to-follow, easy-to-reproduce workflow from raw mass spectra to pre- and post-processed spectral data. Spectral data can be imported from common formats such mzML, mzXML, msp, metabolomics-USI, MGF, or JSON (e.g. GNPS-syle JSON files). Matchms then provides filters for metadata cleaning and checking, as well as for basic peak filtering. Finally, matchms was built to import and apply different similarity measures to compare large amounts of spectra. This includes common Cosine scores, but can also easily be extended by custom measures. Examples of spectrum similarity measures that were designed to work in matchms are `Spec2Vec <https://github.com/iomega/spec2vec>`_ and `MS2DeepScore <https://github.com/matchms/ms2deepscore>`_.

If you use matchms in your research, please cite the following software paper:  

F Huber, S. Verhoeven, C. Meijer, H. Spreeuw, E. M. Villanueva Castilla, C. Geng, J.J.J. van der Hooft, S. Rogers, A. Belloum, F. Diblen, J.H. Spaaks, (2020). matchms - processing and similarity evaluation of mass spectrometry data. Journal of Open Source Software, 5(52), 2411, https://doi.org/10.21105/joss.02411


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
    :scale: 100%
    :target: https://matchms.readthedocs.io/en/latest/?badge=latest

.. |Sonarcloud Quality Gate Badge| image:: https://sonarcloud.io/api/project_badges/measure?project=matchms_matchms&metric=alert_status
   :target: https://sonarcloud.io/dashboard?id=matchms_matchms
   :alt: Sonarcloud Quality Gate

.. |Sonarcloud Coverage Badge| image:: https://sonarcloud.io/api/project_badges/measure?project=matchms_matchms&metric=coverage
   :target: https://sonarcloud.io/component_measures?id=matchms_matchms&metric=Coverage&view=list
   :alt: Sonarcloud Coverage

**********************************
Latest changes (matchms >= 0.18.0)
**********************************

Pipeline class
==============

To make typical matchms workflows (data import, processing, score computations) more accessible to users, matchms now offers a `Pipeline` class to handle complex workflows. This also allows to create, import, export, or modify workflows using yaml files. See code examples below (and soon: updated tutorial).

Sparse scores array
===================

We realized that many matchms-based workflows aim to compare many-to-many spectra whereby not all pairs and scores are equally important. Often, for instance, it will be about searching similar or related spectra/compounds. This also means that often not all scores need to be stored (or computed). For this reason, we now shifted to a sparse handling of scores in matchms (that means: only storing actually computed, non-null values).

.. image:: readthedocs/_static/matchms_sketch.png
   :target: readthedocs/_static/matchms_sketch.png
   :align: left
   :alt: matchms code design


***********************
Documentation for users
***********************
For more extensive documentation `see our readthedocs <https://matchms.readthedocs.io/en/latest/>`_ and our `matchms introduction tutorial <https://blog.esciencecenter.nl/build-your-own-mass-spectrometry-analysis-pipeline-in-python-using-matchms-part-i-d96c718c68ee>`_.

Installation
============

Prerequisites:  

- Python 3.8 - 3.11, (higher versions should work as well, but are not yet tested systematically)
- Anaconda (recommended)

We recommend installing matchms in a new virtual environment to avoid dependency clashes

.. code-block:: console

  conda create --name matchms python=3.9
  conda activate matchms
  conda install --channel bioconda --channel conda-forge matchms

Alternatively, matchms can also be installed using ``pip``. In the most basic version matchms will then come without ``rdkit`` so that several filter functions related to processing and cleaning chemical metadata will not run. To include ``rdkit`` install matchms as ``matchms[chemistry]``:

.. code-block:: console

  pip install matchms  # simple install w/o rdkit
  pip install matchms[chemistry]  # full install including rdkit

matchms ecosystem -> additional functionalities
===============================================

Matchms functionalities can be complemented by additional packages.  
To date, we are aware of:

+ `Spec2Vec <https://github.com/iomega/spec2vec>`_ an alternative machine-learning spectral similarity score that can simply be installed by `pip install spec2vec` and be imported as `from spec2vec import Spec2Vec` following the same API as the scores in `matchms.similarity`.

+ `MS2DeepScore <https://github.com/matchms/ms2deepscore>`_ a supervised, deep-learning based spectral similarity score that can simply be installed by `pip install ms2deepscore` and be imported as `from ms2deepscore import MS2DeepScore` following the same API as the scores in `matchms.similarity`.

+ `matchmsextras <https://github.com/matchms/matchmsextras>`_ which contains additional functions to create networks based on spectral similarities, to run spectrum searchers against `PubChem`, or additional plotting methods.

+ `MS2Query <https://github.com/iomega/ms2query>`_ Reliable and fast MS/MS spectral-based analogue search, running on top of matchms.

+ `memo <https://github.com/mandelbrot-project/memo>`_ a method allowing a Retention Time (RT) agnostic alignment of metabolomics samples using the fragmentation spectra (MS2) of their constituents.

+ `RIAssigner <https://github.com/RECETOX/RIAssigner>`_ a tool for retention index calculation for gas chromatography - mass spectrometry (GC-MS) data.

+ `MSMetaEnhancer <https://github.com/RECETOX/MSMetaEnhancer>`_ is a python package to collect mass spectral library metadata using various web services and computational chemistry packages.

*(if you know of any other packages that are fully compatible with matchms, let us know!)*

Introduction
============

To get started with matchms, we recommend following our `matchms introduction tutorial <https://blog.esciencecenter.nl/build-your-own-mass-spectrometry-analysis-pipeline-in-python-using-matchms-part-i-d96c718c68ee>`_.

Below is an example of using default filter steps for cleaning spectra, 
followed by calculating the Cosine score between mass Spectrums in the `tests/testdata/pesticides.mgf <https://github.com/matchms/matchms/blob/master/tests/testdata/pesticides.mgf>`_ file.

.. code-block:: python

    from matchms.Pipeline import Pipeline, create_workflow

    workflow = create_workflow(
        yaml_file_name="my_config_file.yaml", # The workflow will be stored in a yaml file, this can be used to rerun your workflow or to share it with others.
        score_computations=[["cosinegreedy", {"tolerance": 1.0}]],
        )
    pipeline = Pipeline(workflow)
    pipeline.logging_file = "my_pipeline.log"  # for pipeline and logging message
    pipeline.run("tests/testdata/pesticides.mgf")
    
Below is a more advanced code example showing how you can make a specific pipeline for your needs.

.. code-block:: python

    import os
    from matchms.Pipeline import Pipeline, create_workflow
    from matchms.filtering.default_pipelines import DEFAULT_FILTERS, LIBRARY_CLEANING
    
    results_folder = "./results"
    os.makedirs(results_folder, exist_ok=True)
    
    workflow = create_workflow(
        yaml_file_name=os.path.join(results_folder, "my_config_file.yaml"),  # The workflow will be stored in a yaml file.
        query_filters=DEFAULT_FILTERS,
        reference_filters=LIBRARY_CLEANING + ["add_fingerprint"],
        score_computations=[["precursormzmatch", {"tolerance": 100.0}],
                            ["cosinegreedy", {"tolerance": 1.0}],
                            ["filter_by_range", {"name": "CosineGreedy_score", "low": 0.2}]],
    )
    pipeline = Pipeline(workflow)
    pipeline.logging_file = os.path.join(results_folder, "my_pipeline.log")  # for pipeline and logging message
    pipeline.logging_level = "WARNING"  # To define the verbosety of the logging
    pipeline.run("tests/testdata/pesticides.mgf", "my_reference_library.mgf",
                 cleaned_query_file=os.path.join(results_folder, "cleaned_query_spectra.mgf"),
                 cleaned_reference_file=os.path.join(results_folder,
                                                     "cleaned_library_spectra.mgf"))  # choose your own files


Alternatively, in particular, if you need more room to add custom functions and steps, the individual steps can run without using the matchms ``Pipeline``:

.. code-block:: python
    
    from matchms.importing import load_from_mgf
    from matchms.filtering import default_filters, normalize_intensities
    from matchms import calculate_scores
    from matchms.similarity import CosineGreedy

    # Read spectrums from a MGF formatted file, for other formats see https://matchms.readthedocs.io/en/latest/api/matchms.importing.html 
    file = load_from_mgf("tests/testdata/pesticides.mgf")

    # Apply filters to clean and enhance each spectrum
    spectrums = []
    for spectrum in file:
        # Apply default filter to standardize ion mode, correct charge and more.
        # Default filter is fully explained at https://matchms.readthedocs.io/en/latest/api/matchms.filtering.html .
        spectrum = default_filters(spectrum)
        # Scale peak intensities to maximum of 1
        spectrum = normalize_intensities(spectrum)
        spectrums.append(spectrum)

    # Calculate Cosine similarity scores between all spectrums
    # For other similarity score methods see https://matchms.readthedocs.io/en/latest/api/matchms.similarity.html .
    scores = calculate_scores(references=spectrums,
                              queries=spectrums,
                              similarity_function=CosineGreedy())

    # Matchms allows to get the best matches for any query using scores_by_query
    query = spectrums[15]  # just an example
    best_matches = scores.scores_by_query(query, 'CosineGreedy_score', sort=True)

    # Print the calculated scores for each spectrum pair
    for (reference, score) in best_matches[:10]:
        # Ignore scores between same spectra
        if reference is not query:
            print(f"Reference scan id: {reference.metadata['scans']}")
            print(f"Query scan id: {query.metadata['scans']}")
            print(f"Score: {score[0]:.4f}")
            print(f"Number of matching peaks: {score[1]}")
            print("----------------------------")


Different spectrum similarity scores
====================================

Matchms comes with numerous different scoring methods in `matchms.similarity` and can further seamlessly work with `Spec2Vec` or `MS2DeepScore`.

Code example: 

.. code-block:: python

    from matchms.importing import load_from_usi
    import matchms.filtering as msfilters
    import matchms.similarity as mssim


    usi1 = "mzspec:GNPS:GNPS-LIBRARY:accession:CCMSLIB00000424840"
    usi2 = "mzspec:MSV000086109:BD5_dil2x_BD5_01_57213:scan:760"

    mz_tolerance = 0.1

    spectrum1 = load_from_usi(usi1)
    spectrum1 = msfilters.select_by_mz(spectrum1, 0, spectrum1.get("precursor_mz"))
    spectrum1 = msfilters.remove_peaks_around_precursor_mz(spectrum1,
                                                           mz_tolerance=0.1)

    spectrum2 = load_from_usi(usi2)
    spectrum2 = msfilters.select_by_mz(spectrum2, 0, spectrum1.get("precursor_mz"))
    spectrum2 = msfilters.remove_peaks_around_precursor_mz(spectrum2,
                                                           mz_tolerance=0.1)
    # Compute scores:
    similarity_cosine = mssim.CosineGreedy(tolerance=mz_tolerance).pair(spectrum1, spectrum2)
    similarity_modified_cosine = mssim.ModifiedCosine(tolerance=mz_tolerance).pair(spectrum1, spectrum2)
    similarity_neutral_losses = mssim.NeutralLossesCosine(tolerance=mz_tolerance).pair(spectrum1, spectrum2)

    print(f"similarity_cosine: {similarity_cosine}")
    print(f"similarity_modified_cosine: {similarity_modified_cosine}")
    print(f"similarity_neutral_losses: {similarity_neutral_losses}")

    spectrum1.plot_against(spectrum2)


****************************
Documentation for developers
****************************

Installation
============

To install matchms, do:

.. code-block:: console

  git clone https://github.com/matchms/matchms.git
  cd matchms
  conda create --name matchms-dev python=3.8
  conda activate matchms-dev
  # Install rdkit using conda, rest of dependencies can be installed with pip
  conda install -c conda-forge rdkit
  python -m pip install --upgrade pip
  pip install --editable .[dev]

Run the linter with:

.. code-block:: console

  prospector

Automatically fix incorrectly sorted imports:

.. code-block:: console

  isort .

Files will be changed in place and need to be committed manually. If you only want to inspect the isort suggestions then simply run:

.. code-block:: console

  isort --check-only --diff .


Run tests (including coverage) with:

.. code-block:: console

  pytest


Conda package
=============

The conda packaging is handled by a `recipe at Bioconda <https://github.com/bioconda/bioconda-recipes/blob/master/recipes/matchms/meta.yaml>`_.

Publishing to PyPI will trigger the creation of a `pull request on the bioconda recipes repository <https://github.com/bioconda/bioconda-recipes/pulls?q=is%3Apr+is%3Aopen+matchms>`_
Once the PR is merged the new version of matchms will appear on `https://anaconda.org/bioconda/matchms <https://anaconda.org/bioconda/matchms>`_

Flowchart
=========

.. figure:: paper/flowchart_matchms.png
  :width: 400
  :alt: Flowchart
  
  Flowchart of matchms workflow. Reference and query spectrums are filtered using the same
  set of set filters (here: filter A and filter B). Once filtered, every reference spectrum is compared to
  every query spectrum using the matchms.Scores object.

Contributing
============

If you want to contribute to the development of matchms,
have a look at the `contribution guidelines <CONTRIBUTING.md>`_.

*******
License
*******

Copyright (c) 2023, Düsseldorf University of Applied Sciences & Netherlands eScience Center

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

*******
Credits
*******

This package was created with `Cookiecutter
<https://github.com/audreyr/cookiecutter>`_ and the `NLeSC/python-template
<https://github.com/NLeSC/python-template>`_.
