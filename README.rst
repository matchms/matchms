.. image:: readthedocs/_static/matchms_header.png
   :target: readthedocs/_static/matchms.png
   :align: left
   :alt: matchms

Matchms is an open-source Python package to import, process, clean, and compare mass spectrometry data (MS/MS). It allows to implement and run an easy-to-follow, easy-to-reproduce workflow from raw mass spectra to pre- and post-processed spectral data. Spectral data can be imported from common formats such mzML, mzXML, msp, metabolomics-USI, MGF, or json (e.g. GNPS-syle json files). Matchms then provides filters for metadata cleaning and checking, as well as for basic peak filtering. Finally, matchms was build to import and apply different similarity measures to compare large amounts of spectra. This includes common Cosine scores, but can also easily be extended by custom measures. One example for a spectrum similarity measure that was designed to work in matchms is `Spec2Vec <https://github.com/iomega/spec2vec>`_.

If you use matchms in your research, please cite the following software paper:  

F Huber, S. Verhoeven, C. Meijer, H. Spreeuw, E. M. Villanueva Castilla, C. Geng, J.J.J. van der Hooft, S. Rogers, A. Belloum, F. Diblen, J.H. Spaaks, (2020). matchms - processing and similarity evaluation of mass spectrometry data. Journal of Open Source Software, 5(52), 2411, https://doi.org/10.21105/joss.02411

.. list-table::
   :widths: 25 25
   :header-rows: 1

   * - 
     - Badges
   * - **fair-software.nl recommendations**
     - 
   * - \1. Code repository
     - |GitHub Badge|
   * - \2. License
     - |License Badge|
   * - \3. Community Registry
     - |Conda Badge| |Pypi Badge| |Research Software Directory Badge|
   * - \4. Enable Citation
     - |JOSS Badge| |Zenodo Badge|
   * - \5. Checklists
     - |CII Best Practices Badge| |Howfairis Badge|
   * - **Code quality checks**
     -
   * - Continuous integration
     - |CI Build|
   * - Documentation
     - |ReadTheDocs Badge|
   * - Code Quality
     - |Sonarcloud Quality Gate Badge| |Sonarcloud Coverage Badge|


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

***********************
Documentation for users
***********************
For more extensive documentation `see our readthedocs <https://matchms.readthedocs.io/en/latest/>`_ and our `matchms introduction tutorial <https://blog.esciencecenter.nl/build-your-own-mass-spectrometry-analysis-pipeline-in-python-using-matchms-part-i-d96c718c68ee>`_.

Installation
============

Prerequisites:  

- Python 3.7 or 3.8 
- Anaconda (recommended)

We recommend installing matchms from Anaconda Cloud with

.. code-block:: console

  # install matchms in a new virtual environment to avoid dependency clashes
  conda create --name matchms python=3.8
  conda activate matchms
  conda install --channel bioconda --channel conda-forge matchms

Alternatively, matchms can also be installed using ``pip`` but users will then either have to install ``rdkit`` on their own or won't be able to use the entire functionality. Without ``rdkit`` installed several filter functions related to processing and cleaning chemical metadata will not run.
To install matchms with ``pip`` simply run

.. code-block:: console

  pip install matchms

matchms universe -> additional functionalities
==============================================

Matchms functionalities can be complemented by additional packages.  
To date we are aware of:

+ `Spec2Vec <https://github.com/iomega/spec2vec>`_ an alternative machine-learning spectral similarity score that can simply be installed by `pip install spec2vec` and be imported as `from spec2vec import Spec2Vec` following the same API as the scores in `matchms.similarity`.

+ `MS2DeepScore <https://github.com/matchms/ms2deepscore>`_ a supervised, deep-learning based spectral similarity score that can simply be installed by `pip install ms2deepscore` and be imported as `from ms2deepscore import MS2DeepScore` following the same API as the scores in `matchms.similarity`.

+ `matchmsextras <https://github.com/matchms/matchmsextras>`_ which contains additional functions to create networks based on spectral similarities, to run spectrum searchers against `PubChem`, or additional plotting methods.

*(if you know of any other packages that are fully compatible with matchms, let us know!)*

Introduction
============

To get started with matchms, we recommend following our `matchms introduction tutorial <https://blog.esciencecenter.nl/build-your-own-mass-spectrometry-analysis-pipeline-in-python-using-matchms-part-i-d96c718c68ee>`_.

Alternatively, here below is a small example of using matchms to calculate the Cosine score between mass Spectrums in the `tests/pesticides.mgf <https://github.com/matchms/matchms/blob/master/tests/pesticides.mgf>`_ file.

.. code-block:: python

    from matchms.importing import load_from_mgf
    from matchms.filtering import default_filters
    from matchms.filtering import normalize_intensities
    from matchms import calculate_scores
    from matchms.similarity import CosineGreedy

    # Read spectrums from a MGF formatted file, for other formats see https://matchms.readthedocs.io/en/latest/api/matchms.importing.html 
    file = load_from_mgf("tests/pesticides.mgf")

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

    # Print the calculated scores for each spectrum pair
    for score in scores:
        (reference, query, score) = score
        # Ignore scores between same spectrum and
        # pairs which have less than 20 peaks in common
        if reference is not query and score["matches"] >= 20:
            print(f"Reference scan id: {reference.metadata['scans']}")
            print(f"Query scan id: {query.metadata['scans']}")
            print(f"Score: {score['score']:.4f}")
            print(f"Number of matching peaks: {score['matches']}")
            print("----------------------------")

Glossary of terms
=================

.. list-table::
   :header-rows: 1

   * - Term
     - Description
   * - adduct / addition product
     - During ionization in a mass spectrometer, the molecules of the injected compound break apart
       into fragments. When fragments combine into a new compound, this is known as an addition
       product, or adduct.  `Wikipedia <https://en.wikipedia.org/wiki/Adduct>`__
   * - GNPS
     - Knowledge base for sharing of mass spectrometry data (`link <https://gnps.ucsd.edu/ProteoSAFe/static/gnps-splash.jsp>`__).
   * - InChI / :code:`INCHI`
     - InChI is short for International Chemical Identifier. InChIs are useful
       in retrieving information associated with a certain molecule from a
       database.
   * - InChIKey / InChI key / :code:`INCHIKEY`
     - An identifier for molecules. For example, the InChI key for carbon
       dioxide is :code:`InChIKey=CURLTUGMZLYLDI-UHFFFAOYSA-N` (yes, it
       includes the substring :code:`InChIKey=`).
   * - MGF File / Mascot Generic Format
     - A plan ASCII file format to store peak list data from a mass spectrometry experiment. Links: `matrixscience.com <http://www.matrixscience.com/help/data_file_help.html#GEN>`__,
       `fiehnlab.ucdavis.edu <https://fiehnlab.ucdavis.edu/projects/lipidblast/mgf-files>`__.
   * - parent mass / :code:`parent_mass`
     - Actual mass (in Dalton) of the original compound prior to fragmentation.
       It can be recalculated from the precursor m/z by taking
       into account the charge state and proton/electron masses.
   * - precursor m/z / :code:`precursor_mz`
     - Mass-to-charge ratio of the compound targeted for fragmentation.
   * - SMILES
     - A line notation for describing the structure of chemical species using
       short ASCII strings. For example, water is encoded as :code:`O[H]O`,
       carbon dioxide is encoded as :code:`O=C=O`, etc. SMILES-encoded species may be converted to InChIKey `using a resolver like this one <https://cactus.nci.nih.gov/chemical/structure>`__. The Wikipedia entry for SMILES is `here <https://en.wikipedia.org/wiki/Simplified_molecular-input_line-entry_system>`__.


****************************
Documentation for developers
****************************

Installation
============

To install matchms, do:

.. code-block:: console

  git clone https://github.com/matchms/matchms.git
  cd matchms
  conda env create matchms-dev
  conda activate matchms-dev
  # Install rdkit using conda, rest of dependencies can be installed with pip
  conda install -c conda-forge rdkit
  pip install --editable .[dev]

Run the linter with:

.. code-block:: console

  prospector

Automatically fix incorrectly sorted imports:

.. code-block:: console

  isort --recursive .

Files will be changed in place and need to be committed manually.

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

Copyright (c) 2020, Netherlands eScience Center

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
