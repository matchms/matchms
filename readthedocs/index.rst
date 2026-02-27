.. matchms documentation master file, created by
   sphinx-quickstart on Tue Apr  7 09:16:44 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to matchms's documentation!
===================================

Matchms is an open-access Python package to import, process, clean, and compare mass spectrometry data (MS/MS). It allows to implement and run an easy-to-follow, easy-to-reproduce workflow from raw mass spectra to pre- and post-processed spectral data.

.. toctree::
   :maxdepth: 3
   :caption: Contents:

   API <api/matchms.rst>

Introduction
============

Matchms was designed to easily build custom spectra processing pipelines and to compute spectra similarities (see flowchart). Spectral data can be imported from common formats such mzML, mzXML, msp, metabolomics-USI, MGF, or json (e.g. GNPS-syle json files). Matchms then provides filters for metadata cleaning and checking, as well as for basic peak filtering. Finally, matchms was build to import and apply different similarity measures to compare large amounts of spectra. This includes common Cosine scores, but can also easily be extended by custom measures.

.. image:: _static/flowchart_matchms.png
  :width: 800
  :alt: matchms workflow illustration

Installation
============

Prerequisites:

- Python 3.9 - 3.12
- Anaconda

Install matchms from Anaconda Cloud with

.. code-block:: console

  # install matchms in a new virtual environment to avoid dependency clashes
  conda create --name matchms python=3.12
  conda activate matchms
  conda install --channel bioconda --channel conda-forge matchms

Example
=======

Below is a small example of using matchms to calculate the Cosine score between mass Spectra in the `tests/testdata/pesticides.mgf <https://github.com/matchms/matchms/blob/master/tests/testdata/pesticides.mgf>`_ file.

.. testcode::

    from matchms.importing import load_from_mgf
    from matchms.filtering import default_filters
    from matchms.filtering import normalize_intensities
    from matchms import calculate_scores
    from matchms.similarity import CosineGreedy

    # Read spectra from a MGF formatted file, for other formats see https://matchms.readthedocs.io/en/latest/api/matchms.importing.html
    file = load_from_mgf("../tests/testdata/pesticides.mgf")

    # Apply filters to clean and enhance each spectrum
    spectra = []
    for spectrum in file:
        # Apply default filter to standardize ion mode, correct charge and more.
        # Default filter is fully explained at https://matchms.readthedocs.io/en/latest/api/matchms.filtering.html .
        spectrum = default_filters(spectrum)
        # Scale peak intensities to maximum of 1
        spectrum = normalize_intensities(spectrum)
        spectra.append(spectrum)

    # Calculate Cosine similarity scores between all spectra
    # For other similarity score methods see https://matchms.readthedocs.io/en/latest/api/matchms.similarity.html .
    # Because references and queries are here the same spectra, we can set is_symmetric=True
    scores = calculate_scores(references=spectra,
                              queries=spectra,
                              similarity_function=CosineGreedy(),
                              is_symmetric=True)

    # This computed all-vs-all similarity scores, the array of which can be accessed as scores.scores
    print(f"Size of matrix of computed similarities: {scores.scores.shape}")

    # Matchms allows to get the best matches for any query using scores_by_query
    query = spectra[15]  # just an example
    best_matches = scores.scores_by_query(query, 'CosineGreedy_score', sort=True)

    # Print the calculated scores for each spectrum pair
    for (reference, score) in best_matches[:10]:
        # Ignore scores between same spectrum
        if reference is not query:
            print(f"Reference scan id: {reference.metadata['scans']}")
            print(f"Query scan id: {query.metadata['scans']}")
            print(f"Score: {score[0]:.4f}")
            print(f"Number of matching peaks: {score[1]}")
            print("----------------------------")

Should output

.. testoutput::

    Size of matrix of computed similarities: (76, 76, 2)
    Reference scan id: 613
    Query scan id: 2161
    Score: 0.8646
    Number of matching peaks: 14
    ----------------------------
    Reference scan id: 603
    Query scan id: 2161
    Score: 0.8237
    Number of matching peaks: 14
    ----------------------------
    ...

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
