.. matchms documentation master file, created by
   sphinx-quickstart on Tue Apr  7 09:16:44 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to matchms's documentation!
===================================

Matchms is an open-access Python package to import, process, clean, and compare mass spectrometry data (MS/MS).
It allows to implement and run an easy-to-follow, easy-to-reproduce workflow from raw mass spectra to
pre- and post-processed spectral data.

.. toctree::
   :maxdepth: 3
   :caption: Contents:

   API <api/matchms.rst>

Introduction
============

Matchms allows to easily build custom spectra processing pipelines and to compute spectra similarities.

.. image:: ../_static/flowchart_matchms.png
  :width: 400
  :alt: matchms workflow illustration

Example
=======

Below is a small example of using matchms to calculate the Cosine score between mass Spectrums in the `tests/pesticides.mgf <https://github.com/matchms/matchms/blob/master/tests/pesticides.mgf>`_ file.

.. testcode::

    from matchms.importing import load_from_mgf
    from matchms.filtering import default_filters
    from matchms.filtering import normalize_intensities
    from matchms import calculate_scores
    from matchms.similarity import CosineGreedy

    # Read spectrums from a MGF formatted file, for other formats see https://matchms.readthedocs.io/en/latest/api/matchms.importing.html 
    file = load_from_mgf("../tests/pesticides.mgf")

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
        (reference, query, score, n_matching) = score
        # Ignore scores between same spectrum and
        # pairs which have less than 20 peaks in common
        if reference != query and n_matching >= 20:
            print(f"Reference scan id: {reference.metadata['scans']}")
            print(f"Query scan id: {query.metadata['scans']}")
            print(f"Score: {score:.4f}")
            print(f"Number of matching peaks: {n_matching}")
            print("----------------------------")

Should output

.. testoutput::

    Removed adduct M-H from compound name.
    Added adduct M-H to metadata.
    ...
    Reference scan id: 675
    Query scan id: 2833
    Score: 0.0293
    Number of matching peaks: 20
    ----------------------------
    Reference scan id: 1320
    Query scan id: 2833
    Score: 0.0137
    Number of matching peaks: 24
    ...

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
