.. matchms documentation master file, created by
   sphinx-quickstart on Tue Apr  7 09:16:44 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to matchms's documentation!
===================================

Python library for fuzzy comparison of mass spectrum data and other Python objects.

.. toctree::
   :maxdepth: 3
   :caption: Contents:

   API <api/matchms.rst>

Example
=======

Below is a small example of using matchms to calculate the Cosine score between mass Spectrums.

.. code-block:: python

    from matchms.importing import load_from_mgf
    from matchms.filtering import default_filters
    from matchms.filtering import normalize_intensities
    from matchms import calculate_scores
    from matchms.similarity import CosineGreedy

    file = load_from_mgf("tests/pesticides.mgf")

    def my_filter(spectrum):
        '''Clean and enhance the spectrums with a filter'''
        spectrum = default_filters(spectrum)
        spectrum = normalize_intensities(spectrum)
        return spectrum

    spectrums = [my_filter(spectrum) for spectrum in file]

    similarity_function = CosineGreedy()
    scores = calculate_scores(spectrums, spectrums, similarity_function)

    for score in scores:
        (reference, query, score, n_matching) = score
        # Ignore scores between same spectrum and
        # pairs which have less than 20 peaks in common
        if reference != query and n_matching >= 20:
            print(f"Reference scan id: {reference.metadata['scans']}")
            print(f"Query scan id: {query.metadata['scans']}")
            print(f"Score: {score:.4f}")
            print(f"Number of matching peaks {n_matching}")
            print("----------------------------")

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
