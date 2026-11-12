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
