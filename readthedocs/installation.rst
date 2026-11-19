Installation
============

Prerequisites:

- A compatible Python version (check ``requires-python`` in
  ``pyproject.toml`` for the development branch). Python 3.13 is used in
  the examples below.

Install the published matchms release with uv (recommended). Create a
virtual environment before using ``uv pip install``:

.. code-block:: console

    uv venv --python 3.13
    uv pip install matchms

To install the matchms >=1.0 run:

.. code-block:: console

    uv venv --python 3.13
    uv pip install "matchms>=1.0"

Or install matchms with pip:

.. code-block:: console

    python -m venv matchms
    source matchms/bin/activate
    pip install matchms

Matchms can also be installed using conda (from Bioconda), but releases on bioconda may be notably slower,
so the new matchms >= 1.0 might only later become available via conda.

.. code-block:: console

    conda create --name matchms python=3.13
    conda activate matchms
    conda install --channel bioconda --channel conda-forge matchms

