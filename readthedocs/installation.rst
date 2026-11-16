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

To install the matchms 1.0 pre-release (starting with
``1.0.0.dev1``), allow pre-release packages explicitly:

.. code-block:: console

    uv venv --python 3.13
    uv pip install --prerelease allow "matchms>=1.0.0.dev1,<2"

For a regular stable installation, ``uv pip install matchms`` installs the
latest stable release available at the time of installation. It may not yet
contain the matchms 1.0 APIs described here.

Install matchms with conda:

.. code-block:: console

    conda create --name matchms python=3.13
    conda activate matchms
    conda install --channel bioconda --channel conda-forge matchms

Install matchms with pip:

.. code-block:: console

    python -m venv matchms
    source matchms/bin/activate
    pip install matchms
