"""Matchms logger.

Matchms functions and method report unexpected or undesired behavior as
logging WARNING, and additional information as INFO.
The default logging level is set to WARNING. If you want to output additional
logging messages, you can lower the logging level to INFO using setLevel:

.. code-block:: python

    import logging
    from matchms import calculate_scores, Spectrum


    logging.getLogger("matchms").setLevel(logging.INFO)

If you want to suppress logging warnings, you can also raise the logging level
to ERROR by:

.. code-block:: python

    logging.getLogger("matchms").setLevel(logging.ERROR)

"""
import logging
import logging.config
import os
import yaml


here = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(here, 'logging.yml'), 'r', encoding="utf-8") as f:
    config = yaml.safe_load(f.read())
    logging.config.dictConfig(config)

logger = logging.getLogger("matchms")
logger.info('Completed configuring logger()!')
