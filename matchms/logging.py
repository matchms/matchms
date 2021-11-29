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
