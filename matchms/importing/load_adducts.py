import os
from functools import lru_cache
from typing import Dict
import yaml


@lru_cache(maxsize=4)
def load_adducts(filename: str = None) -> Dict:
    """Load dictionary of known adducts. Makes sure that file loading is cached.

    Parameters
    ----------
    filename:
        Yaml file containing adducts.
    """
    if filename is None:
        known_adducts_file = os.path.join(os.path.dirname(__file__), "..", "data", "known_adducts.yaml")
    else:
        known_adducts_file = filename

    if os.path.isfile(known_adducts_file):
        with open(known_adducts_file, 'r') as f:
            known_adducts = yaml.safe_load(f)
    else:
        print("Could not find yaml file with known adducts.")
        known_adducts = {'adducts_positive': [],
                         'adducts_negative': []}

    return known_adducts
