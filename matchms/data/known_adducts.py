import os
import yaml


def load_known_adducts(filename="known_adducts.yaml"):

    known_adducts_file = os.path.join(os.path.dirname(__file__), filename)
    if os.path.isfile(known_adducts_file):
        with open(known_adducts_file, 'r') as f:
            known_adducts = yaml.safe_load(f)
    else:
        print("Could not find yaml file with known adducts.")
        known_adducts = {'adducts_positive': [],
                         'adducts_negative': []}

    return known_adducts
