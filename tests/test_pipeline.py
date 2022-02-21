import os
from matchms import Pipeline


module_root = os.path.join(os.path.dirname(__file__), "..")
spectrums_file = os.path.join(module_root, "tests", "MoNA-export-GC-MS-first10.msp")


def test_pipeline():
    pipeline = Pipeline()
    pipeline.query_data = spectrums_file
    pipeline.run()

