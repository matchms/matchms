import os
import numpy as np
from matchms import Pipeline
from matchms.importing import load_ms2_dataset, load_from_mgf
from matchms import SpectraCollectionProcessor, SpectrumProcessor
from matchms.filtering.default_pipelines import CLEAN_PEAKS


def test_clean_peaks_workflow_on_collection_and_spectra_list():
    module_root = os.path.join(os.path.dirname(__file__), "..")
    spectra_file = os.path.join(module_root, "tests", "testdata", "pesticides.mgf")

    # load collection
    collection = load_ms2_dataset(spectra_file)

    # load spectra list
    spectra_list = list(load_from_mgf(spectra_file))

    # run processor on collection
    processor = SpectraCollectionProcessor(CLEAN_PEAKS)
    processed_collection = processor.process_collection(collection)

    # run processor on spectra list
    processor = SpectrumProcessor(CLEAN_PEAKS)
    processed_spectra_list = processor.process_spectra(spectra_list)[0]

    assert processed_collection.n_spectra == len(processed_spectra_list)
    num_peaks_in_lst = np.sum([s.fragments.mz.shape[0] for s in processed_spectra_list])
    assert processed_collection.fragments.count().sum() == num_peaks_in_lst