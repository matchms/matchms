import os
import numpy as np
from matchms.importing import load_ms2_dataset, load_from_mgf
from matchms import SpectraProcessor
from matchms.filtering.default_pipelines import BASIC_FILTERS, CLEAN_PEAKS, REPAIR_ANNOTATION


def test_clean_peaks_workflow_on_collection_and_spectra_list():
    module_root = os.path.join(os.path.dirname(__file__), "..")
    spectra_file = os.path.join(module_root, "tests", "testdata", "pesticides.mgf")

    # load collection
    collection = load_ms2_dataset(spectra_file)

    # load spectra list
    spectra_list = list(load_from_mgf(spectra_file))

    # run processor on collection
    processor = SpectraProcessor(
        BASIC_FILTERS + CLEAN_PEAKS + REPAIR_ANNOTATION
        )
    report = processor.create_processing_report()
    processed_collection = processor.process_collection(
        collection,
        processing_report=report,
    )

    # run processor on spectra list
    processed_spectra_list = processor.process_spectra(spectra_list)

    assert processed_collection.n_spectra == len(processed_spectra_list) == 73
    num_peaks_in_lst = np.sum([s.fragments.mz.shape[0] for s in processed_spectra_list])
    assert processed_collection.fragments.count().sum() == num_peaks_in_lst == 3545

    # check report
    df = report.to_dataframe()
    assert df.shape == (25, 5)  # Adjust when filter pipelines change in matchms

    # store cleaned spectra to file
    output_file = os.path.join(module_root, "tests", "testdata", "pesticides_cleaned.mgf")
    processed_collection.to_mgf(output_file)

    # load again and check that the number of spectra and peaks is the same
    reloaded_collection = load_ms2_dataset(output_file)
    assert reloaded_collection.n_spectra == processed_collection.n_spectra == 73


    # run processor on collection again...
    processor = SpectraProcessor(
        REPAIR_ANNOTATION
        )
    report = processor.create_processing_report()
    _ = processor.process_collection(
        reloaded_collection,
    )