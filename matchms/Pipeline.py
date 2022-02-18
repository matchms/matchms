import matchms.ffiltering as msfilters
import matchms.importing as msimport


_importing_functions = {".json": msimport.load_from_json,
                        ".msp": msimport.load_from_msp}


_filter_functions = {key: f for key, f in msfilters.__dict__.items() if callable(f)}


class Pipeline:
    """Central pipeline class.

    """
    def __init__(self, steps = []):
        """
        """
        self.spectrums1 = None
        self.spectrums2 = None
        self.is_symmetric = False

    def run(self):
        self.import_data()
        
        # Processing
        for spectrum in self.spectrums1:
            for step in self.steps:
                if step[0] in _filter_functions:
                    self.apply_filter(self, spectrum, step)
        
        # Score computation and masking
        pass

    def import_data(self, query_data, reference_data=None):
        if isinstance(query_data, str):
            query_data = [query_data]
        if isinstance(reference_data, str):
            reference_data = [reference_data]
        spectrums1 = []
        for query_file in query_data:
           spectrums1.append(_spectrum_importer(query_file))
        self.spectrums1 = spectrums1
        if reference_data is None:
            self.is_symmetric = True
            self.spectrums2 = self.spectrums1
        else:
            spectrums2 = []
            for reference_file in reference_data:
               spectrums2.append(_spectrum_importer(reference_file))
            self.spectrums2 = spectrums2

    def apply_filter(self, spectrum, filter_list):
        for filter_name, filter_params in filter_list:
            spectrum = _filter_functions[filter_name](spectrum, **filter_params)



def _spectrum_importer(filename):
    file_ending = filename.split(".")[1]
    importer_function = _importing_functions.get(file_ending)
    return list(importer_function(filename))
