from matchms import calculate_scores
import matchms.filtering as msfilters
import matchms.importing as msimport
import matchms.similarity as mssimilarity


_importing_functions = {"json": msimport.load_from_json,
                        "msp": msimport.load_from_msp}
_filter_functions = {key: f for key, f in msfilters.__dict__.items() if callable(f)}
_score_functions = {key.lower(): f for key, f in mssimilarity.__dict__.items() if callable(f)}


class Pipeline:
    """Central pipeline class.

    """
    def __init__(self):
        """
        """
        self.spectrums_1 = []
        self.spectrums_2 = []
        self.is_symmetric = False
        self.query_data = self.reference_data = None
        self.filter_steps_1 = [["default_filters"]]
        self.filter_steps_2 = self.filter_steps_1
        self.score_computations = []
        self.scores = None

    def run(self):
        self.check_pipeline()
        self.import_data(self.query_data, self.reference_data)
        
        # Processing
        for spectrum in self.spectrums_1:
            for step in self.filter_steps_1:
                if step[0] in _filter_functions:
                    self.apply_filter(spectrum, step)
        if self.is_symmetric is False:
            for spectrum in self.spectrums_2:
                for step in self.filter_steps_2:
                    if step[0] in _filter_functions:
                        self.apply_filter(spectrum, step)        
        # Score computation and masking
        for i, computation in enumerate(self.score_computations):
            if i == 0:
                similarity_function = _score_functions[computation[0]](**computation[1])
                self.scores = calculate_scores(self.spectrums_1,
                                               self.spectrums_2,
                                               similarity_function,
                                               is_symmetric=self.is_symmetric)
            else:
                similarity_func = _score_functions[computation[0]](**computation[1])
                new_scores = similarity_func.sparse_array(references=self.spectrums_1,
                                                          queries=self.spectrums_2,
                                                          idx_row=self.scores.scores.row,
                                                          idx_col=self.scores.scores.col,
                                                          is_symmetric=self.is_symmetric)
                self.scores._scores.add_sparse_data(new_scores, similarity_func.__class__.__name__)

    def check_pipeline(self):
        # check if files exist
        # check if all steps exist
        pass
        
    def import_data(self, query_data, reference_data=None):
        if isinstance(query_data, str):
            query_data = [query_data]
        if isinstance(reference_data, str):
            reference_data = [reference_data]
        spectrums_1 = []
        for query_file in query_data:
           spectrums_1 += _spectrum_importer(query_file)
        self.spectrums_1 += spectrums_1
        if reference_data is None:
            self.is_symmetric = True
            self.spectrums_2 = self.spectrums_1
        else:
            spectrums_2 = []
            for reference_file in reference_data:
               spectrums_2 += _spectrum_importer(reference_file)
            self.spectrums_2 += spectrums_2

    def apply_filter(self, spectrum, filter_step):
        filter_name = filter_step[0]
        if len(filter_step) > 1:
            filter_params = filter_step[1:]
            spectrum = _filter_functions[filter_name](spectrum, **filter_params)
        else: 
            spectrum = _filter_functions[filter_name](spectrum)

    def apply_score_operation(self):
        pass

def _spectrum_importer(filename):
    file_ending = filename.split(".")[-1]
    importer_function = _importing_functions.get(file_ending)
    return list(importer_function(filename))
