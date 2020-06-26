from typing import Generator
import numpy
from ..Spectrum import Spectrum

def parse_msp_file(filename: str):
    molecules = []
    params = {}
    masses = []
    intensities = []
    peakNo = None
    peaksCount = 0

    with open(filename, 'r') as f:
        for line in f:
            rline  = line.rstrip()
            if len(rline) == 0:
                continue
            elif rline.lower().startswith("name:") or rline.lower().startswith("synonym:") or rline.lower().startswith("db#:") or rline.lower().startswith("inchikey:") or rline.lower().startswith("mw:") or rline.lower().startswith("formula:") or rline.lower().startswith("comments:") or rline.lower().startswith("num peaks:") or rline.lower().startswith("precursormz:"):
                splitted_line = rline.split(":", 1)
                params[splitted_line[0].lower()] = splitted_line[1].strip()

                if 'num peak' == splitted_line[0]:
                    peakNo = int(splitted_line[1])

            else:  
                peaksCount += 1
                splitted_line = rline.split(" ")
                
                masses.append(float(splitted_line[0]))
                intensities.append(float(splitted_line[1]))

                if int(params['num peaks']) == peaksCount:
                    peaksCount = 0
                    molecules.append(
                        {
                            'params': params,
                            'm/z array': masses, 
                            'intensity array': intensities
                        }
                    )
                    params = {}
                    masses = []
                    intensities = []

    return molecules
                

def load_from_msp(filename: str) -> Generator[Spectrum, None, None]:
    """Load spectrum(s) from msp file."""

    print(parse_msp_file(filename))
    
        

    

