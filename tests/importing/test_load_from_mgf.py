import os
from io import StringIO
from pathlib import Path
import pytest
from matchms import Spectrum
from matchms.importing import load_from_mgf


def test_load_from_mgf_using_filepath():
    module_root = os.path.join(os.path.dirname(__file__), "..")
    spectra_file = os.path.join(module_root, "testdata", "pesticides.mgf")

    spectra = list(load_from_mgf(spectra_file))

    assert len(spectra) > 0
    assert isinstance(spectra[0], Spectrum)

    spectra_file_path = Path(module_root).joinpath("testdata", "pesticides.mgf")
    spectra = list(load_from_mgf(spectra_file_path))

    assert len(spectra) > 0
    assert isinstance(spectra[0], Spectrum)


def test_load_missing_mgf_raises():
    with pytest.raises(FileNotFoundError):
        load_from_mgf("does-not-exist.mgf")


def test_load_from_stream():
    spectra = list(load_from_mgf(TEXT_IO))

    assert len(spectra) > 0
    assert isinstance(spectra[0], Spectrum)


TEXT_IO = StringIO(
    """
BEGIN IONS
PEPMASS=183.057
CHARGE=1
MSLEVEL=2
SOURCE_INSTRUMENT= -Q-Exactive Plus Orbitrap Res 70k
FILENAME=Pesticide_Mix6_neg.mzXML
SEQ=*..*
IONMODE=negative
ORGANISM=GNPS-COLLECTIONS-PESTICIDES-NEGATIVE
NAME=Pesticide6_Fuberidazole_C11H8N2O_2-(2-Furyl)-1H-benzimidazole M-H
PI=Dorrestein/Touboul
DATACOLLECTOR=lfnothias
SMILES=C1=CC=C2C(=C1)NC(=N2)C3=CC=CO3
INCHI=InChI=1S/C11H8N2O/c1-2-5-9-8(4-1)12-11(13-9)10-6-3-7-14-10/h1-7H,(H,12,13)
INCHIAUX=N/A
PUBMED=n/a
SUBMITUSER=mwang87
TAGS=
LIBRARYQUALITY=1
SPECTRUMID=CCMSLIB00001058235
SCANS=675
70.786774	213.612045
72.976173	241.782242
73.493057	210.330109
73.515923	211.937332
74.17305	209.306686
75.004799	313.5
77.80809	210.065475
82.050003	216.289841
82.620819	261.580231
82.860512	222.143372
88.244919	209.090591
89.454147	229.945343
94.881096	219.268311
94.917038	10964.588867
94.918587	462.001892
94.993851	322.640625
102.743492	237.246567
106.645241	249.806686
106.876747	269.230896
109.907501	269.221008
112.186974	237.601974
112.985817	2100.576172
113.004906	236.479034
115.000427	1164.228882
115.333664	267.86319
116.995949	5871.84668
117.046104	1942.958862
117.763412	227.911606
122.933907	268.506836
125.587662	259.168152
126.956421	319.40094
133.029556	431.949585
133.420456	231.658493
133.435806	278.857513
136.894592	413.464294
138.90712	4373.975098
138.965378	330.018127
141.046158	365.30246
142.99292	855.966187
142.995346	1030.517456
146.785355	272.870209
154.053802	3683.994141
154.951416	443.910065
154.992584	1055.799194
155.061676	147161.03125
163.001526	2383.76001
170.366013	283.286346
182.987579	1709.164062
183.007996	5187.443359
183.056702	285898.9375
183.121933	448.362885
190.211334	244.807159
202.308212	327.976257
END IONS
    """
)
