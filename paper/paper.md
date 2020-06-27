---
title: matchms - processing and similarity evaluation of mass spectrometry data.
tags:
  - Python
  - mass spectrometry
  - metadata cleaning
  - data processing
  - similarity measures
  - metabolomics

authors:
  - name: Florian Huber
    orcid: 0000-0002-3535-9406
    affiliation: 1
  - name: Stefan Verhoeven
    orcid: 0000-0002-5821-2060
    affiliation: 1
  - name: Christiaan Meijer
    orcid: 0000-0002-5529-5761
    affiliation: 1
  - name: Hanno Spreeuw
    orcid: 0000-0002-5057-0322
    affiliation: 1
  - name: Cunliang Geng
    orcid: 0000-0002-1409-8358
    affiliation: 1
  - name: Simon Rogers
    orcid: 0000-0003-3578-4477
    affiliation: 2
  - name: Justin J. J. van der Hooft
    orcid: 0000-0002-9340-5511
    affiliation: 3
  - name: Adam Belloum
    orcid: 0000-0001-6306-6937
    affiliation: 1
  - name: Faruk Diblen
    orcid: 0000-0002-0989-929X
    affiliation: 1
  - name: Juriaan H. Spaaks
    orcid: 0000-0002-7064-4069
    affiliation: 1

affiliations:
 - name: Netherlands eScience Center, Science Park 140, 1098XG Amsterdam, The Netherlands
   index: 1
 - name: School of Computing Science, University of Glasgow, Glasgow, United Kingdom
   index: 2
 - name: Bioinformatics Group, Plant Sciences Group, University of Wageningen, Wageningen, the Netherlands
   index: 3
date: 16 June 2020
bibliography: paper.bib

---

# Summary

Mass spectrometry data is at the heart of numerable applications in the biomedical and life sciences.
With growing use of high throughput techniques researchers need to analyse larger and more complex datasets. In particular through joint effort in the research community, fragmentation mass spectrometry datasets are growing in size and number.
Platforms such as MassBank [@horai_massbank_2010], GNPS [@Wang2016] or MetaboLights [@haug_metabolights_2020] serve as an open-access hub for sharing of raw, processed, or annotated fragmentation mass spectrometry data (MS/MS).
Without suitable tools, however, full quantitative analysis and exploitation of such datasets remains overly challenging.
In particular, large collected datasets contain data aquired using different instruments and measurement conditions, and can further contain a significant fraction of inconsistent, wrongly labeled, or incorrect metadata (annotations).

``Matchms`` is an open-access Python package to import, process, clean, and compare mass spectrometry data (MS/MS) (see \autoref{fig:flowchart}).
It allows to implement and run an easy-to-follow, easy-to-reproduce workflow from raw mass spectra to pre- and post-processed spectral data. 
Raw data can be imported from commonly used MGF files (via pyteomics [@levitsky_pyteomics_2019][@goloborodko_pyteomicspython_2013]) or more convenient-to-handle json files. 
``Matchms`` contains a large number of metadata cleaning and harmonizing filter functions that can easily be stacked to construct a desired pipeline (\autoref{fig:filtering}), which can also easily be extended by custom functions wherever needed. Available filters include extensive cleaning, correcting, checking of key metadata fields such as compound name, structure annotations (InChI, Smiles, InchiKey), ionmode, adduct, or charge. 

![Flowchart of ``matchms`` workflow. Reference and query spectrums are filtered using the same set of set filters (here: filter A and filter B). Once filtered, every reference spectrum is compared to every query spectrum using the ``matchms.Scores`` object. \label{fig:flowchart}](flowchart_matchms.png)

``Matchms`` further provides functions to derive different similarity scores between spectra. Those include the established spectra-based measures of the cosine score or modified cosine score [@watrous_mass_2012].
The package also offers fast implementations of common similarity measures (Dice, Jaccard, Cosine) that can be used to compute similarity scores between molecular fingerprints (rdkit, morgan1, morgan2, morgan3, all implemented using rdkit [@rdkit]).
``Matchms`` easily facilitates deriving similarity measures between large number of spectra at comparably fast speed due to score implementations based on Numpy [@van_der_walt_numpy_2011], Scipy [@2020SciPy-NMeth], and Numba [@LLVM:CGO04]. Additional similarity measures can easily be added using the ``matchms`` API. 
The provided API also allows to quickly compare, sort, and inspect query versus reference spectra using either the included similarity scores or added custom measures.
The API was designed to be easily extensible so that users can add their own filters for spectra preocessing, or their own similarity functions for spectral comparisons.
The present set of filters and similarity functions was mostly geared towards smaller molecules and natural compounds, but it could easily be extended by functions specific to larger peptides or proteins.

``Matchms`` is freely accessible either as conda package (https://anaconda.org/nlesc/matchms), or in form of source-code on GitHub (https://github.com/matchms/matchms). For further code examples and documentation see https://matchms.readthedocs.io/en/latest/.
All main functions are covered by tests and continuous integration to offer reliable functionality.
We explicitly value future contributions from a mass spectrometry interested community and hope that "matchms" can serve as a reliable and accessible entry point for handling complex mass spectrometry datasets using Python. 


# Example workflow
A typical workflow with ``matchms`` looks as indicated in \autoref{fig:flowchart}, or as described in the following code example.
```python
from matchms.importing import load_from_mgf
from matchms.filtering import default_filters
from matchms.filtering import normalize_intensities
from matchms import calculate_scores
from matchms.similarity import CosineGreedy

# Read spectrums from a MGF formatted file
file = load_from_mgf("all_your_spectrums.mgf")

# Apply filters to clean and enhance each spectrum
spectrums = []
for spectrum in file:
    spectrum = default_filters(spectrum)
    spectrum = normalize_intensities(spectrum)
    spectrums.append(spectrum)

# Calculate Cosine similarity scores between all spectrums
scores = calculate_scores(references=spectrums,
                          queries=spectrums,
                          similarity_function=CosineGreedy())

# Print the calculated scores for each spectrum pair
for score in scores:
    (reference, query, score, n_matching) = score
    # Ignore scores between same spectrum and
    # pairs which have less than 20 peaks in common
    if reference != query and n_matching >= 20:
        print(f"Reference scan id: {reference.metadata['scans']}")
        print(f"Query scan id: {query.metadata['scans']}")
        print(f"Score: {score:.4f}")
        print(f"Number of matching peaks: {n_matching}")
        print("----------------------------")
```

![``Matchms`` provided a range of filter functions to process spectrum peaks and metadata. Filters can easily be stacked and combined to build a desired pipeline. The API also makes it easy to extend customer pipelines by adding own filter functions. \label{fig:filtering}](filtering_sketch.png)

# Processing spectrum peaks and plotting
``Matchms`` provides numerous filters to process mass spectra peaks. Below a simple example to remove low intensity peaks from a spectrum (\autoref{fig:peak_filtering}).
```python
from matchms.filtering import require_minimum_number_of_peaks
from matchms.filtering import select_by_mz
from matchms.filtering import select_by_relative_intensity

def process_peaks(s):
    s = select_by_mz(s, mz_from=0, mz_to=1000)
    s = select_by_relative_intensity(s, intensity_from=0.001)
    s = require_minimum_number_of_peaks(s, n_required=10)
    return s

# Apply processing steps to spectra (here to a single "spectrum_raw")
spectrum_processed = process_peaks(spectrum_raw)

# Plot raw spectrum (all and zoomed in)
spectrum_raw.plot()
spectrum_raw.plot(intensity_to=0.02)

# Plot processed spectrum (all and zoomed in)
spectrum_processed.plot()
spectrum_processed.plot(intensity_to=0.02)
```

![Example of ``matchms`` peak filtering applied to an actual spectrum using ``select_by_relative_intensity`` to remove peaks of low relative intensity. Spectra are plotted using the provided ``spectrum.plot()`` function. \label{fig:peak_filtering}](peak_filtering.png)


# References
