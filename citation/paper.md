---
title: matchms - processing and similarity evaluation of mass spectrometry data.
tags:
  - Python
  - mass spectrometry
  - metadata cleaning
  - data processing
  - similarity measures

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
With growing use of high throughput techniques, but also through joint effort in the research community, fragmentation mass spectrometry datasets are growing in size and number.
Platforms such as GNPS serve as an open-access hub for sharing of raw, processed, or annotated fragmentation mass spectrometry data (MS/MS) [@Wang2016].
Without suitable tools, however, full quantitative analysis and exploitation of such datasets remains overly challenging.
In particular, large collected datasets contain data of wide range of instruments and measurement conditions, and can further contain a significant fraction of inconsistent, or incorrect metadata (annotations).

Matchms is an open-access Python package to import, process, clean, and compare mass spectrometry data (MS/MS).
It allows to implement and run an easy-to-follow, easy-to-reproduce workflow from raw mass spectra to pre- and post-processed spectral data. 
Raw data can be imported from commonly used MGF files or more convenient-to-handle json files. 
Matchms contains a large number of metadata cleaning and harmonizing filter functions that can easily be stacked to build a desired pipeline, which can also easily be extend by custom functions wherever needed. Available filters include extensive cleaning, correcting, checking of key metadata fields such as compound name, structure annotations (InChI, Smiles, InchiKey), ionmode, adduct, or charge. 

Matchms further provides functions to derive different similarity scores between spectra. Those include the established spectra-based measures of the cosine score or modified cosine score [@watrous_mass_2012], as well as a number of common similarity measures (Dice, Jaccard, Cosine) between molecular fingerprints (rdkit, morgan1, morgan2, morgan3).
Matchms easily facilitates deriving similarity measures between large number of spectra at comparably fast speed due to score implementations using the Numba compiler [@LLVM:CGO04]. Additional similarity measures can easily be added using the matchms API. 
The provided API also allows to quickly compare, sort, and inspect query versus reference spectra using either the included similarity scores or added custom measures.

Matchms is freely accessible either as conda package (https://anaconda.org/nlesc/matchms), or in form of source-code on GitHub (https://github.com/matchms/matchms).
All main functions are covered by tests and continuous integration to offer reliable functionality.
We explicitly value future contributions from a mass spectrometry interested community and hope that matchms can serve as a reliable and accessible entry point to handling complex mass spectrometry dataset using Python. 


# Example workflow
A typical workflow with matchms will look as indicated in \autoref{fig:flowchart}, or as described in the following code example.
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


![Flowchart of matchms workflow. Reference and query spectrums are filtered using the same set of set filters (here: filter A and filter B). Once filtered, every reference spectrum is compared to every query spectrum using the matchms.Scores object. \label{fig:flowchart}](flowchart_matchms.png)


# References
