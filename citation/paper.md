---
title: “matchms: Processing and similarity measures of mass spectrometry data."
tags:
  - Python
  - mass spectrometry
  - metadata cleaning
  - data processing
  - similarity evaluation

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
    affiliation: 3
  - name: Justin J. J. van der Hooft
  - orcid: 0000-0002-9340-5511
    affiliation: 2
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
 - name: Wageningen University and Research,...
   index: 2
 - name: University of Glasgow
   index: 3
date: 16 June 2020
bibliography: paper.bib

# Optional fields if submitting to a AAS journal too, see this blog post:
# https://blog.joss.theoj.org/2018/12/a-new-collaboration-with-aas-publishing
aas-doi: 10.3847/xxxxx <- update this with the DOI from AAS once you know it.
aas-journal: Astrophysical Journal <- The name of the AAS journal.
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
Matchms easily facilitates deriving similarity measures between large number of spectra at comparably fast speed due to score implementations using the Numba compiler [@LLVM:CGO04]. Addition similarity measures can easily be added using the matchms API. 
The provided API also allows to quickly compare, sort, and inspect query versus reference spectra using either the included similarity scores or added custom measures.

Matchms is freely accessible either as conda package, or in form of source-code on GitHub.
All main functions are covered by tests and continuous integration to offer reliable functionality.
We explicitly value future contributions from a mass spectrometry interested community and hope that matchms can serve as a reliable and accessible entry point to handling complex mass spectrometry dataset using Python. 


# Figures

Figures can be included like this:
![Caption for example figure.\label{fig:example}](figure.png)
and referenced from text using \autoref{fig:example}.

Fenced code blocks are rendered with syntax highlighting:
```python
for n in range(10):
    yield f(n)
```	


# References