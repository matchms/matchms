# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Added function to infer filetype when loading spectra
- CI test runs now include Python 3.10

### Changed

- Support reading old NIST and GOLM MSP formats [#392](https://github.com/matchms/matchms/issues/392)
- expanded options to handle different metadata key styles for (msp) file export [#300](https://github.com/matchms/matchms/issues/300)
- light refactoring of `Metadata` constructor to reduce spectra reading time [#371](https://github.com/matchms/matchms/pull/371/files#)
- two minor corrections of adduct masses (missing electron mass) [#374](https://github.com/matchms/matchms/issues/374)
- Arranged test in folders [#408](https://github.com/matchms/matchms/pull/408)
- Updated datatype of peak_comments returned by load_from_mgf reader [#410](https://github.com/matchms/matchms/pull/410)

### Fixed

- Support sparse score arrays also for FingerprintSimilarity scores [#389](https://github.com/matchms/matchms/issues/389)

## [0.18.0] - 2023-01-05

### Added

- new `Pipeline` class to define entire matchms workflows. This includes importing one or several datasets, processing using matchms filtering/processing functions as well as similartiy computations. Also allows to import/export workflows as yaml files.

### Changed

- major change of `Scores` class. Internally, scores are now stored as a stacked sparse array. This allows to store several different scores for spectrum-spectrums pairs in an efficient way. Also makes it possible to run large-scale comparisons in particular when pipelines start with rapid selective similarity scoring methods such as MetadataMatch or PrecursorMzMatch.
- Scoring/similarity methods now also get a `.sparse_array()` method (next to the previous `.pair()` and `.matrix()` methods).

### Fixed

- minor fix in `interpret_pepmass` function.

## [0.17.0] - 2022-08-23

### Added
- `Scores`: added functionality for writing and reading `Scores` objects to/from disk as JSON and Pickle files [#353](https://github.com/matchms/matchms/pull/353)
- `save_as_msp()` now has a `mode` option (write/append) [#346](https://github.com/matchms/matchms/pull/346)

## [0.16.0] - 2022-06-12

### Added
- `Spectrum` objects now also have `.mz` and `.intensities` properties [#339](https://github.com/matchms/matchms/pull/339)
- `SimilarityNetwork`: similarity-network graphs can now be exported to [cyjs](http://manual.cytoscape.org/en/stable/index.html),
[gexf](http://gexf.net/schema.html), [gml](https://web.archive.org/web/20190207140002/http://www.fim.uni-passau.de/index.php?id=17297&L=1),
and node-link JSON formats [#349](https://github.com/matchms/matchms/pull/349)

### Changed
- metadata filtering: made prefilter check for SMILES and InChI more lenient, eventually resulting in longer runtimes but more accurate checks [#337](https://github.com/matchms/matchms/pull/337)

## [0.15.0] - 2022-03-09

Added neutral losses similarity score (cosine-type score) and a few small fixes.

### Added

- new spectral similarity score: `NeutralLossesCosine` which is based on matches between neutral losses of two spectra [#329](https://github.com/matchms/matchms/pull/329)

### Changed

- added key conversion: "precursor_type" to "adduct" [#332](https://github.com/matchms/matchms/pull/332)
- added key conversion: "rtinseconds" to "retention_time" [#331](https://github.com/matchms/matchms/pull/331)

### Fixed

- handling of duplicate entries in spectrum files (e.g. as field and again in the comments field in msp files) by ugrade of pickydict to 0.4.0 [#332](https://github.com/matchms/matchms/pull/332)

## [0.14.0] - 2022-02-18

This is the first of a few releases to work our way towards matchms 1.0.0, which also means that a few things in the API will likely change. Here the main change is that `Spectrum.metadata` is no longer a simple Python dictionary but became a `Metadata` object. In this context metadata field-names/keys will now be harmonized by default (e.g. "Precursor Mass" will become "precursor_mz). For list of conversions see [matchms key conversion table](https://github.com/matchms/matchms/blob/development/matchms/data/known_key_conversions.csv).

### Added

- new `MetadataMatch`similarity measure in matchms.similarity. This can be used to find matches between metadata entries and currently supports either full string matches or matches of numerical entries within a specified tolerance [#315](https://github.com/matchms/matchms/pull/315)
- metadata is now stored using new `Metadata` class which automatically applied restrictions to used field names/keys to avoid confusion between different format styles [#293](https://github.com/matchms/matchms/pull/293)
- all metadata keys must be lower-case, spaces will be changed to underscores.
- Known key conversions are applied to metadata entries using a [matchms key conversion table](https://github.com/matchms/matchms/blob/development/matchms/data/known_key_conversions.csv)
- new `interpret_pepmass()` filter to handle different pepmass entries found in data [#298][https://github.com/matchms/matchms/issues/298] 

### Changed

- Metadata harmonization will now happen by default! This includes changing field name style and applying known key conversions. To avoid the key conversions user have to make this explicit by setting `metadata_harmonization=False` [#293](https://github.com/matchms/matchms/pull/293)
- `Spikes` class has become `Fragments` class [#293](https://github.com/matchms/matchms/pull/293)
- Change import style (now: isort 5 and slightly different style) [#323](https://github.com/matchms/matchms/pull/323)

### Fixed

- can now handle charges that come as a string of type "2+" or "1-" [#301](https://github.com/matchms/matchms/issues/301)
- new `Metadata`class fixes issue of equality check for different entry orders [#285](https://github.com/matchms/matchms/issues/285)

## [0.13.0] - 2022-02-08

### Added

- Updated and extended plotting functionality, now located in `matchms.plotting`.
Contains three plot types: `plot_spectrum()` or `spectrum.plot()`, `plot_spectra_mirror()` or `spectrum.plot_against()` and `plot_spectra_array()` [#303](https://github.com/matchms/matchms/pull/303)

### Changed

- `Spectrum` objects got an update of the basic spectrum plots `spectrum.plot()` [#303](https://github.com/matchms/matchms/pull/303)
- `require_precursor_mz()` filter will now also discard nonsensical m/z values < 10.0 (value can be adapted by user) [#309](https://github.com/matchms/matchms/pull/309)

### Fixed

- Updated to new url for `load_from_usi` function (old link was broken) [#310](https://github.com/matchms/matchms/pull/310)
- Small bug fix: `add_retention` filters can now properly handle TypeError for empty list. [#314](https://github.com/matchms/matchms/pull/314)

## [0.12.0] - 2022-01-18

### Added

- peak comments (as an `mz: comment` dictionary) are now part of metadata and can be addressed via a `Spectrum()` object `peak_comments` property [#284](https://github.com/matchms/matchms/pull/284)
- peak comments are dynamically updated whenever the respective peaks are changed [#277](https://github.com/matchms/matchms/pull/277)

### Changed

- Major refactoring of unit test layout now using a spectrum builder pattern [#261](https://github.com/matchms/matchms/pull/261)
- Spikes object now has different getitem method that allows to extract specific peaks as mz/intensity pair (or array) [#291](https://github.com/matchms/matchms/pull/291)
- `add_parent_mass()` filter now better handles existing entries (including fields "parent_mass", "exact_mass" and "parentmass") [#292](https://github.com/matchms/matchms/pull/292)
- minor improvement of compound name cleaning in `derive_adduct_from_name()` filter [#280](https://github.com/matchms/matchms/pull/280)
- `save_as_msp()` now writes peak comments (if present) to the output file [#277](https://github.com/matchms/matchms/pull/277)
- `load_from_msp()` now also reads peak comments [#277](https://github.com/matchms/matchms/pull/277)

### Fixed

- able to handle spectra containg empty/zero intensities [#289](https://github.com/matchms/matchms/pull/289)

## [0.11.0] - 2021-12-16

## Added

- better, more flexible string handling of `ModifiedCosine` [#275](https://github.com/matchms/matchms/pull/275)
- matchms logger, replacing all former `print` statments to better control logging output [#271](https://github.com/matchms/matchms/pull/271)
- `add_logging_to_file()`, `set_matchms_logger_level()`, `reset_matchms_logger()` functions to adapt logging output to user needs [#271](https://github.com/matchms/matchms/pull/271)

## Changed

- `save_as_msp()` can now also write to files with other than ".msp" extensions such as ".dat" [#276](https://github.com/matchms/matchms/pull/276)
- refactored `add_precursor_mz`, including better logging [#275](https://github.com/matchms/matchms/pull/275)

## [0.10.0] - 2021-11-21

### Added

- `Spectrum()` objects now also allows generating hashes, e.g. `hash(spectrum)` [#259](https://github.com/matchms/matchms/pull/259)
- `Spectrum()` objects can generate `.spectrum_hash()` and `.metadata_hash()` to track changes to peaks or metadata [#259](https://github.com/matchms/matchms/pull/259)
- `load_from_mgf()` now accepts both a path to a mgf file or a file-like object from a preloaded MGF file [#258](https://github.com/matchms/matchms/pull/258)
- `add_retention` filters with function `add_retention_time()` and `add_retention_index()` [#265](https://github.com/matchms/matchms/pull/265)

### Changed

- Code linting triggered by pylint update [#257](https://github.com/matchms/matchms/pull/257)
- Refactored `add_parent_mass()` filter can now also handle missing charge entries (if ionmode is known) [#252](https://github.com/matchms/matchms/pull/252)

## [0.9.2] - 2021-07-20

### Added

- Support for Python 3.9 [#240](https://github.com/matchms/matchms/issues/240)

### Changed

- Use `bool` instead of `np.bool` [#245](https://github.com/matchms/matchms/pull/245)

## [0.9.1] - 2021-06-16

### Fixed

- Correctly handle charge=0 entries in `add_parent_mass` filter [#236](https://github.com/matchms/matchms/pull/236)
- Reordered written metadata in MSP export for compatability with MS-FINDER & MS-DIAL [#230](https://github.com/matchms/matchms/pull/230)
- Update README.rst to fix fstring-quote python example [#226](https://github.com/matchms/matchms/pull/226)

## [0.9.0] - 2021-05-06

### Added

- new `matchms.networking` module which allows to build and export graphs from `scores` objects [#198](https://github.com/matchms/matchms/pull/198)
- Expand list of known negative ionmode adducts and conversion rules [#213](https://github.com/matchms/matchms/pull/213)
- `.to_numpy` method for Spikes class which allows to run `spectrum.peaks.to_numpy` [#214](https://github.com/matchms/matchms/issues/214)
- `save_as_msp()` function to export spectrums to .msp file [#215](https://github.com/matchms/matchms/pull/215)

### Changed

- `add_precursor_mz()` filter now also checks for metadata in keys `precursormz` and `precursor_mass` [#223](https://github.com/matchms/matchms/pull/223)
- `load_from_msp()` now handles .msp files containing multiple peaks per line separated by `;` [#221](https://github.com/matchms/matchms/pull/221)
- `add_parent_mass()` now includes `overwrite_existing_entry` option (default is False) [#225](https://github.com/matchms/matchms/pull/225)

### Fixed

- `add_parent_mass()` filter now makes consistent use of cleaned adducts [#225](https://github.com/matchms/matchms/pull/225)

## [0.8.2] - 2021-03-08

### Added

- Added filter function 'require_precursor_mz' and added 1 assert function in 'ModifiedCosine' [#191](https://github.com/matchms/matchms/pull/191)

- `make_charge_int()` to convert charge field to integer [#184](https://github.com/matchms/matchms/issues/184)

### Changed

- now deprecated: `make_charge_scalar()`, use `make_charge_int()` instead [#183](https://github.com/matchms/matchms/pull/183)

### Fixed

- Make `load_from_msp` work with different whitespaces [#192](https://github.com/matchms/matchms/issues/192)
- Very minor bugs in `add_parent_mass` [#188](https://github.com/matchms/matchms/pull/188)

## [0.8.1] - 2021-02-19

### Fixed

- Add package data to pypi tar.gz file (to fix Bioconda package) [#179](https://github.com/matchms/matchms/pull/179)

## [0.8.0] - 2021-02-16

### Added

- helper functions to clean adduct strings, `clean_adduct()` [#170](https://github.com/matchms/matchms/pull/170)

### Changed

- more thorough adduct cleaning effecting `derive_adduct_from_name()` and `derive_ionmode()` [#171](https://github.com/matchms/matchms/issues/171)
- significant expansion of `add_parent_mass()` filter to take known adduct properties into account [#170](https://github.com/matchms/matchms/pull/170)

## Fixed

- too unspecific formula detection (and removal) from given compound names in `derive_formula_from_name` [#172](https://github.com/matchms/matchms/issues/172)
- no longer ignore n_max setting in `reduce_to_number_of_peaks` filter [#177](https://github.com/matchms/matchms/issues/177)

## [0.7.0] - 2021-01-04

### Added

- `scores_by_query` and `scores_by reference` now accept sort=True to return sorted scores [#153](https://github.com/matchms/matchms/pull/153)

### Changed

- `Scores.scores` is now returning a structured array [#153](https://github.com/matchms/matchms/pull/153)

### Fixed

- Minor bug in `add_precursor_mz` [#161](https://github.com/matchms/matchms/pull/161)
- Minor bug in `Spectrum` class (missing metadata deepcopy) [#153](https://github.com/matchms/matchms/pull/153)
- Minor bug in `Spectrum` class (__eq__ method was not working with numpy arrays in metadata) [#153](https://github.com/matchms/matchms/pull/153)

## [0.6.2] - 2020-12-03

### Changed

- Considerable performance improvement for CosineGreedy and CosineHungarian [#159](https://github.com/matchms/matchms/pull/159)

## [0.6.1] - 2020-11-26

### Added

- PrecursorMzMatch for deriving precursor m/z matches within a given tolerance [#156](https://github.com/matchms/matchms/pull/156)

### Changed

- Raise error for improper use of reduce_to_number_of_peaks filter [#151](https://github.com/matchms/matchms/pull/151)
- Renamed ParentmassMatch to ParentMassMatch [#156](https://github.com/matchms/matchms/pull/156)

### Fixed

- Fix minor issue with msp importer to avoid failing with unknown characters [#151](https://github.com/matchms/matchms/pull/151)

## [0.6.0] - 2020-09-14

### Added

- Four new peak filtering functions [#119](https://github.com/matchms/matchms/pull/119)
- score_by_reference and score_by_query methods to Scores [#142](https://github.com/matchms/matchms/pull/142)
- is_symmetric option to speed up all-vs-all type score calculation [#59](https://github.com/matchms/matchms/issues/59)
- Support for Python 3.8 [#145](https://github.com/matchms/matchms/pull/145)

### Changed

- Refactor similarity scores to be instances of BaseSimilarity class [#135](https://github.com/matchms/matchms/issues/135)
- Marked Scores.calculate() method as deprecated [#135](https://github.com/matchms/matchms/issues/135)

### Removed

- calculate_parallel function [#135](https://github.com/matchms/matchms/issues/135)
- Scores.calculate_parallel method [#135](https://github.com/matchms/matchms/issues/135)
- similarity.FingerprintSimilarityParallel class (now part of similarity.FingerprintSimilarity) [#135](https://github.com/matchms/matchms/issues/135)
- similarity.ParentmassMatchParallel class (now part of similarity.ParentmassMatch) [#135](https://github.com/matchms/matchms/issues/135)

## [0.5.2] - 2020-08-26

### Changed

- Revision of JOSS manuscript [#137](https://github.com/matchms/matchms/pull/137)

## [0.5.1] - 2020-08-19

### Added

- Basic submodule documentation and more code examples [#128](https://github.com/matchms/matchms/pull/128)

### Changed

- Extended, updated, and corrected documentation for filter functions [#118](https://github.com/matchms/matchms/pull/118)

## [0.5.0] - 2020-08-05

### Added

- Read mzML and mzXML files to create Spectrum objects from it [#110](https://github.com/matchms/matchms/pull/110)
- Read msp files to create Spectrum objects from it [#102](https://github.com/matchms/matchms/pull/102)
- Peak weighting option for CosineGreedy and ModifiedCosine score [#96](https://github.com/matchms/matchms/issues/96)
- Peak weighting option for CosineHungarian score [#112](https://github.com/matchms/matchms/pull/112)
- Similarity score based on comparing parent masses [#79](https://github.com/matchms/matchms/pull/79)
- Method for instantiating a spectrum from the metabolomics USI [#93](https://github.com/matchms/matchms/pull/93)

### Changed

- CosineGreedy function is now numba based [#86](https://github.com/matchms/matchms/pull/86)
- Extended readthedocs documentation [#82](https://github.com/matchms/matchms/issues/82)

### Fixed

- Incorrect denominator for cosine score normalization [#98](https://github.com/matchms/matchms/pull/98)

## [0.4.0] - 2020-06-11

### Added

- Filter add_fingerprint to derive molecular fingerprints [#42](https://github.com/matchms/matchms/issues/42)
- Similarity scores based on molecular fingerprints [#42](https://github.com/matchms/matchms/issues/42)
- Add extensive compound name cleaning and harmonization [#23](https://github.com/matchms/matchms/issues/23)
- Faster cosine score implementation using numba [#29](https://github.com/matchms/matchms/issues/29)
- Cosine score based on Hungarian algorithm [#40](https://github.com/matchms/matchms/pull/40)
- Modified cosine score [#26](https://github.com/matchms/matchms/issues/26)
- Import and export of spectrums from json files [#15](https://github.com/matchms/matchms/issues/15)
- Doc strings for many methods [#49](https://github.com/matchms/matchms/issues/49)
- Examples in doc strings which are tested on CI [#49](https://github.com/matchms/matchms/issues/49)

### Changed

- normalize_intensities filter now also normalizes losses [#69](https://github.com/matchms/matchms/issues/69)

### Removed

## [0.3.4] - 2020-05-29

### Changed

- Fix verify step in conda publish workflow
- Fixed mixed up loss intensity order. [#20](https://github.com/matchms/matchms/issues/20)

## [0.3.3] - 2020-05-27

### Added

- Build workflow runs the tests after installing the package [#47](https://github.com/matchms/matchms/pull/47)

### Changed

- tests were removed from the package (see setup.py) [#47](https://github.com/matchms/matchms/pull/47)

## [0.3.2] - 2020-05-26

### Added

- Workflow improvements
  - Use artifacts in build workflow
  - List artifact folder in build workflow

### Changed

- Workflow improvements [#244](https://github.com/matchms/matchms-backup/pull/244)
  - merge anaconda and python build workflows
  - fix conda package install command in build workflow
  - publish only on ubuntu machine
  - update workflow names
  - test conda packages on windows and unix separately
  - install conda package generated by the workflow
  - split workflows into multiple parts
  - use default settings for conda action
- data folder is handled by setup.py but not meta.yml

### Removed

- remove python build badge [#244](https://github.com/matchms/matchms-backup/pull/244)
- Moved ``spec2vec`` similarity related functionality from ``matchms`` to [iomega/spec2vec](https://github.com/iomega/spec2vec)
- removed build step in build workflow
- removed conda build scripts: conda/build.sh and conda/bld.bat
- removed conda/condarc.yml
- removed conda_build_config.yaml
- removed testing from publish workflow

## [0.3.1] - 2020-05-19

### Added

- improve conda package [#225](https://github.com/matchms/matchms/pull/225)
  - Build scripts for Windows and Unix(MacOS and Linux) systems
  - verify conda package after uploading to anaconda repository by installing it
  - conda package also includes `matchms/data` folder

### Changed

- conda package fixes [#223](https://github.com/matchms/matchms/pull/223)
  - move conda receipe to conda folder
  - fix conda package installation issue
  - add extra import tests for conda package
  - add instructions to build conda package locally
  - automatically find matchms package in setup.py
  - update developer instructions
  - increase verbosity while packaging
  - skip builds for Python 2.X
  - more flexible package versions
  - add deployment requirements to meta.yml
- verify conda package [#225](https://github.com/matchms/matchms/pull/225)
  - use conda/environment.yml when building the package
- split anaconda workflow [#225](https://github.com/matchms/matchms/pull/225)
  - conda build: tests conda packages on every push and pull request
  - conda publish: publish and test conda package on release
  - update the developer instructions
  - move conda receipe to conda folder

## [0.3.0] - 2020-05-13

### Added

- Spectrum, Scores class, save_to_mgf, load_from_mgf, normalize_intensities, calculate_scores [#66](https://github.com/matchms/matchms/pull/66) [#67](https://github.com/matchms/matchms/pull/67) [#103](https://github.com/matchms/matchms/pull/103) [#108](https://github.com/matchms/matchms/pull/108) [#113](https://github.com/matchms/matchms/pull/113) [#115](https://github.com/matchms/matchms/pull/115) [#151](https://github.com/matchms/matchms/pull/151) [#152](https://github.com/matchms/matchms/pull/152) [#121](https://github.com/matchms/matchms/pull/121) [#154](https://github.com/matchms/matchms/pull/154) [#134](https://github.com/matchms/matchms/pull/134) [#159](https://github.com/matchms/matchms/pull/159) [#161](https://github.com/matchms/matchms/pull/161) [#198](https://github.com/matchms/matchms/pull/198)
- Spikes class [#150](https://github.com/matchms/matchms/pull/150) [#167](https://github.com/matchms/matchms/pull/167)
- Anaconda package [#70](https://github.com/matchms/matchms/pull/70) [#68](https://github.com/matchms/matchms/pull/68) [#181](https://github.com/matchms/matchms/pull/181)
- Sonarcloud [#80](https://github.com/matchms/matchms/pull/80) [#79](https://github.com/matchms/matchms/pull/79) [#149](https://github.com/matchms/matchms/pull/149) [#169](https://github.com/matchms/matchms/pull/169)
- Normalization filter [#83](https://github.com/matchms/matchms/pull/83)
- SpeciesString filter [#181](https://github.com/matchms/matchms/pull/181)
- Select by relative intensity filter [#98](https://github.com/matchms/matchms/pull/98)
- Select-by capability based on mz and intensity [#87](https://github.com/matchms/matchms/pull/87)
- Default filters [#97](https://github.com/matchms/matchms/pull/97)
- integration test [#89](https://github.com/matchms/matchms/pull/89) [#147](https://github.com/matchms/matchms/pull/147) [#156](https://github.com/matchms/matchms/pull/156) [#194](https://github.com/matchms/matchms/pull/194)
- cosine greedy similarity function [#112](https://github.com/matchms/matchms/pull/112)
- parent mass filter [#116](https://github.com/matchms/matchms/pull/116) [#122](https://github.com/matchms/matchms/pull/122) [#158](https://github.com/matchms/matchms/pull/158)
- require_minimum_number_of_peaks filter [#131](https://github.com/matchms/matchms/pull/131) [#155](https://github.com/matchms/matchms/pull/155)
- reduce_to_number_of_peaks filter [#209](https://github.com/matchms/matchms/pull/209)
- inchi filters [#145](https://github.com/matchms/matchms/pull/145) [#127](https://github.com/matchms/matchms/pull/127) [#181](https://github.com/matchms/matchms/pull/181)
- losses [#160](https://github.com/matchms/matchms/pull/160)
- vesion string checks [#185](https://github.com/matchms/matchms/pull/185)
- Spec2Vec [#183](https://github.com/matchms/matchms/pull/183) [#165](https://github.com/matchms/matchms/pull/165)
- functions to verify inchies [#181](https://github.com/matchms/matchms/pull/181) [#180](https://github.com/matchms/matchms/pull/180)
- documentation using radthedocs [#196](https://github.com/matchms/matchms/pull/196) [#197](https://github.com/matchms/matchms/pull/197)
- build status badges [#174](https://github.com/matchms/matchms/pull/174)
- vectorize spec2vec [#206](https://github.com/matchms/matchms/pull/206)

### Changed

- Seperate filters [#97](https://github.com/matchms/matchms/pull/97)
- Translate filter steps to new structure (interpret charge and ionmode) [#73](https://github.com/matchms/matchms/pull/73)
- filters returning a new spectrum [#100](https://github.com/matchms/matchms/pull/100)
- Flowchart diagram [#135](https://github.com/matchms/matchms/pull/135)
- numpy usage [#191](https://github.com/matchms/matchms/pull/191)
- consistency of the import statements [#189](https://github.com/matchms/matchms/pull/189)

## [0.2.0] - 2020-04-03

### Added

- Anaconda actions

## [0.1.0] - 2020-03-19

### Added

- This is the initial version of Spec2Vec from https://github.com/iomega/Spec2Vec
- (later splitted into matchms + spec2vec)

[Unreleased]: https://github.com/matchms/matchms/compare/0.18.0...HEAD
[0.18.0]: https://github.com/matchms/matchms/compare/0.17.0...0.18.0
[0.17.0]: https://github.com/matchms/matchms/compare/0.16.0...0.17.0
[0.16.0]: https://github.com/matchms/matchms/compare/0.15.0...0.16.0
[0.15.0]: https://github.com/matchms/matchms/compare/0.14.0...0.15.0
[0.14.0]: https://github.com/matchms/matchms/compare/0.13.0...0.14.0
[0.13.0]: https://github.com/matchms/matchms/compare/0.12.0...0.13.0
[0.12.0]: https://github.com/matchms/matchms/compare/0.11.0...0.12.0
[0.11.0]: https://github.com/matchms/matchms/compare/0.10.0...0.11.0
[0.10.0]: https://github.com/matchms/matchms/compare/0.9.2...0.10.0
[0.9.2]: https://github.com/matchms/matchms/compare/0.9.0...0.9.2
[0.9.1]: https://github.com/matchms/matchms/compare/0.9.0...0.9.1
[0.9.0]: https://github.com/matchms/matchms/compare/0.8.2...0.9.0
[0.8.2]: https://github.com/matchms/matchms/compare/0.8.1...0.8.2
[0.8.1]: https://github.com/matchms/matchms/compare/0.8.0...0.8.1
[0.8.0]: https://github.com/matchms/matchms/compare/0.7.0...0.8.0
[0.7.0]: https://github.com/matchms/matchms/compare/0.6.2...0.7.0
[0.6.2]: https://github.com/matchms/matchms/compare/0.6.1...0.6.2
[0.6.1]: https://github.com/matchms/matchms/compare/0.6.0...0.6.1
[0.6.0]: https://github.com/matchms/matchms/compare/0.5.2...0.6.0
[0.5.2]: https://github.com/matchms/matchms/compare/0.5.1...0.5.2
[0.5.1]: https://github.com/matchms/matchms/compare/0.5.0...0.5.1
[0.5.0]: https://github.com/matchms/matchms/compare/0.4.0...0.5.0
[0.4.0]: https://github.com/matchms/matchms/compare/0.3.4...0.4.0
[0.3.4]: https://github.com/matchms/matchms/compare/0.3.3...0.3.4
[0.3.3]: https://github.com/matchms/matchms/compare/0.3.2...0.3.3
[0.3.2]: https://github.com/matchms/matchms/compare/0.3.1...0.3.2
[0.3.1]: https://github.com/matchms/matchms/compare/0.3.0...0.3.1
[0.3.0]: https://github.com/matchms/matchms/compare/0.2.0...0.3.0
[0.2.0]: https://github.com/matchms/matchms/compare/0.1.0...0.2.0
[0.1.0]: https://github.com/matchms/matchms/releases/tag/0.1.0
