# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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

### Fixed
-

### Removed
-


## [0.2.0] - 2020-04-03
### Added
- Anaconda actions


## [0.1.0] - 2020-03-19
### Added
- This is the initial version of Spec2Vec from https://github.com/iomega/Spec2Vec


[Unreleased]: https://github.com/matchms/matchms/compare/v0.3.0...HEAD

[0.3.0]: https://github.com/matchms/matchms/compare/v0.2.0...v0.3.0

[0.2.0]: https://github.com/matchms/matchms/compare/v0.1.0...v0.2.0

[0.1.0]: https://github.com/matchms/matchms/releases/tag/v0.1.0
