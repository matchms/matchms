# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.30.1] - 2025-06-13
### Changed
- Fix load_from_msp peak_comments numPy scalar instead of value [#807](https://github.com/matchms/matchms/pull/807)
- Bump requests from 2.32.3 to 2.32.4 [#808](https://github.com/matchms/matchms/pull/808)
- Fix reading spectra where abundance is in scientific notation [#809](https://github.com/matchms/matchms/pull/809)

## [0.30.0] - 2025-05-26
### Added
- support for python 3.13 [#728](https://github.com/matchms/matchms/issues/728) and [#803](https://github.com/matchms/matchms/issues/803)

### Changed
- dropped compatibility with numpy `<2.0`

## [0.29.0] - 2025-05-06
### Added
- Implemented preliminary mzSpecLib export [#757](https://github.com/matchms/matchms/pull/757)
- added BinnedEmbeddingSimilarity and BaseEmbeddingSimilarity Classes [#749](https://github.com/matchms/matchms/pull/749)
- added Fingerprints Class to compute and store inchikey-fingerprint mapping for a list of spectra [#717](https://github.com/matchms/matchms/pull/717)
- some reference spectra were added [#781](https://github.com/matchms/matchms/pull/781)

### Changed
- `compound_name` is now always the first attribute to be written for each spectrum [#762](https://github.com/matchms/matchms/pull/762)
- added option to use different peak separators for msp export [#762](https://github.com/matchms/matchms/pull/762)
- filtering: cloning of Spectra is now optional in filtering and disabled in SpectrumProcessor. 
  Enable cloning for the use of ProcessingReport with `create_report = True` [#754](https://github.com/matchms/matchms/issues/754)
- importing now supports `pathlib.Path` [#738](https://github.com/matchms/matchms/pull/738)
- exporting: empty spectra are saved as empty file instead of not saving at all [#722](https://github.com/matchms/matchms/pull/722)
- exporting: `save_as_mgf` now supports write and append mode [#741](https://github.com/matchms/matchms/pull/741)
- importing: `load_mgf` supports StringIO [#745](https://github.com/matchms/matchms/pull/745)
- importing: fixed bug `load_from_usi` API call [#759](https://github.com/matchms/matchms/pull/759)
- filtering: `normalize_intensities` will now set intensities to `0` instead of `None` when peak intensities are 0 [#750](https://github.com/matchms/matchms/pull/750)
- Support GOLM style MSP files [#763](https://github.com/matchms/matchms/pull/763)
- omit prospector, isort, black in favor of ruff [#790](https://github.com/matchms/matchms/pull/790)

## [0.28.2] - 2024-11-11

### Changed
- Accept both Numpy 1.x (>1.24) and Numpy 2.x to avoid incompatibilities with other packages.

## [0.28.1] - 2024-11-06

### Added
- Increased Test Coverage by @julianpollmann in [#701](https://github.com/matchms/matchms/pull/701)
- Enable metadata exporting with tab separators by @hechth in [#712](https://github.com/matchms/matchms/pull/712)
- add logging for writing spectra to file by @florian-huber in [#645](https://github.com/matchms/matchms/pull/645)

### Changed
- Rename CudaMS -> SimMS, tweak description a bit by @tornikeo in [#703](https://github.com/matchms/matchms/pull/703)
- Update utils.py by @niekdejonge in [#705](https://github.com/matchms/matchms/pull/705)
- Updated matchms dependencies by @hechth in [#709](https://github.com/matchms/matchms/pull/709)
- IndexError in `.matrix` when all scores are 0 by @tornikeo in [#702](https://github.com/matchms/matchms/pull/702)

## [0.27.0] -2024-07-10

### Changed
- Avoid using unstable sorting while sorting collected matching peaks [#636](https://github.com/matchms/matchms/pull/636).
- Losses will no longer be stored as part of a `Spectrum` object, but will be computed on the fly (using `spectrum.losses` or `spectrum.compute_losses(loss_mz_from, loss_mz_to)`)[#681](https://github.com/matchms/matchms/pull/681)
- Jaccard/Tanimoto `@njit`/numba-based similarity functions were replaced by 10-50x faster numpy matrix multiplications [#638](https://github.com/matchms/matchms/pull/638).
- Dependencies were updated to allow newer numpy and numba versions [691](https://github.com/matchms/matchms/pull/691).
- Renamed method names and parameters to align `spectrums` -> `spectra`
- Python support changed from 3.8 - 3.11 to 3.9 to 3.12, and dependency versions were updated [640](https://github.com/matchms/matchms/pull/640).

### Removed
- `add_losses()` filter was removed. Losses will no longer be stored as part of a `Spectrum` object, but will be computed on the fly [#681](https://github.com/matchms/matchms/pull/681).

### Fixed
- Remove empty spectra before exporting to file [#686](https://github.com/matchms/matchms/pull/686).
- Name position in mirror plots [#678](https://github.com/matchms/matchms/pull/678).

## [0.26.4] -2024-06-14

### Added
- Added require_maximum_number_of_peaks as filter
- Added derive_formula_from_smiles as filter

## [0.26.3] -2024-06-07

### Added
- repair_adduct_and_parent_mass_based_on_smiles does not repair parent mass anymore if it is already close to the smiles
- repair_paren_mass_from_smiles was added as a filter

## [0.26.2] -2024-06-03
### Added
- Added require correct ms level

### Changed
- Fixed bug in repair_adduct_and_parent_mass_based_on_smiles if mass from smiles is None

## [0.26.1] -2024-06-03
### Changed
- Fixed bug. Removing spectra in spectrum processor would break the saving, since trying to save None values.

## [0.26.0] -2024-06-03

## Unreleased
### Added
- Added remove_profile_spectra filter
- Allowed peaks to have any floating point dtype
- Added require_matching_ionmode_and_adduct filter
- Added remove_noise_below_frequent_intensities

### Removed:
- Require_precursor_below_mz is deprecated, require_precursor_mz now also allows for argument maximum_mz 


## [0.25.0] -2024-05-21
### Added
- filters `require_formula` and `require_compound_name`. [#627](https://github.com/matchms/matchms/pull/627)
- filters `require_retention_time` and `require_retention_index`. [#585](https://github.com/matchms/matchms/pull/602)

### Changed
- Removed repair_precursor_is_parent_mass
- repair_adduct_based_on_smiles does not repair adducts [M]+ and [M]- anymore, since these cases could also be due to a mistake in filling in the parent mass instead of the precursor mz. 
- repair_parent_mass_is_molar_weight does only repair parent mass and does not change the precursor mz.
- Change repair_parent_mass_is_mol_wt to repair_parent_mass_is_molar_mass
- `SpectrumProcessor` will try to incrementally save when destination files are of type .msp or .mgf
- Use StackedSparseArray for MetadataMatch equal_match when array_type is sparse [#642](https://github.com/matchms/matchms/pull/642)
- Set RDKIT version to rdkit = ">=2023.3.2,<2023.9.5" to fix installation issues. 

## [0.24.4] -2024-01-16
### Changed
- return processing_report by pipeline

## [0.24.3] -2024-01-16
### Changed
- Removed repair_precursor_is_parent_mass

- Removed option accept_parent_mass_is_mol_wt in Repair_adduct_based_on_smiles
- Merged require_precursor_mz and require_precursor_mz_below_mz into require_precursor_mz_below_mz
- Added repair_adduct_based_on_parent_mass
- Changed repair_adduct_and_parent_mass_based_on_smiles to update parent mass to the monoisotopic mass of the smiles, instead of updating based on precursor_mz and new adduct. 
## [0.24.1] -2024-01-16

- Derive_ionmode now also derives ionmode from charge, before it was only derived from the adduct. 
### Fixed
- Fix to handle spectra with empty peak arrays. [#598](https://github.com/matchms/matchms/issues/598)
- Fix instability introduced in CosineGreedy by `np.argsort`. [#595](https://github.com/matchms/matchms/issues/595)

### Changed
- Speed up save_to_mgf by preventing repetitive file opening
- Code refactoring for import functions [#593](https://github.com/matchms/matchms/pull/593).

## [0.24.0] -2023-11-21
### Added
- Option to set custom key replacements [#547](https://github.com/matchms/matchms/pull/547)
- Option to set the export style in `save_as_mgf` and `save_as_json` to choose other than matchms styles such as `nist`, `riken`, `gnps` [#557](https://github.com/matchms/matchms/pull/557)
- Added a save spectra function. To automatically save in the specified file format. [#543](https://github.com/matchms/matchms/pull/543)
- Add saving function in SpectrumProcessor [#543](https://github.com/matchms/matchms/pull/543)
### Fixed
- Fixed bug when loading empty metadata in msp [#548](https://github.com/matchms/matchms/issues/548)
- Handle missing `precursor_mz` in representation and [#452](https://github.com/matchms/matchms/issues/452) introduced by [#514](https://github.com/matchms/matchms/pull/514/files)[#540](https://github.com/matchms/matchms/pull/540)
- Fixed retention time harmonization for msp files [#551](https://github.com/matchms/matchms/issues/551)
- Fix closing mgf file after loading and prevent reopening. [#555](https://github.com/matchms/matchms/issues/555)

### Changed
- Renamed derive_smiles_from_pubchem_compound_name_search to derive_annotation_from_compound_name. [#559](https://github.com/matchms/matchms/pull/559)
- Derive_annotation_from_compound_name does not add smile or inchi when this cannot be interpreted by rdkit. [#559](https://github.com/matchms/matchms/pull/559)
- Refactored SpectrumProcessor. Reduced code repetition and improved modularity. Matchms filters can now be added as functions and in a different position than specified. [#565](https://github.com/matchms/matchms/pull/565)
- The default pipelines now stores matchms functions instead of string representation. [#565](https://github.com/matchms/matchms/pull/565)
- The option to add predefined pipelines to SpectrumProcessor has been removed. Predefined pipelines can now just be added by adding the default_pipelines (which is a list) to the filters parameter. [#565](https://github.com/matchms/matchms/pull/565)

## [0.23.1] - 2023-10-18
### Added
- Additional tests for filter pipeline order
- ProcessingReport. This adds an overview of the number of spectra changed by each filter step. (multiple PR's)
- `repair_not_matching_annotation` filter [#505](https://github.com/matchms/matchms/pull/505)
- Missing docstring documentions [#507](https://github.com/matchms/matchms/pull/507)

### Changed

- Logger warning for rdkit molecule conversion [#507](https://github.com/matchms/matchms/pull/507)
- Repair_smiles_from_compound_name, now works without matchmsextras [#509](https://github.com/matchms/matchms/pull/509/)
  - pubchempy was added as dependency
- Default filters are now stored in the yaml file as separate filters [#496](https://github.com/matchms/matchms/pull/496)
- Duplicated filters are only added once to the pipeline [#524](https://github.com/matchms/matchms/pull/524)
- Custom filters are added after default filters or at a position specified by the user [#498](https://github.com/matchms/matchms/pull/498)
- The file structure of metadata_utils was refactored [#503](https://github.com/matchms/matchms/pull/503)
- interpret_pepmass now removes the pepmass field after entering precursor_mz [#533](https://github.com/matchms/matchms/pull/533)
- Filters that did not have any effect are also mentioned in processing report [#530](https://github.com/matchms/matchms/pull/530)
- Added regex to pepmass reading to properly interpret string representations [#539](https://github.com/matchms/matchms/pull/539)


### Fixed

- handle missing weight information in `repair_parent_mass_is_mol_wt` filter [#507](https://github.com/matchms/matchms/pull/507)
- handle missing smiles in `repair_smiles_of_salts` filter [#507](https://github.com/matchms/matchms/pull/507)
- The filter settings are now stored as well in logging. [#536](https://github.com/matchms/matchms/pull/536)

## [0.22.0] - 2023-08-18

### Added

- New `SpectrumProcessing` class to be the central hub for all filter functions [#455](https://github.com/matchms/matchms/pull/455). Also takes care that filters are executed in a useful order. This is also integrated into the `Pipeline` class.

### Changed

- Adjustment to logger levels to remove uninformative warnings [#484](https://github.com/matchms/matchms/pull/484) and [#487](https://github.com/matchms/matchms/pull/487).
- Extensive code refactoring and cleaning.
- Pipeline class refactoring, Loading of yaml file happens outside Pipeline class [#479](https://github.com/matchms/matchms/pull/479)
- Yaml file now stores individual filters in the correct order [#480](https://github.com/matchms/matchms/pull/480)
- File names are not stored in yaml file anymore, they are now supplied when calling run in Pipeline [#481](https://github.com/matchms/matchms/pull/481)
- Yaml does not store logging information and spectrum files anymore [#481](https://github.com/matchms/matchms/pull/481) and [#482](https://github.com/matchms/matchms/pull/482)


## [0.21.2] - 2023-08-01

### Added
### Changed

- no more warning if precursor m/z field is updated but change is < 0.001 in `interpret_pepmass` filter step [#460](https://github.com/matchms/matchms/pull/460).
- using poetry as a build system [#466](https://github.com/matchms/matchms/pull/466)

### Fixed
- reading MoNA msp files which specify RT in minutes [#462](https://github.com/matchms/matchms/issues/462)
- added missing pyyaml dependency [#463](https://github.com/matchms/matchms/issues/463)

## [0.21.1] - 2023-07-03

### Added

- missing code documentations [#454](https://github.com/matchms/matchms/pull/454)

### Changed

- Moved matchms filter functions into new folder structure [#454](https://github.com/matchms/matchms/pull/454).
- Removed outdated (redundant) filters: `make_ionmode_lowercase` and `set_ionmode_na_when_missing` [#454](https://github.com/matchms/matchms/pull/454).

## [0.21.0] - 2023-06-30

### Added
- New filter functions to repair a smiles that do not match parent mass [#440](https://github.com/matchms/matchms/pull/440)
  - Updated adduct conversion and known adducts
  - added repair_adduct_based_on_smiles
  - added repair_parent_mass_is_mol_wt
  - added repair_precursor_is_parent_mass
  - added repair_smiles_of_salts
  - added require_parent_mass_match_smiles
  - added function to combine this in repair_parent_mass_match_smiles_wrapper
- Added repair_smiles_from_compound_name [#448](https://github.com/matchms/matchms/pull/448)
- Added require_correct_ionmode [#449](https://github.com/matchms/matchms/pull/449)
- Added require_valid_annotation [#451](https://github.com/matchms/matchms/pull/451)
- 
### Changed
- Use pandas for loading adducts dict
- Moved functions from add_parent_mass to derive_precursor_mz_and_parent_mass from
- Updated reiterate_peak_comments function to convert the peak_comments keys to float [#437](https://github.com/matchms/matchms/pull/437)
- Removed filter_by_range non-inplace version [#438](https://github.com/matchms/matchms/pull/438)
- Updated regex in get_peak_values function [#439](https://github.com/matchms/matchms/pull/439)

### Fixed
- Fixed mistake in calculating parent mass from adduct
- Added `metadata_harmonization` parameter to `load_spectra` function [#443](https://github.com/matchms/matchms/pull/443)

## [0.20.0] - 2023-05-30

### Added

- min_mz, max_mz and title parameters to spectrum plot (mostly array plot) [#419](https://github.com/matchms/matchms/pull/419)

### Changed

- Fixed pipeline filter [#414](https://github.com/matchms/matchms/pull/414)
- Removed fingerprint writing to file [#416](https://github.com/matchms/matchms/pull/416)
- Updated harmonize_values function to remove invalid metadata [#418](https://github.com/matchms/matchms/pull/418)
- Fixed metadata export style bug [#423](https://github.com/matchms/matchms/pull/423)
- Updated comment parsing logic in load_from_msp [#420](https://github.com/matchms/matchms/pull/420)
- Minor changes to regular expressions in clean_compound_name [#424](https://github.com/matchms/matchms/pull/424)

### Fixed

## [0.19.0] - 2023-05-10

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

[Unreleased]: https://github.com/matchms/matchms/compare/0.30.1...HEAD
[0.30.1]: https://github.com/matchms/matchms/compare/0.30.0...0.30.1
[0.30.0]: https://github.com/matchms/matchms/compare/0.29.0...0.30.0
[0.29.0]: https://github.com/matchms/matchms/compare/0.28.2...0.29.0
[0.28.2]: https://github.com/matchms/matchms/compare/0.28.1...0.28.2
[0.28.1]: https://github.com/matchms/matchms/compare/0.28.0...0.28.1
[0.28.0]: https://github.com/matchms/matchms/compare/0.27.0...0.28.0
[0.27.0]: https://github.com/matchms/matchms/compare/0.26.4...0.27.0
[0.26.4]: https://github.com/matchms/matchms/compare/0.26.3...0.26.4
[0.26.3]: https://github.com/matchms/matchms/compare/0.26.2...0.26.3
[0.26.2]: https://github.com/matchms/matchms/compare/0.26.1...0.26.2
[0.26.1]: https://github.com/matchms/matchms/compare/0.26.0...0.26.1
[0.26.0]: https://github.com/matchms/matchms/compare/0.25.0...0.26.0
[0.25.0]: https://github.com/matchms/matchms/compare/0.24.4...0.25.0
[0.24.4]: https://github.com/matchms/matchms/compare/0.24.3...0.24.4
[0.24.3]: https://github.com/matchms/matchms/compare/0.24.2...0.24.3
[0.24.2]: https://github.com/matchms/matchms/compare/0.24.1...0.24.2
[0.24.1]: https://github.com/matchms/matchms/compare/0.24.0...0.24.1
[0.24.0]: https://github.com/matchms/matchms/compare/0.23.1...0.24.0
[0.22.0]: https://github.com/matchms/matchms/compare/0.21.2...0.22.0
[0.21.2]: https://github.com/matchms/matchms/compare/0.20.1...0.21.2
[0.21.1]: https://github.com/matchms/matchms/compare/0.21.0...0.21.1
[0.21.0]: https://github.com/matchms/matchms/compare/0.20.0...0.21.0
[0.20.0]: https://github.com/matchms/matchms/compare/0.19.0...0.20.0
[0.19.0]: https://github.com/matchms/matchms/compare/0.18.0...0.19.0
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
