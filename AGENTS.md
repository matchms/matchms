# matchms

This file provides guidance to AI coding agents when working with code in this
repository.

## Communication

Use ASD-STE100 Simplified Technical English when you communicate with the user.
Use short sentences, active voice, and consistent terms. Keep code identifiers,
API names, and quoted source text exact. State the result and any limits clearly.

## Project overview

Matchms is an open-source Python package for importing, processing, cleaning,
exporting, and comparing tandem mass spectrometry data (MS/MS). It supports
reproducible workflows that transform raw spectra from common file formats
into cleaned, harmonized, and comparable spectral datasets.

## Project structure

```text
matchms/                 # main Python package
.agents/skills/          # usage skills shipped inside the package (see below)
tests/                   # pytest suite and test data
readthedocs/             # Readthedocs API documentation and examples
```

## Skills

- **Development skills** (for contributors working *on* matchms) are in this file.
- **Usage skills** (for agents *using* matchms) are shipped
  inside the package at
  [`.agents/skills/matchms/`](.agents/skills/matchms/SKILL.md).

## Developer documentation

Before making changes, read the "Documentation for developers" section in [`README.rst`](README.rst) and follow it. It is the source of truth for setup, testing, code style, and contribution workflow. If this file and the README disagree, the README wins.

## Code standards

- Keep public APIs typed and compatible with Python 3.10+.
- Add focused tests for behavior changes; regenerate reference data only when
  conversion outputs intentionally change.
- Do not add trivial or self-validating tests. Tests should verify meaningful application behavior, regressions, or integration boundaries, not restate assumptions about well-established library functionality or implementation details introduced only to validate the agent’s own code changes. Avoid mock-heavy tests unless mocking is the clearest way to exercise a real contract or failure mode.

## Rules

- Run the tests and linters described in "Documentation for developers" section before proposing a change, and make sure they pass.
- Keep changes focused and minimal. Do not reformat or refactor unrelated code.
- Keep edits scoped and consistent with the surrounding module.
- Add or update tests for any behavior change.
- Do not edit generated files by hand.
- Do not change the public API without being asked to.
- Update README.rst/examples when user-facing behavior changes.
