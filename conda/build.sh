#!/bin/bash

# Show environment variables
env

$PYTHON -m pip install --no-deps --ignore-installed . -vv

# # Copy data folder
# mkdir -p $PREFIX/site-packages/matchms/data
# cp matchms/data/* $PREFIX/site-packages/matchms/data/
