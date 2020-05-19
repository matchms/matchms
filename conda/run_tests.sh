#!/bin/bash

echo ${PREFIX}
echo ${RECIPE_DIR}

pip install -r ${RECIPE_DIR}/conda/requirements-dev.txt

prospector -o grouped -o pylint:pylint-report.txt

isort --check-only --diff --conda-env matchms --recursive --wrap-length 79 --lines-after-imports 2 --force-single-line --no-lines-before FUTURE --no-lines-before STDLIB --no-lines-before THIRDPARTY --no-lines-before FIRSTPARTY --no-lines-before LOCALFOLDER matchms/ tests/ integration-tests/

cat readthedocs/_build/coverage/python.txt
UNCOVERED_MEMBERS=$(grep '*' readthedocs/_build/coverage/python.txt | wc -l)
UNCOVERED_MEMBERS_ALLOWED=25
if (( $UNCOVERED_MEMBERS > $UNCOVERED_MEMBERS_ALLOWED )) ; then echo "There are currently ${UNCOVERED_MEMBERS} uncovered members in the documentation, which is more than the ${UNCOVERED_MEMBERS_ALLOWED} allowed."; exit 1;fi
echo "The code is sufficiently documented with ${UNCOVERED_MEMBERS} uncovered members out of ${UNCOVERED_MEMBERS_ALLOWED} allowed.";

python setup.py test

pytest --cov --cov-report term --cov-report xml --junitxml=xunit-result.xml
