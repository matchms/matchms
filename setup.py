#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os

from setuptools import setup

here = os.path.abspath(os.path.dirname(__file__))

# To update the package version number, edit matchms/__version__.py
version = {}
with open(os.path.join(here, 'matchms', '__version__.py')) as f:
    exec(f.read(), version)

with open('README.rst') as readme_file:
    readme = readme_file.read()

setup(
    name='matchms',
    version=version['__version__'],
    description="Vector representation and similarity measure for mass spectrometry data",
    long_description=readme + '\n\n',
    author="Netherlands eScience Center",
    author_email='generalization@esciencecenter.nl',
    url='https://github.com/matchms/matchms',
    packages=[
        'matchms',
    ],
    include_package_data=True,
    license="Apache Software License 2.0",
    zip_safe=False,
    keywords='matchms',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: Apache Software License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7'
    ],
    test_suite='tests',
    install_requires=[
        # see environment.yml
    ],
    setup_requires=[
        # see environment.yml
    ],
    tests_require=[
        # see environment.yml
    ],
    extras_require={
        # see environment.yml
    }
)
