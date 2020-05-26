#!/usr/bin/env python
import os
from setuptools import find_packages
from setuptools import setup


here = os.path.abspath(os.path.dirname(__file__))

version = {}
with open(os.path.join(here, "matchms", "__version__.py")) as f:
    exec(f.read(), version)

with open("README.rst") as readme_file:
    readme = readme_file.read()

setup(
    name="matchms",
    version=version["__version__"],
    description="Python library for fuzzy comparison of mass spectrum data and other Python objects",
    long_description=readme + "\n\n",
    author="Netherlands eScience Center",
    author_email="generalization@esciencecenter.nl",
    url="https://github.com/matchms/matchms",
    packages=find_packages(exclude=['*tests*']),
    include_package_data=True,
    license="Apache Software License 2.0",
    zip_safe=False,
    keywords=[
        "similarity measures",
        "mass spectrometry",
        "fuzzy matching",
        "fuzzy search"
    ],
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: Apache Software License",
        "Natural Language :: English",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7"
    ],
    test_suite="tests",
    install_requires=[
        # see conda/environment.yml
    ],
    setup_requires=[
    ],
    tests_require=[
        # see conda/environment-dev.yml
    ],
    extras_require={
    },
    package_data={"matchms": ["data/*.yaml"]},
)
