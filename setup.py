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
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8"
    ],
    test_suite="tests",
    install_requires=[
        "deprecated",
        "matplotlib",
        "numba >=0.47",
        "numpy",
        "pyteomics >=4.2",
        "requests",
        "pyyaml",
    ],
    setup_requires=[
    ],
    tests_require=[
        "bump2version",
        "deprecated",
        "isort>=4.2.5,<5",
        "matplotlib",
        "numba >=0.47",
        "numpy",
        "prospector[with_pyroma]",
        "pyteomics >=4.2",
        "pytest",
        "pytest-cov",
        "pytest-runner",
        "pyyaml",
        "recommonmark",
        "requests",
        "sphinx>=3.0.0,!=3.2.0,<4.0.0",
        "sphinx_rtd_theme",
        "sphinxcontrib-apidoc",
        "yapf",
    ],
    extras_require={"cosine_hungarian": ["scipy"],
                    "chemistry": ["rdkit >=2020.03.1"]},
    package_data={"matchms": ["data/*.yaml"]},
)
