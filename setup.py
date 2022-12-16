#!/usr/bin/env python
import os
from setuptools import find_packages, setup


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
    long_description=readme,
    long_description_content_type="text/x-rst",
    author="Netherlands eScience Center",
    author_email="generalization@esciencecenter.nl",
    url="https://github.com/matchms/matchms",
    packages=find_packages(exclude=['*tests*']),
    package_data={"matchms": ["data/*.csv"]},
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
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9"
    ],
    python_requires='>=3.7',
    install_requires=[
        "deprecated",
        "lxml",
        "matplotlib",
        "networkx",
        "numba >=0.47",
        "numpy",
        "pickydict >= 0.4.0",
        "pyteomics >=4.2",
        "requests",
        "scipy",
        "sparsestack >= 0.4.1",
        "tqdm",
    ],
    extras_require={"dev": ["bump2version",
                            "decorator",
                            "isort>=5.1.0",
                            "pylint<2.12",
                            "prospector[with_pyroma]",
                            "pytest",
                            "pytest-cov",
                            "sphinx>=4.0.0",
                            "sphinx_rtd_theme",
                            "sphinxcontrib-apidoc",
                            "testfixtures",
                            "yapf",],
                    "chemistry": ["rdkit >=2020.03.1"]},
)
