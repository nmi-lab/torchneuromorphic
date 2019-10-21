"""
DCLL library
"""

# Always prefer setuptools over distutils
from setuptools import setup, find_packages
# To use a consistent encoding
from codecs import open
from os import path

here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name = "pytorch_datasets",
    version = "0.1",
    author = "Emre Neftci",
    author_email = "eneftci@uci.edu",
    description = ("Dataset loaders for pytorch"),
    packages=find_packages(exclude=['contrib', 'docs', 'tests']),
    test_suite = 'nose.collector',
    long_description=long_description,
    license='Apache License 2.0',
    install_requires=[
        "torch>=1.1.0",
        "scipy>=1.0",
        "h5py"
    ]
)
