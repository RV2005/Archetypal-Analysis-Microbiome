#!/usr/bin/env python3
"""
Setup script for archetypal_analysis package.
"""

from setuptools import setup, find_packages
import os

# Read long description from README
here = os.path.abspath(os.path.dirname(__file__))
try:
    with open(os.path.join(here, '..', 'README.md'), 'r', encoding='utf-8') as f:
        long_description = f.read()
except FileNotFoundError:
    long_description = "Archetypal Analysis Package for Genetic Heritage Group Research"

setup(
    name="archetypal-analysis",
    version="1.0.0",
    author="Genetic Heritage Group",
    author_email="research@ghg.org",
    description="Archetypal Analysis Package for Genetic Heritage Group Research",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ghg/archetypal-analysis",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
    ],
    python_requires=">=3.8",
    install_requires=[
        "pandas>=1.3.0",
        "numpy>=1.20.0",
        "matplotlib>=3.3.0",
        "scikit-learn>=1.0.0",
        "scikit-bio>=0.5.6",
        "tqdm>=4.60.0",
        "biopython>=1.78",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "black>=21.0",
            "flake8>=3.8",
            "mypy>=0.910",
        ],
    },
    package_data={
        "archetypal_analysis": ["*.json", "*.yaml", "*.yml"],
    },
    include_package_data=True,
    zip_safe=False,
)
