"""
Setup configuration for the archetypal_analysis package.
"""

from setuptools import setup, find_packages
import os

# Read the README file for long description
def read_readme():
    readme_path = os.path.join(os.path.dirname(__file__), 'README.md')
    if os.path.exists(readme_path):
        with open(readme_path, 'r', encoding='utf-8') as f:
            return f.read()
    return ""

# Package metadata
setup(
    name="archetypal-analysis",
    version="1.0.0",
    author="Genetic Heritage Group",
    author_email="your.email@nyu.edu",
    description="A modular package for microbiome archetypal analysis",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/archetypal-analysis",
    
    # Package configuration
    packages=find_packages(),
    python_requires=">=3.7",
    
    # Dependencies
    install_requires=[
        "pandas>=1.0.0",
        "numpy>=1.18.0",
        "matplotlib>=3.0.0",
        "scikit-learn>=0.22.0",
        "scikit-bio>=0.5.5",
        "tqdm>=4.0.0",
    ],
    
    # Optional dependencies
    extras_require={
        "blast": ["biopython>=1.78"],
        "dev": [
            "pytest>=6.0",
            "pytest-cov",
            "black",
            "flake8",
            "jupyter",
            "ipykernel"
        ],
        "all": [
            "biopython>=1.78",
            "pytest>=6.0", 
            "pytest-cov",
            "black",
            "flake8",
            "jupyter",
            "ipykernel"
        ]
    },
    
    # Package data
    include_package_data=True,
    package_data={
        'archetypal_analysis': [
            'examples/*.py',
            'config/*.json',
        ],
    },
    
    # Classifiers
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8", 
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    
    # Keywords
    keywords="microbiome, archetypal-analysis, bioinformatics, metagenomics",
    
    # Entry points (if you want command-line tools)
    entry_points={
        'console_scripts': [
            'archetypal-analysis=archetypal_analysis.examples.example_workflow:main',
        ],
    },
    
    # Project URLs
    project_urls={
        "Bug Reports": "https://github.com/yourusername/archetypal-analysis/issues",
        "Source": "https://github.com/yourusername/archetypal-analysis",
        "Documentation": "https://archetypal-analysis.readthedocs.io/",
    },
)
