
![](documentation/logo/shapfirelogov2.png)


# About

[![License](https://img.shields.io/badge/Code%20style-Black-white)](https://img.shields.io/badge/Code%20style-Black-white)
[![Code Style](https://img.shields.io/badge/License-MIT-white.svg)](https://shapfire.readthedocs.io/en/latest/?badge=latest)


ShapFire is an automated and *wrapper-based* approach to *feature importance
ranking* and *feature selection* based on *SHAP* Feature Importance Rank
Ensembling (SHAPFire, stylized ShapFire).

ShapFire is built on top of Microsofts gradient boosting decision tree framework
[LightGBM](https://github.com/microsoft/LightGBM/) and the
[SHAP](https://github.com/slundberg/shap/) (SHapley Additive exPlanations)
Python library for Machine Learning (ML) model inspection and interpretation.

The ShapFire approach is motivated by the fact that highly associated features
in an input dataset can affect ML model interpretability, making
it hard to obtain accurate feature importance rankings.

ShapFire aims to work specifically in a setting where the input dataset
contains several highly associated features that need to be assigned a globally
consistent ranking that, e.g., a domain expert can further assess.


# Getting Started

Install the development version from the git source:

```bash
pip install git+https://github.com/nicklasxyz/shapfire.git
```

Then see: [Binary Classification Demo](https://nicklasxyz.github.io/shapfire/source/examples/classification_demo.html)

# Development

```bash
# First, fork or clone the repo, then:
conda create -n shapfire python=3.9 && \
pip install poetry && \
poetry install

# Install pandoc to be able to work with python notebooks
# in the documentation
conda install -c conda-forge pandoc

# Generate docs
sphinx-build documentation docs && touch docs/.nojekyll
```
