# About

ShapFire is an automated, wrapper-based approach for feature importance ranking and feature selection based on SHAP Feature Importance Rank Ensembling (SHAPFire, stylized ShapFire).

ShapFire is built on top of Microsofts gradient boosting decision tree framework
[LightGBM](https://github.com/microsoft/LightGBM/) and the
[SHAP](https://github.com/slundberg/shap/) (SHapley Additive exPlanations)
Python library for Machine Learning (ML) model inspection and interpretation.

## Reference

This branch contains the version of ShapFire applied in the following paper:

> Skovbo, J. S., Andersen, N. S., Obel, L. M., Laursen, M. S., Riis, A. S., Houlind, K. C., Pyndt Diederichsen, A. C., & Lindholt, J. S. (2025). Individual risk assessment for rupture of abdominal aortic aneurysm using artificial intelligence. Journal of Vascular Surgery, 81(3), 613–622.e5.
> https://doi.org/10.1016/j.jvs.2024.11.017

## Setup 

First of all, clone the repository and enter into the root of the directory, then proceed as follows.

### Create a virtual environment

ShapFire has been tested with Python 3.11, so create a virtual environment using this version (e.g., using [conda](/docs/getting-started/miniconda/main#should-i-install-miniconda-or-anaconda-distribution)):

```bash
conda create -n shapfire python=3.11;
conda activate shapfire
```

### Install dependencies

```bash
pip install -r requirements.txt
```

## Running the Analysis Notebook

The analysis code used for the paper is provided in the Jupyter notebook `AAA-analysis.ipynb`, and can be run by additionally installing Jupyter and opening the notebook:

```bash
pip install jupyter
jupyter notebook
```

Once the notebook is open, then run all cells.

## Applying Pre-trained Models

Three pre-trained models are available in the repository. Each includes model-specific data, metadata, and a schema for handling input/output:

- `export_all_features_prod/`: Production model trained using all available features
- `export_selected_features_prod/`: Production model trained using ShapFire-selected features
- `export_single_feature_prod/`: Production model trained on a single feature (maximal anterior–posterior diameter)

Predictions can be generated using the provided `predict.py` script (referencing a model-specific directory):

```bash
python predict.py \
    --export-dir export_selected_features_prod \
    --input dataset.csv \
    --output predictions.csv
```

Note that input datasets (`dataset.csv`) must match the feature names and structure defined in the corresponding `schema.json` contained in the specified `export-dir` (in this case the `export_selected_features_prod`). Furthermore, the script will save predictions to the specified output file (`predictions.csv`) in CSV format.