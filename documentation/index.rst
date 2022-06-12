.. image:: logo/shapfirelogov2.png
   :scale: 66 %
   :alt: alternate text
   :align: center
   :class: no-scaled-link

|

About
======
..
    |Documentation Status| |PyPI Version| |PyPI downloads| |Code Style|

|License| |Code Style|

ShapFire is an automated and *wrapper-based* approach to *feature importance
ranking* and *feature selection* based on *SHAP* Feature Importance Rank
Ensembling (SHAPFire, stylized ShapFire).

ShapFire is built on top of Microsofts gradient boosting decision tree framework
`LightGBM <https://github.com/microsoft/LightGBM/>`_ and the `SHAP
<https://github.com/slundberg/shap/>`_ (SHapley Additive exPlanations) Python
library for Machine Learning (ML) model inspection and interpretation.

The ShapFire approach is motivated by the fact that highly associated features
in an input dataset can affect ML model interpretability, making
it hard to obtain accurate feature importance rankings.

ShapFire aims to work specifically in a setting where the input dataset
contains several highly associated features that need to be assigned a globally
consistent ranking that, e.g., a domain expert can further assess.


Getting Started
===============

Install the development version from the git source:

.. code-block:: bash

   pip install git+https://github.com/nicklasxyz/shapfire.git

Then see: `Binary Classification Demo <./source/examples/classification_demo.ipynb>`_

.. toctree::
   :hidden:

   self

.. toctree::
   :hidden:

   source/examples/index

.. toctree::
   :hidden:

   source/api/api_reference


.. |Documentation Status| image:: https://readthedocs.org/projects/shapfire/badge/?version=latest
   :alt: Documentation Status
   :target: https://shapfire.readthedocs.io/en/latest/?badge=latest

.. |PyPI Downloads| image:: https://img.shields.io/pypi/dm/shapfire?label=PyPI%20Downloads
   :alt: PyPI - Downloads
   :target: https://pypi.org/project/shapfire/

.. |PyPI Version| image:: https://img.shields.io/pypi/v/shapfire.svg
   :target: https://pypi.org/project/shapfire/

.. |Code Style| image:: https://img.shields.io/badge/Code%20style-Black-white
   :alt: Code Style
   :target: https://img.shields.io/badge/Code%20style-Black-white

.. |License| image:: https://img.shields.io/badge/License-MIT-white.svg
   :alt: License
   :target: https://opensource.org/licenses/MIT
