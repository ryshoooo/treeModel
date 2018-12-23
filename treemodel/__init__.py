"""
This module contains the whole functionality of the :mod:`treemodel` library.

The library is currently divided into 2 submodules:
    - :mod:`treemodel.datamodel`: Contains all the functionality for building and parsing trees from the input data to the :class:`treemodel.datamodel.base.TreeDataSet`.
    - :mod:`treemodel.linear_model`: Contains functionality for building standard sklearn linear models on TreeDataSets.
"""
from . import datamodel

__version__ = "0.0.1"
