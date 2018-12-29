"""
This module contains the main functionality for defining a single node, a forking node and a tree itself.

The current implementation follows the following design:
    - A single data point is considered to be a `leaf`. The implementation of the `leaf` representation is in the class :class:`treemodel.datamodel.tree.ChildNode`.
    - Each single data point needs to have assigned a `DataType` of class :class:`treemodel.datamodel.datatypes.DataType`. There are multiple choices for a `DataType`:
        - :class:`treemodel.datamodel.datatypes.FloatDataType`: covers integers, floating points and numerically discrete random variables.
        - :class:`treemodel.datamodel.datatypes.StringDataType`: covers string (i.e. categorical) variables. Any other data type can be converted into the `StringDataType`.
        - :class:`treemodel.datamodel.datatypes.DateDataType`: a special type of strings which can be represented as dates or datetimes.
        - :class:`treemodel.datamodel.datatypes.ArrayDataType`: a sequence of values of the same data type.
        - :class:`treemodel.datamodel.datatypes.ListDataType`: a sequence of values of different types.
    - In case we have a collection of data points under 1 single branch, then the fork is considered to be a new data point. The implementation of this data points collection is in the class :class:`treemodel.datamodel.tree.ForkNode`.
    - Each `ForkNode` has a specific data type known as :class:`treemodel.datamodel.tree.TreeDataType`, which bears the full schema of the underlying fork node.
    - Finally a single tree is constructed as a collection of fork and leaf nodes combined in the correct hierarchy under 1 single ``root`` fork.
    - A single tree is the basis for :class:`treemodel.datamodel.base.TreeRow`, which represents a single row in the :class:`treemodel.datamodel.base.TreeDataSet`.
"""
