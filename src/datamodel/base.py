"""
This module contains base classes and methods for the input data model.
"""
import collections
import numpy as np

from .tree import Node


class BaseTreeRow(object):
    """
    The superclass containing the base tree input row.
    """

    def __init__(self, input_row, schema=None):
        self.input_row = input_row

        if not isinstance(schema, TreeSchema):
            raise AttributeError("The schema for the row has to be of TreeSchema instance!")

        self.schema = schema

    def set_schema(self, schema):
        self.schema = schema


class TreeSchema(object):
    """
    Base class for input schema for a particular dataset.
    """

    def __init__(self, nodes):
        self.num_levels = 0

        if not isinstance(nodes, (collections.Sequence, np.ndarray)) or isinstance(nodes, str):
            raise AttributeError("Incorrect format of input nodes!")

        for node in nodes:
            if not isinstance(node, Node):
                raise AttributeError("Nodes have to be of Node instance!")

        self.dict_schema = self.build_dict_schema(nodes)
