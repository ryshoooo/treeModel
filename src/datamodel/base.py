"""
This module contains base classes and methods for the input data model.
"""
import collections
import numpy as np


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


class Node(object):
    """
    Superclass to all the tree nodes.
    """

    def __init__(self):
        self.children = None
        self.name = None
        self.data_type = None

    def is_child(self):
        if self.children is None and self.name is not None and self.data_type is not None:
            return True
        else:
            return False

    def is_fork(self):
        if self.children is None and self.name is not None and self.data_type is not None:
            return False
        else:
            return True

    def overwrite_children(self, children, name):
        if not isinstance(children, (collections.Sequence, np.ndarray)) or isinstance(children, str):
            raise AttributeError("Incorrect format of input children nodes!")

        for child in children:
            if not isinstance(child, Node):
                raise AttributeError("Nodes have to be of Node instance!")

        self.children = children

        if not isinstance(name, str):
            raise AttributeError("The name of the node has to be a string!")

        self.name = name

        return self

    def overwrite_child(self, name, data_type):
        if not isinstance(name, str):
            raise AttributeError("The name of the node has to be a string!")

        self.name = name

        if not isinstance(data_type, TreeDataType):
            raise AttributeError("Unsupported input data type: '{}'".format(data_type))

        self.data_type = data_type

        return self

    def get_children(self):
        if not self.is_fork():
            raise AttributeError("Cannot get children from a leaf!")

        if self.children is None:
            raise RuntimeError("Empty children leaves!")

        return self.children


class TreeDataType(object):
    """
    Conversion between numpy and python types for the Tree input data type.
    """


class ForkNode(Node):
    """
    Fork node.
    """

    def __init__(self, name, children):
        super().__init__()

        self.overwrite_children(children, name)


class ChildNode(Node):
    """
    Superclass to all the leafs.
    """

    def __init__(self, name, data_type):
        super().__init__()

        self.overwrite_child(name, data_type)
