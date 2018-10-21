"""
Base module for tree input data
"""

import collections
import numpy as np

from .datatypes import TreeDataType


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

    def get_name(self):
        if self.name is None:
            raise RuntimeError("The name of the node is missing!")

        return self.name


class ForkNode(Node):
    """
    Fork node.
    """

    def __init__(self, name, children):
        super().__init__()

        self.overwrite_children(children, name)


class ChildNode(Node):
    """
    Leaf.
    """

    def __init__(self, name, data_type):
        super().__init__()

        self.overwrite_child(name, data_type)
