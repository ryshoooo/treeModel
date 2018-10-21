"""
Base module for tree input data
"""

import collections
import numpy as np

from .datatypes import DataType, ListDataType


class Node(object):
    """
    Main node object, can be considered as a data point or a collection of data points.
    """

    def __init__(self):
        """
        Instantiation of the main node object.
        """
        # Set children nodes, name and the data_type of the node to None.
        self.children = None
        self.name = None
        self.data_type = None

    def is_child(self):
        """
        Simple method to determine whether the node is a leaf.
        :return: Boolean
        """
        return self.children is None and self.name is not None and self.data_type is not None and not isinstance(
            self.data_type, ListDataType)

    def is_fork(self):
        """
        Simple method to determine whether the node is forking.
        :return: Boolean
        """
        return self.children is not None and self.name is not None and self.data_type is not None and isinstance(
            self.data_type, ListDataType)

    def overwrite_children(self, name, children, data_type):
        """
        Force method which sets the name and the children leaves to the node.
        :param children: Array-like of Nodes.
        :param name: String specifying the name of the node.
        :param data_type: Tree-like data type of instance DataType specifying the underlying fork.
        :return: Instance of the object itself with children and name set.
        """
        if not isinstance(children, (collections.Sequence, np.ndarray)) or isinstance(children, str):
            raise AttributeError("Incorrect format of input children nodes!")

        for child in children:
            if not isinstance(child, Node):
                raise AttributeError("Nodes have to be of Node instance!")

        self.children = children

        if not isinstance(name, str):
            raise AttributeError("The name of the node has to be a string!")

        self.name = name

        if not isinstance(data_type, DataType):
            raise AttributeError("Unsupported input data type: '{}'".format(data_type))

        self.data_type = data_type

        return self

    def overwrite_child(self, name, data_type):
        """
        Force method which sets the name and the data type to the node.
        :param name: String specifying the name of the node.
        :param data_type: Instance of DataType specifying the type of data for the node.
        :return: Instance of the object itself with name and data type set.
        """
        if not isinstance(name, str):
            raise AttributeError("The name of the node has to be a string!")

        self.name = name

        if not isinstance(data_type, DataType):
            raise AttributeError("Unsupported input data type: '{}'".format(data_type))

        self.data_type = data_type

        return self

    def get_children(self):
        """
        Get the list of the children nodes.
        :return: List of Nodes.
        """
        if not self.is_fork():
            raise AttributeError("Cannot get children from a leaf!")

        if self.children is None:
            raise RuntimeError("Empty children leaves!")

        return self.children

    def get_name(self):
        """
        Get the name of the node.
        :return: String.
        """
        if self.name is None:
            raise RuntimeError("The name of the node is missing!")

        return self.name

    def get_data_type(self):
        """
        Get the DataType of the node.
        :return: DataType.
        """
        if self.data_type is None:
            raise RuntimeError("The data type is missing!")

        return self.data_type


class ForkNode(Node):
    """
    Fork node.
    """

    def __init__(self, name, children, data_type):
        super(ForkNode, self).__init__()

        self.overwrite_children(name=name, children=children, data_type=data_type)


class ChildNode(Node):
    """
    Leaf.
    """

    def __init__(self, name, data_type):
        super(ChildNode, self).__init__()

        self.overwrite_child(name=name, data_type=data_type)
