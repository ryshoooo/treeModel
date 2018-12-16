"""
This module contains base classes and methods for the input data model.
"""
import collections
import numpy as np
from copy import deepcopy
from functools import reduce

from .datatypes import FloatDataType, StringDataType, ArrayDataType, ListDataType
from .tree import TreeSchema, ForkNode, ChildNode


class TreeRow(object):
    """
    The superclass containing the base tree input row.

    :param input_row: Dictionary with input data.
    :param schema: Either None or TreeSchema object specifying the input_row types. In case the schema is None,
    the schema will be automatically inferred from the input_row.
    """

    def __init__(self, input_row, schema=None):
        """
        Initialize the TreeRow object.
        """
        if schema is None:
            self.schema = self.infer_schema(input_row)
        else:
            self.set_schema(schema)

        self.row = None

    def build_row(self, input_row, method):
        """
        Construct TreeRow object from the input_row and specified schema.
        :param input_row: Dictionary with input data.
        :param method: String
        :return: TreeRow object with the input data.
        """
        self.row = self.build_tree(input_row, method)
        return self

    def get_schema(self):
        """
        Getter method for tree's schema.
        :return: TreeSchema.
        """
        if self.schema is None:
            raise AttributeError("The schema for the TreeRow is missing!")

        return deepcopy(self.schema)

    def set_schema(self, schema):
        """
        Sets schema for the TreeRow object.
        :param schema: TreeSchema
        :return: Instance of the TreeRow object with updated schema.
        """
        if not isinstance(schema, TreeSchema):
            raise AttributeError("The schema for the row has to be of TreeSchema instance!")
        else:
            self.schema = schema

        return self

    def build_tree(self, input_row, method):
        """
        Method which builds tree from input dictionary.
        :param input_row: Dictionary with the input data.
        :param method: String
        :return: Dictionary with the typed data.
        """
        if not isinstance(input_row, dict):
            input_row = dict(input_row)

        if method == 'numpy':
            return self.schema.base_fork_node.get_data_type().build_numpy_value(input_row)
        elif method == 'python':
            return self.schema.base_fork_node.get_data_type().build_python_value(input_row)
        else:
            raise RuntimeError("Unknown method: '{}'".format(method))

    @staticmethod
    def _is_float(n):
        """
        Helper method which determines whether an object is a float or not.
        :param n: Object
        :return: Boolean
        """
        try:
            float(n)
            return True
        except (ValueError, TypeError):
            return False

    def _infer_element(self, value, name, current_level, within_array=False):
        """
        Helper method which creates a Node object based on the input element.
        :param value: Input value, which's type is being inferred.
        :param name: Name of the Node.
        :param current_level: Integer specifying the level of the Node in the tree hierarchy.
        :param within_array: Boolean specifying whether this value is from a list
        :return: Node object with specified type and name
        """
        if isinstance(value, dict):
            return self._infer_fork_type(value, name, current_level + 1 + within_array)
        elif isinstance(value, list):
            return self._infer_list_type(value, name, current_level)
        elif isinstance(value, float) or self._is_float(value):
            return ChildNode(name=name, data_type=FloatDataType())
        else:
            return ChildNode(name=name, data_type=StringDataType())

    def _infer_list_type(self, value, name, current_level):
        """
        Helper method which creates a Node object based on the value in list format.
        :param value: List of values
        :param name: String representing the name of the node
        :param current_level: Integer specifying the level of the node
        :return: Node
        """
        # First infer element of each list element
        elements = [self._infer_element(x, name, current_level, True) for x in value]

        # Get the inferred data types
        try:
            first, *rest = [x.get_data_type() for x in elements]
        except ValueError:
            first, rest = StringDataType(), []

        # In case all the data types are equal, return ArrayDataType
        if all([first == x for x in rest]):
            return ChildNode(name=name, data_type=ArrayDataType(element_data_type=first))

        # Otherwise return list data type
        else:
            elements_types = [element.set_name("{}_{}".format(name, ind)).get_data_type() for ind, element in
                              enumerate(elements)]
            return ChildNode(name=name, data_type=ListDataType(element_data_types=elements_types,
                                                               level=current_level + 1))

    def _infer_fork_type(self, input_dict, key, level):
        """
        Helper method which infers forks from the input dictionary with correct data type.
        :param input_dict: Dictionary with the input data.
        :param key: String specifying the name of the fork.
        :param level: Integer specifying the current level of the fork in the tree hierarchy.
        :return: ForkNode with the specified children of the inferred data type.
        """
        sorted_children_names = sorted(input_dict.keys())
        children = [self._infer_element(input_dict[name], name, level) for name in sorted_children_names]
        return ForkNode(name=key, children=children, level=level)

    def infer_schema(self, input_dict, initial_name='base'):
        """
        Method to infer the schema for the input_row.
        :param input_dict: Dictionary with the input data.
        :param initial_name: Name for the row.
        :return: TreeSchema object specifying the schema of the row.
        """
        if not isinstance(input_dict, dict):
            input_dict = dict(input_dict)

        return TreeSchema(base_fork_node=self._infer_fork_type(input_dict, initial_name, 1))


class TreeDataSet(object):
    """
    Base class for dataset of trees.

    :param input_rows: A collection of built TreeRows or python dictionaries.
    :param schema: Either None or TreeSchema specifying the schema for every row or collection of TreeSchemas
        specifying the TreeSchema for each input row in order. In case the schema is None, each row will infer the
        schema automatically.
    :param method: String specifying the method to build each row, either 'python' or 'numpy'
    """

    def __init__(self, input_rows, schema=None, method='numpy'):
        """
        Initialize the dataset object.
        """
        if not isinstance(input_rows, (collections.Sequence, np.ndarray)) or isinstance(input_rows, str):
            raise AttributeError("Incorrect format of input rows!")

        for row in input_rows:
            if not isinstance(row, (dict, TreeRow)):
                raise AttributeError("Input rows have to be of dictionary or tree type!")

        if method not in ['numpy', 'python']:
            raise ValueError("Unknown input method: '{}'".format(method))

        if schema is None or isinstance(schema, TreeSchema):
            self.data = np.array([self._get_tree_row(input_row=row, schema=schema, method=method)
                                  for row in input_rows])
        elif isinstance(schema, (collections.Sequence, np.ndarray)):
            self.data = np.array([self._get_tree_row(input_row=input_rows[ind], schema=val, method=method)
                                  for ind, val in enumerate(schema)])
        else:
            raise AttributeError("Incorrect format of input schema!")

    @staticmethod
    def _get_tree_row(input_row, schema, method):
        """
        Helper method to get tree row with specified schema.
        :param input_row: Either a python dictionary or built TreeRow.
        :param schema: None or TreeSchema. In case of None, the schema for the row will be inferred automatically.
        :param method: String specifying the built method.
        :return: TreeRow with built row.
        """
        if isinstance(input_row, TreeRow):
            if schema is None:
                return input_row
            elif isinstance(schema, TreeSchema):
                input_row = input_row.set_schema(schema)
                return input_row.build_row(input_row.row, method)
            else:
                raise AttributeError("Input schema parameter is not a TreeSchema object!")
        else:
            if schema is None:
                return TreeRow(input_row=input_row).build_row(input_row=input_row, method=method)
            elif isinstance(schema, TreeSchema):
                return TreeRow(input_row=input_row, schema=schema).build_row(input_row=input_row, method=method)
            else:
                raise AttributeError("Input schema parameter is not a TreeSchema object!")

    def uniformize_schema(self, method='fixed', schema=None):
        """
        Uniformizes the schema on each row of the dataset.
        :param method: String specifying the method of uniformization. Currently only 'intersection', 'union' and
        'fixed' are supported. With value 'fixed' additional parameter `schema` has to be specified.
        :param schema: TreeSchema specifying the schema to set in case of 'fixed' method on each row.
        :return: TreeDataSet
        """
        if method == 'intersection':
            schema_arr = np.apply_along_axis(lambda x: x[0].get_schema(), 0, self.data.reshape((1, -1)))
            schema_res = np.multiply.reduce(schema_arr, 0)
            self.data = np.apply_along_axis(lambda x: x[0].set_schema(schema_res), 0, self.data.reshape((1, -1)))

            return self
        elif method == 'union':
            schema_arr = np.apply_along_axis(lambda x: x[0].get_schema(), 0, self.data.reshape((1, -1)))
            schema_res = np.add.reduce(schema_arr, 0)
            self.data = np.apply_along_axis(lambda x: x[0].set_schema(schema_res), 0, self.data.reshape((1, -1)))

            return self
        elif method == 'fixed':
            if schema is None:
                raise ValueError("Parameter 'schema' is missing for method 'fixed'.")

            self.data = np.apply_along_axis(lambda x: x[0].set_schema(schema), 0, self.data.reshape((1, -1)))

            return self
        else:
            raise ValueError("Unknown method '{}'!".format(method))
