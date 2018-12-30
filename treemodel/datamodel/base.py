"""
This module contains the base functionality being used in the tree type of modelling.
To do any kind of analysis we need to have established classes, which allow us to work with tree-structured data.
In this module one can find the implementation of:
    - :class:`treemodel.datamodel.base.TreeRow`: Row-like representation of the input tree data.
    - :class:`treemodel.datamodel.base.TreeDataSet`: Collection of :class:`treemodel.datamodel.base.TreeRow` forming a dataset-like structure and allowing dataset-like functionality.
"""

import collections
import numpy as np
from copy import deepcopy

from .datatypes import FloatDataType, StringDataType, ArrayDataType, ListDataType
from .tree import TreeSchema, ForkNode, ChildNode


class TreeRow(object):
    """
    The base class of a single row in the :class:`treemodel.datamodel.base.TreeDataSet`. The row contains a single tree
    and all their input data as well.

    :param input_row: Input data in the raw python format (usually dictionaries).
    :type input_row: dict

    :param schema: Either :const:`None` or :class:`treemodel.datamodel.tree.TreeSchema` object specifying the input_row types. In case the schema is None, the schema will be automatically inferred from the ``input_row`` variable.
    :type schema: TreeSchema or None

    :ivar schema: Contains the schema for the particular input data.
    :ivar row: Contains the built input data by the input schema.

    :vartype schema: TreeSchema or None
    :vartype row: dict
    """

    def __init__(self, input_row, schema=None):
        """
        Initialize the TreeRow object.
        """
        self.schema = None
        self.row = None

        if schema is None:
            self.set_schema(self.infer_schema(input_row))
        else:
            self.set_schema(schema)

    def build_row(self, input_row, method):
        """
        Construct TreeRow object from the input_row and specified schema.

        :param input_row: Input data in a tree-like structured format (JSON, XML).
        :param method: Specification of which implementation of the treemodel library should be used. The options are ``python`` and ``numpy``, where ``numpy`` is the default and technically more optimal choice.

        :type input_row: dict
        :type method: str

        :returns: The instance of the TreeRow object with attribute :attr:`row`, where the input data were transformed into a tree object and stored.
        :rtype: TreeRow
        """
        self.row = self.build_tree(input_row, method)
        return self

    def get_schema(self):
        """
        Gets the schema which is currently set.

        :raises: :class:`AttributeError` in case the input schema is not set for this object.

        :return: The current schema of the ``TreeRow`` instance.
        :rtype: TreeSchema
        """
        if self.schema is None:
            raise AttributeError("The schema for the TreeRow is missing!")

        return deepcopy(self.schema)

    def set_schema(self, schema):
        """
        Sets the input schema for the TreeRow object.

        :param schema: Input schema to be set for the instance of the ``TreeRow`` class.
        :type schema: TreeSchema

        :raises: :class:`AttributeError` In case the input variable ``schema`` is not an instance of :class:`treemodel.datamodel.tree.TreeSchema`.

        :return: The same instance of the TreeRow object with updated schema.
        :rtype: TreeRow
        """
        if not isinstance(schema, TreeSchema):
            raise AttributeError("The schema for the row has to be of TreeSchema instance!")
        else:
            self.schema = schema

        return self

    def build_tree(self, input_row, method):
        """
        This method functions as a main method to convert input data into the particular format specified by the schema
        of the class itself.

        :param input_row: Input tree-like structured data (e.g. JSON, XML).
        :param method: Specification of which implementation of the treemodel library should be used. The options are ``python`` and ``numpy``, where ``numpy`` is the default and technically more optimal choice.

        :type input_row: dict
        :type method: str

        :raises: :class:`RuntimeError` in case the method is unknown.

        :return: Converted input data into a dictionary object with the correct type specified by the schema. In case the schema specifies entries which are missing, the dictionary will contain the entries with ``NaN`` values.
        :rtype: dict
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
    def _assert_transformation_possible(input_children, base_fork):
        """
        Helper method which checks that the each name on the list of input names appears only once in the whole
        forking branch.

        :param input_children: Collection of names.
        :param base_fork: Branch to investigate.

        :type input_children: list(str)
        :type base_fork: ForkNode

        :raises: :class:`RuntimeError` in case the transformation is not possible.

        :return: None
        """
        for input_data_child in input_children:
            found_children = base_fork.find_child_in_any_branch(name=input_data_child, as_fork=False, as_copy=True)

            if len(found_children) > 1:
                raise RuntimeError(
                    "Unable to transform input data to the new tree shape due to non-uniqueness of the node '{}'".
                        format(input_data_child))

    def _transform_fork_value(self, input_value, subfork, method):
        """
        Helper method which transforms the given input value by a given fork schema by the specified method.

        :param input_value: Input data to be transformed.
        :param subfork: Schema the transformed data should be following.
        :param method: Specification of the method to use to build the output data, either ``'python'`` or ``'numpy'``.

        :type input_value: dict or None
        :type subfork: ForkNode
        :type method: str

        :raises: :class:`ValueError` in case the given method is unknown.

        :return: Transformed input data to the given schema. In case of ``None`` in the input data, missing value will be given instead.
        :rtype: dict
        """
        if input_value is not None:
            return self.transform_tree(input_value, subfork, method)
        else:
            if method == 'numpy':
                return subfork.get_data_type().build_numpy_value(subfork.get_data_type().numpy_na_value)
            elif method == 'python':
                return subfork.get_data_type().build_python_value(subfork.get_data_type().python_na_value)
            else:
                raise ValueError("Unknown method received '{}'.".format(method))

    @staticmethod
    def _transform_child_value(input_value, subleaf, method):
        """
        Helper method which transforms the given input value by a given leaf schema by the specified method.

        :param input_value: Input data to be transformed.
        :param subfork: Schema the transformed data should be following.
        :param method: Specification of the method to use to build the output data, either ``'python'`` or ``'numpy'``.

        :type input_value: any or None
        :type subfork: ChildNode
        :type method: str

        :raises: :class:`ValueError` in case the given method is unknown.

        :return: Transformed input data to the given schema. In case of ``None`` in the input data, missing value will be given instead.
        :rtype: any
        """
        if method == 'numpy':
            if input_value is not None:
                return subleaf.get_data_type().build_numpy_value(input_value)
            else:
                return subleaf.get_data_type().build_numpy_value(subleaf.get_data_type().numpy_na_value)
        elif method == 'python':
            if input_value is not None:
                return subleaf.get_data_type().build_python_value(input_value)
            else:
                return subleaf.get_data_type().build_python_value(subleaf.get_data_type().python_na_value)
        else:
            raise ValueError("Unknown method received '{}'.".format(method))

    def transform_tree(self, input_data, base_fork, method):
        """
        This method transforms the input data into the wanted shape specified by the input fork.

        :param input_data: Input data in a tree-like format.
        :param base_fork: Specified schema via the fork node, which is to be followed and expected in the output transformation.
        :param method: Specifies the method of transformation to the new schema, either ``'numpy'`` or ``'python'``.

        :type input_data: dict
        :type base_fork: ForkNode
        :type method: str

        :return: Input data transformed by the specified schema.
        :rtype: dict
        """
        if not isinstance(input_data, dict):
            raise ValueError("Input data are in the incorrect format!")

        if not base_fork.is_fork():
            raise ValueError("Input schema is not in the correct format!")

        input_data_children = input_data.keys()
        schema_children = base_fork.get_children_names()
        output_data = {}

        self._assert_transformation_possible(input_data_children, base_fork)

        for schema_child in schema_children:
            schema_child_node = base_fork.find_child(schema_child)

            if schema_child in input_data_children:
                input_data_value = input_data[schema_child]
            else:
                input_data_value = None

            if schema_child_node.is_fork():
                if not isinstance(input_data_value, dict) or input_data_value is not None:
                    raise RuntimeError(
                        "Unable to transform input data to the new tree shape, a single value cannot be transformed into a fork '{}'.".format(
                            schema_child))

                output_data[schema_child] = self._transform_fork_value(input_data_value, schema_child_node, method)

            elif schema_child_node.is_child():
                if isinstance(input_data_value, dict):
                    raise RuntimeError(
                        "Unable to transform input data to the new tree shape, cannot merge forked values into a single value '{}'".format(
                            schema_child))

                output_data[schema_child] = self._transform_child_value(input_data_value, schema_child_node, method)
            else:
                raise NotImplementedError("Applying new schema to a custom subclass of Node is not implemented.")

        return output_data

    def apply_schema(self, method='numpy'):
        """
        This method applies the schema set for the instance of the TreeRow and uses it to transform the input data
        stored in the ``row`` attribute to the schema stored in the ``schema`` attribute.

        :param method: Specifies the method of transformation to the new schema, either ``'numpy'`` or ``'python'``.
        :type method: str

        :return: The same instance of the TreeRow with input data reshaped by the current schema.
        :rtype: TreeRow
        """
        if self.row is None:
            raise AttributeError("The input data are missing! Cannot apply schema to missing data!")

        self.row = self.transform_tree(self.row, self.get_schema().base_fork_node, method)

        return self

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
        This method infers the schema based on the input tree-like structured datum. It is being invoked every time
        the input schema for a particular data input is missing. The inference of the schema proceeds in these steps:
            1. iterate through each item of the tree and do the following
                a. if the item is float or can be converted to a float, set the data type to ``float``.
                b. if the item is a list, then determine whether the inferred data types of the inner elements of the list are the same. If so, set the data type to an ``array``, otherwise set it to the ``list``.
                c. if the item is a dictionary, create a new fork node and perform the same procedure (recursively) on the elements of the dictionary.
                d. otherwise set the data type of the item to ``string``.
            2. sort the order of the items alphabetically and return them as a single ``fork``.

        :param input_dict: Tree-like structured input data (e.g. JSON, XML).
        :param initial_name: Name for the initial (root) fork, which will be always established.

        :type input_dict: dict
        :type initial_name: str

        :return: Inferred ``TreeSchema`` for the input tree data.
        :rtype: TreeSchema
        """
        if not isinstance(input_dict, dict):
            input_dict = dict(input_dict)

        return TreeSchema(base_fork_node=self._infer_fork_type(input_dict, initial_name, 1))


class TreeDataSet(object):
    """
    Main class which provides dataset-like functionality for input tree data.

    In the current implementation the ``TreeDataSet`` provides this functionality:
        - :meth:`uniformize_schema` finds and sets a uniformized schema for the dataset.

    :param input_rows: A collection of built :class:`treemodel.datamodel.base.TreeRow` or python dictionaries containing the input tree-structured like data (e.g. JSON, XML).
    :param schema: Can be either ``None``, or it can be a ``TreeSchema`` specifying the schema for every row, or it can also be a collection of ``TreeSchema`` specifying the schema for each input row in order. In case the schema is None, each row will infer the schema automatically.
    :param method: Specifies the method to build each row, either ``'python'`` or ``'numpy'``.

    :type input_rows: list(TreeRow) or list(dict)
    :type schema: None or TreeSchema or list(TreeSchema)
    :type method: str

    :ivar data: Collection of the input data already in the transformed format of the :class:`treemodel.datamodel.base.TreeRow` object.
    :vartype data: np.ndarray(TreeRow)
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
        Finds and sets a uniformized version of the schema for each row of the ``TreeDataSet``.
        The uniformized version can be found via 2 methods:
            - ``'intersection'``: Performs a schema intersection over all of the schemas of each row.
            - ``'union'``: Performs a schema union over all of the schemas of each each row.
            - ``'fixed'``: Sets a given schema on each row.
        When the schema is set on each row, the input data will get correctly restructured and rebuilt by the new schema.

        :param method: Specifies the method of uniformization. Currently only ``'intersection'``, ``'union'`` and ``'fixed'`` are supported. With value ``'fixed'`` additional parameter ```schema``` has to be specified.
        :param schema: Specifies the schema to be set in case of the ``'fixed'`` method on each row.

        :type method: str
        :type schema: TreeSchema or None

        :raises: :class:`ValueError` in case the method is unknown or in case the parameter schema is missing for method ``'fixed'``.

        :return: The same instance of the ``TreeDataSet`` with new schema applied to each row.
        :rtype: TreeDataSet
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
