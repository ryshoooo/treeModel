"""
This module contains base classes and methods for the input data model.
"""
import collections
import numpy as np

from .datatypes import TreeSchema, ForkNode, ChildNode, FloatDataType, StringDataType, ArrayDataType, ListDataType


class TreeRow(object):
    """
    The superclass containing the base tree input row.
    """

    def __init__(self, input_row, schema=None):
        """
        Initialize the TreeRow object.
        :param input_row: Dictionary with input data.
        :param schema: Either None or TreeSchema object speciyfing the input_row types. In case the schema is None,
        the schema will be automatically inferred from the input_row.
        """
        if schema is None:
            self.schema = self.infer_schema(input_row)
        else:
            self.set_schema(schema)

        self.row = None

    def build_row(self, input_row):
        """
        Construct TreeRow object from the input_row and specified schema.
        :param input_row: Dictionary with input data.
        :return: TreeRow object with the input data.
        """
        self.row = self.build_tree(input_row)
        return self

    def get_schema(self):
        """
        Getter method for tree's schema.
        :return: TreeSchema.
        """
        if self.schema is None:
            raise AttributeError("The schema for the TreeRow is missing!")

        return self.schema

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

    def build_tree(self, input_row):
        """
        Method which builds tree from input dictionary.
        :param input_row: Dictionary with the input data.
        :return: Dictionary with the typed data.
        """
        if not isinstance(input_row, dict):
            try:
                input_row = dict(input_row)
            except Exception as e:
                raise RuntimeError("Failed to interpret the input row as dictionary!")

        return self.schema.base_fork_node.build_value(input_row)

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
        :param current_level: Integers specifying the level of the Node in the tree hiearchy.
        :param within_array: Boolean specifying whether this value is from a list
        :return: Node object with specified type and name
        """
        if isinstance(value, dict) and not within_array:
            return self._infer_fork_type(value, name, current_level + 1)
        elif isinstance(value, dict) and within_array:
            return self._infer_fork_type(value, name, current_level + 2)
        elif isinstance(value, list):
            elements = [self._infer_element(value[x], '{}'.format(name, x), current_level, True) for x in
                        range(len(value))]
            elements_types = [x.get_data_type() for x in elements]
            if len(set([str(x) for x in elements_types])) <= 1:
                return ChildNode(name=name, data_type=ArrayDataType(element_data_type=elements_types[0]))
            else:
                elements_types = [elements[x].set_name("{}_{}".format(name, x)).get_data_type()
                                  for x in range(len(elements))]
                return ChildNode(name=name, data_type=ListDataType(element_data_types=elements_types,
                                                                   level=current_level + 1))
        elif isinstance(value, float) or self._is_float(value):
            return ChildNode(name=name, data_type=FloatDataType())
        else:
            return ChildNode(name=name, data_type=StringDataType(longest_string=len(str(value))))

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
            try:
                input_dict = dict(input_dict)
            except Exception as e:
                raise RuntimeError("Failed to interpret the input row as dictionary!")

        return TreeSchema(base_fork_node=self._infer_fork_type(input_dict, initial_name, 1))


class TreeDataSet(object):
    """
    Base class for dataset of trees.
    """

    def __init__(self, input_rows, schema=None):
        """
        Initialize the dataset object.
        :param input_rows: A collection of built TreeRows or python dictionaries.
        :param schema: Either None or TreeSchema specifying the schema for every row or collection of TreeSchemas
            specifying the TreeSchema for each input row in order. In case the schema is None, each row will infer the
            schema automatically.
        """
        if not isinstance(input_rows, (collections.Sequence, np.ndarray)) or isinstance(input_rows, str):
            raise AttributeError("Incorrect format of input rows!")

        for row in input_rows:
            if not isinstance(row, dict) and not isinstance(row, TreeRow):
                raise AttributeError("Input rows have to be of dictionary or tree type!")

        if schema is None:
            self.data = np.array([self._get_tree_row(input_row=row, schema=None) for row in input_rows])
        elif isinstance(schema, TreeSchema):
            self.data = np.array([self._get_tree_row(input_row=row, schema=schema) for row in input_rows])
        elif isinstance(schema, (collections.Sequence, np.ndarray)):
            self.data = np.array([self._get_tree_row(input_row=input_rows[ind], schema=val)
                                  for ind, val in enumerate(schema)])
        else:
            raise AttributeError("Incorrect format of input schema!")

    @staticmethod
    def _get_tree_row(input_row, schema):
        """
        Helper method to get tree row with specified schema.
        :param input_row: Either a python dictionary or built TreeRow.
        :param schema: None or TreeSchema. In case of None, the schema for the row will be inferred automatically.
        :return: TreeRow with built row.
        """
        if isinstance(input_row, TreeRow):
            if schema is None:
                return input_row
            elif isinstance(schema, TreeSchema):
                input_row = input_row.set_schema(schema)
                return input_row.build_row(input_row.row)
            else:
                raise AttributeError("Input schema parameter is not a TreeSchema object!")
        else:
            if schema is None:
                return TreeRow(input_row=input_row).build_row(input_row=input_row)
            elif isinstance(schema, TreeSchema):
                return TreeRow(input_row=input_row, schema=schema).build_row(input_row=input_row)
            else:
                raise AttributeError("Input schema parameter is not a TreeSchema object!")
