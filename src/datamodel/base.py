"""
This module contains base classes and methods for the input data model.
"""

from .datatypes import TreeSchema, ForkNode, ChildNode, FloatDataType, StringDataType, ArrayDataType, ListDataType


class TreeRow(object):
    """
    The superclass containing the base tree input row.
    """

    def __init__(self, input_row, schema=None):
        if schema is None:
            self.schema = self.infer_schema(input_row)
        else:
            self.set_schema(schema)

        self.row = None

    def build_row(self, input_row):
        self.row = self.build_tree(input_row)

    def set_schema(self, schema):
        if not isinstance(schema, TreeSchema):
            raise AttributeError("The schema for the row has to be of TreeSchema instance!")
        else:
            self.schema = schema

    def build_tree(self, input_row):
        if not isinstance(input_row, dict):
            try:
                input_row = dict(input_row)
            except Exception as e:
                raise RuntimeError("Failed to interpret the input row as dictionary!")

        return self.schema.base_fork_node.build_value(input_row)

    @staticmethod
    def _is_float(n):
        try:
            float(n)
            return True
        except (ValueError, TypeError):
            return False

    def _infer_element(self, value, name, current_level, within_array=False):
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
        sorted_children_names = sorted(input_dict.keys())
        children = [self._infer_element(input_dict[name], name, level) for name in sorted_children_names]
        return ForkNode(name=key, children=children, level=level)

    def infer_schema(self, input_dict, initial_name='base'):
        if not isinstance(input_dict, dict):
            try:
                input_dict = dict(input_dict)
            except Exception as e:
                raise RuntimeError("Failed to interpret the input row as dictionary!")

        return TreeSchema(base_fork_node=self._infer_fork_type(input_dict, initial_name, 1))
