"""
This module contains base classes and methods for the input data model.
"""

from .datatypes import TreeSchema


class BaseTreeRow(object):
    """
    The superclass containing the base tree input row.
    """

    def __init__(self, input_row, schema=None):
        if not isinstance(schema, TreeSchema):
            raise AttributeError("The schema for the row has to be of TreeSchema instance!")

        self.schema = schema
        self.row = self.build_tree(input_row)

    def set_schema(self, schema):
        self.schema = schema

    def build_tree(self, input_row):
        if not isinstance(input_row, dict):
            try:
                input_row = dict(input_row)
            except Exception as e:
                raise RuntimeError("Failed to interpret the input row as dictionary!")

        return [x.get_data_type().build_numpy_value(input_row[x.get_name()]) for x in self.schema.nodes]



