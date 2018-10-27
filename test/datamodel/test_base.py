from unittest import TestCase
from src.datamodel.base import TreeRow


class TestTreeRow(TestCase):
    """
    Test class for the TreeRow.
    """

    def test_print(self):
        input_row = {"level1-float": 12.2,
                     "level1-list": ["s", 2],
                     'level1-fork': {'level2-string': 'wrq2',
                                     'level2-array': [{"array_tree_0": 0, "array_tree_1": "sd"},  {"b": 1}]},
                     "level1": "OK"}
        print(TreeRow(input_row).schema)

    def test_build_row(self):
        self.fail()

    def test_set_schema(self):
        self.fail()

    def test_build_tree(self):
        self.fail()

    def test__is_float(self):
        self.fail()

    def test__infer_element(self):
        self.fail()

    def test__infer_fork_type(self):
        self.fail()

    def test_infer_schema(self):
        self.fail()
