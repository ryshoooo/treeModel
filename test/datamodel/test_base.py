from unittest import TestCase
from src.datamodel.base import TreeRow
from src.datamodel.datatypes import DateDataType, StringDataType, ChildNode, TreeSchema, ForkNode


class TestTreeRow(TestCase):
    """
    Test class for the TreeRow.
    """

    def test_print(self):
        input_row = {"level1-float": 12.2,
                     "level1-list": ["s", 2],
                     'level1-fork': {'level2-string': 'wrq2',
                                     'level2-array': [{"array_tree_0": 0, "array_tree_1": "sd"}, {"b": 1}]},
                     "level1": "OK"}

        expected_output = """base(
	level1(StringDataType)
	level1-float(FloatDataType)
	level1-fork(
		level2-array(ListDataType(
			TreeDataType(level2-array_0(
				array_tree_0(FloatDataType)
				array_tree_1(StringDataType)
			              ))
			TreeDataType(level2-array_1(
				b(FloatDataType)
			              ))
		            ))
		level2-string(StringDataType)
	           )
	level1-list(ListDataType(
		StringDataType
		FloatDataType
	            ))
    )"""

        self.assertEqual(str(TreeRow(input_row).schema), expected_output)

        input_row = {"level1-float": 12.2,
                     "level1-list": ["s", 2],
                     'level1-fork': {'level2-string': 'wrq2',
                                     'level2-array': [{"b": 2}, {"b": 1}]},
                     "level1": "OK"}

        expected_output = """base(
	level1(StringDataType)
	level1-float(FloatDataType)
	level1-fork(
		level2-array(ArrayDataType(TreeDataType(level2-array(
				b(FloatDataType)
			            ))))
		level2-string(StringDataType)
	           )
	level1-list(ListDataType(
		StringDataType
		FloatDataType
	            ))
    )"""
        self.assertEqual(str(TreeRow(input_row).schema), expected_output)

    def test_build_row(self):
        tr = TreeRow({'foo': 12})
        self.assertTrue(tr.row is None)
        tr.build_row({})
        self.assertTrue(tr.row is not None)

    def test_set_schema(self):
        tr = TreeRow({'foo': "2018-01-01"})
        self.assertTrue(isinstance(tr.schema.base_fork_node.find_child('foo').get_data_type(), StringDataType))
        new_schema = TreeSchema(
            base_fork_node=ForkNode(name='base', children=[
                ChildNode(name='foo', data_type=DateDataType(resolution='D', format_string="%Y-%m-%d"))]))
        tr.set_schema(new_schema)
        self.assertTrue(isinstance(tr.schema.base_fork_node.find_child('foo').get_data_type(), DateDataType))

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
