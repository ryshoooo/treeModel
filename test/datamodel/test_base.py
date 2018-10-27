from unittest import TestCase
from src.datamodel.base import TreeRow
from src.datamodel.datatypes import DateDataType, StringDataType, ChildNode, TreeSchema, ForkNode, FloatDataType, \
    ListDataType, ArrayDataType, TreeDataType


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
        # Case 1
        input_row = {'foo': "2018-01-01"}
        tr = TreeRow(input_row)
        output_row = tr.build_tree(input_row)

        self.assertEqual(input_row, output_row)

        # Case 2
        input_row = {'foo': "2018-01-01", 'foo2': [1, 2, 3]}
        tr = TreeRow(input_row)
        output_row = tr.build_tree(input_row)

        self.assertEqual(input_row['foo'], output_row['foo'])
        self.assertTrue((input_row['foo2'] == output_row['foo2']).all())

        # Case 3
        input_row = {"level1-float": 12.2,
                     "level1-list": ["s", 2],
                     'level1-fork': {'level2-string': 'wrq2',
                                     'level2-array': [{"array_tree_0": 0, "array_tree_1": "sd"}, {"b": 1}]},
                     "level1": "OK"}
        tr = TreeRow(input_row)
        output_row = tr.build_tree(input_row)

        self.assertEqual(input_row['level1-float'], output_row['level1-float'])
        self.assertEqual(input_row['level1'], output_row['level1'])
        self.assertEqual(input_row['level1-list'], list(output_row['level1-list'][0]))

    def test__is_float(self):
        self.assertTrue(TreeRow._is_float(2))
        self.assertTrue(TreeRow._is_float(2.2))
        self.assertTrue(TreeRow._is_float(-12.3))
        self.assertFalse(TreeRow._is_float([]))
        self.assertFalse(TreeRow._is_float("sda"))
        self.assertFalse(TreeRow._is_float({'s': 1}))

    def test__infer_element(self):
        tr = TreeRow({'foo': 1})

        child_float = tr._infer_element(value=19.2, name='foo', current_level=1, within_array=False)
        self.assertTrue(isinstance(child_float, ChildNode))
        self.assertTrue(isinstance(child_float.get_data_type(), FloatDataType))

        child_float = tr._infer_element(value="19.2", name='foo', current_level=1, within_array=False)
        self.assertTrue(isinstance(child_float, ChildNode))
        self.assertTrue(isinstance(child_float.get_data_type(), FloatDataType))

        child_string = tr._infer_element(value="klatr", name='foo', current_level=1, within_array=False)
        self.assertTrue(isinstance(child_string, ChildNode))
        self.assertTrue(isinstance(child_string.get_data_type(), StringDataType))

        child_array = tr._infer_element(value=[1, 2, 3, "4"], name='foo', current_level=1, within_array=False)
        self.assertTrue(isinstance(child_array, ChildNode))
        self.assertTrue(isinstance(child_array.get_data_type(), ArrayDataType))

        child_list = tr._infer_element(value=[1, 2, 3, {'foo': 1}], name='foo', current_level=1, within_array=False)
        self.assertTrue(isinstance(child_list, ChildNode))
        self.assertTrue(isinstance(child_list.get_data_type(), ListDataType))

        fork_test = tr._infer_element(value={"foo": 1}, name="base", current_level=1, within_array=False)
        self.assertTrue(isinstance(fork_test, ForkNode))
        self.assertTrue(isinstance(fork_test.get_data_type(), TreeDataType))
        self.assertEqual(fork_test.level, 2)

    def test__infer_fork_type(self):
        tr = TreeRow({'foo': 1})

        # Case 1
        fork_out = tr._infer_fork_type(input_dict={"foo1": 1, "foo2": 2}, key="base", level=1)
        self.assertTrue(isinstance(fork_out, ForkNode))
        for key in ['foo1', 'foo2']:
            self.assertTrue(key in fork_out.get_children_names())
        self.assertTrue(isinstance(fork_out.find_child('foo1').get_data_type(), FloatDataType))
        self.assertTrue(isinstance(fork_out.find_child('foo2').get_data_type(), FloatDataType))

        # Case 2
        fork_out = tr._infer_fork_type(input_dict={"foo1": 1, "foo2": {"arr": [1, 2, 3, "KA"]}}, key="base", level=1)
        self.assertTrue(isinstance(fork_out, ForkNode))
        for key in ['foo1', 'foo2']:
            self.assertTrue(key in fork_out.get_children_names())
        self.assertTrue(isinstance(fork_out.find_child('foo1').get_data_type(), FloatDataType))
        self.assertTrue(isinstance(fork_out.find_child('foo2').get_data_type(), TreeDataType))
        self.assertTrue(isinstance(fork_out.find_child('foo2').find_child('arr').get_data_type(), ListDataType))

    def test_infer_schema(self):
        input_dict = {
            'a': 23,
            'b': {
                'c': "sa",
                'd': [{"s": 1}, 12.3],
                'e': ["a", "b", "c"]
            }
        }

        tr = TreeRow(input_dict)

        expected_output = TreeSchema(base_fork_node=ForkNode(name="base", children=[
            ChildNode(name="a", data_type=FloatDataType()),
            ForkNode(name="b", children=[
                ChildNode(name="c", data_type=StringDataType(longest_string=2)),
                ChildNode(name="d", data_type=ListDataType(element_data_types=[
                    TreeDataType(
                        schema=TreeSchema(base_fork_node=ForkNode(name="d_0", children=[
                            ChildNode(name="s", data_type=FloatDataType())], level=4))),
                    FloatDataType()
                ], level=3)),
                ChildNode(name="e", data_type=ArrayDataType(element_data_type=StringDataType(longest_string=1)))
            ], level=2)
        ], level=1))

        self.assertEqual(str(expected_output), str(tr.infer_schema(input_dict)))
