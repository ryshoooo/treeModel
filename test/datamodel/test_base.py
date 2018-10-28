from unittest import TestCase
import os
import numpy as np
import json

import test.datamodel.testdata as td
from src.datamodel.base import TreeRow, TreeDataSet
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

    def test_get_schema(self):
        tr = TreeRow({'foo': "2018-01-01"})
        self.assertTrue(isinstance(tr.get_schema(), TreeSchema))
        self.assertTrue("foo" in tr.get_schema().base_fork_node.get_children_names())

        new_schema = TreeSchema(
            base_fork_node=ForkNode(name='base', children=[
                ChildNode(name='foo-new', data_type=DateDataType(resolution='D', format_string="%Y-%m-%d"))]))
        tr.set_schema(new_schema)
        self.assertTrue(isinstance(tr.get_schema(), TreeSchema))
        self.assertTrue("foo" not in tr.get_schema().base_fork_node.get_children_names())
        self.assertTrue("foo-new" in tr.get_schema().base_fork_node.get_children_names())

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


class TestTreeDataSet(TestCase):
    """
    Test class for tree dataset.
    """

    @staticmethod
    def get_test_data_path():
        return os.path.abspath(td.__file__).replace("__init__.py", "")

    @staticmethod
    def generate_json_data_same_schema(file_path, num=100):
        with open(file_path, "w") as fp:
            for num_line in range(num):
                to_dump = {
                    "level1-string": str(np.random.choice(["A", "B", "C", "D", "R"], replace=False)),
                    "level1-float": float(np.random.random()),
                    "level1-date": str(np.random.choice(["{}-04-01".format(year) for year in range(1993, 2019)])),
                    "level1-array_float": [float(x) for x in np.random.random(10)],
                    "level1-array_string": [str(x) for x in
                                            np.random.choice(a=["A", "B", "C", "D", "R"], size=10, replace=True)],
                    "level1-list_float_string": [float(x) for x in np.random.random(5)] +
                                                [str(x) for x in
                                                 np.random.choice(a=["A", "B", "C", "D", "R"], size=5, replace=True)],
                    "level1-fork": {
                        "level2-string": str(np.random.choice(["A", "B", "C", "D", "R"], replace=False)),
                        "level2-float": float(np.random.random()),
                        "level2-date": str(np.random.choice(["{}-04-01".format(year) for year in range(1993, 2019)])),
                        "level2-array_float": [float(x) for x in np.random.random(10)],
                        "level2-array_string": [str(x) for x in
                                                np.random.choice(a=["A", "B", "C", "D", "R"], size=10, replace=True)],
                        "level2-list_float_string": [float(x) for x in np.random.random(5)] +
                                                    [str(x) for x in
                                                     np.random.choice(a=["A", "B", "C", "D", "R"], size=5,
                                                                      replace=True)],
                    },
                    "level1-fork2": {
                        "level2-float": float(np.random.random()),
                        "level2-fork": {
                            "level3-float": float(np.random.random()),
                            "level3-array_tree": [
                                {
                                    "level3-array-float": float(np.random.random()),
                                    "level3-array-string": str(
                                        np.random.choice(["A", "B", "C", "D", "R"], replace=False))
                                } for x in range(10)
                            ],
                            "level3-list_tree": [
                                                    {
                                                        "level3-list-float": float(np.random.random()),
                                                        "level3-list-string": str(
                                                            np.random.choice(["A", "B", "C", "D", "R"], replace=False))
                                                    } for x in range(5)
                                                ] + [
                                                    {
                                                        "level3-list-date": str(np.random.choice(
                                                            ["{}-04-01".format(year) for year in range(1993, 2019)])),
                                                        "level3-list-string": str(
                                                            np.random.choice(["A", "B", "C", "D", "R"], replace=False))
                                                    } for x in range(5)
                                                ]
                        }
                    }
                }
                if num_line == num - 1:
                    fp.write(json.dumps(to_dump))
                else:
                    fp.write(json.dumps(to_dump) + "\n")

    @staticmethod
    def load_json_file(file_path):
        with open(file_path, "r") as fp:
            res = []
            for line in fp:
                d = json.loads(line)
                res.append(d)

        return res

    def get_json_data_same_schema(self, overwrite=False):
        test_data_path = self.get_test_data_path() + "test_data_same_schema.json"

        if os.path.exists(test_data_path) and not overwrite:
            return self.load_json_file(test_data_path)
        else:
            self.generate_json_data_same_schema(test_data_path)
            return self.load_json_file(test_data_path)

    def _get_schema_from_dict(self, d, key):
        sorted_children_names = sorted(d.keys())
        children = []
        for name in sorted_children_names:
            if isinstance(d[name], dict):
                children.append(self._get_schema_from_dict(d[name], name).base_fork_node)
            else:
                children.append(ChildNode(name=name, data_type=d[name]))

        return TreeSchema(base_fork_node=ForkNode(name=key, children=children))

    def get_schema_for_json_data_same_schema(self):
        d_data_types = {
            "level1-string": StringDataType(),
            "level1-float": FloatDataType(),
            "level1-date": StringDataType(),
            "level1-array_float": ArrayDataType(FloatDataType()),
            "level1-array_string": ArrayDataType(StringDataType()),
            "level1-list_float_string": ListDataType([FloatDataType()] * 5 + [StringDataType()] * 5),
            "level1-fork": {
                "level2-string": StringDataType(),
                "level2-float": FloatDataType(),
                "level2-date": StringDataType(),
                "level2-array_float": ArrayDataType(FloatDataType()),
                "level2-array_string": ArrayDataType(StringDataType()),
                "level2-list_float_string": ListDataType([FloatDataType()] * 5 + [StringDataType()] * 5),
            },
            "level1-fork2": {
                "level2-float": FloatDataType(),
                "level2-fork": {
                    "level3-float": FloatDataType(),
                    "level3-array_tree": ArrayDataType(
                        TreeDataType(
                            schema=TreeSchema(
                                base_fork_node=ForkNode(
                                    name="level3-array_tree",
                                    children=[
                                        ChildNode(name="level3-array-float", data_type=FloatDataType()),
                                        ChildNode(name="level3-array-string", data_type=StringDataType())
                                    ]
                                )
                            )
                        )
                    ),
                    "level3-list_tree": ListDataType(
                        [
                            TreeDataType(
                                schema=TreeSchema(
                                    base_fork_node=ForkNode(
                                        name="level3-list_tree_{}".format(x),
                                        children=[
                                            ChildNode(name="level3-list-float", data_type=FloatDataType()),
                                            ChildNode(name="level3-list-string", data_type=StringDataType())
                                        ]
                                    )
                                )
                            )
                            for x in range(0, 5)] + [
                            TreeDataType(
                                schema=TreeSchema(
                                    base_fork_node=ForkNode(
                                        name="level3-list_tree_{}".format(x),
                                        children=[
                                            ChildNode(name="level3-list-date", data_type=StringDataType()),
                                            ChildNode(name="level3-list-string", data_type=StringDataType())
                                        ]
                                    )
                                )
                            )
                            for x in range(5, 10)]
                    )
                }
            }
        }

        return self._get_schema_from_dict(d_data_types, "base")

    def test___init__(self):
        # Same schema
        tds = TreeDataSet(input_rows=self.get_json_data_same_schema())
        self.assertEqual(len(set([str(x.schema) for x in tds.data])), 1)

    def test__get_tree_row(self):
        data = self.get_json_data_same_schema()[0]

        # Case 1: Dictionary + no schema
        expected_schema = self.get_schema_for_json_data_same_schema()
        tr = TreeDataSet._get_tree_row(input_row=data, schema=None)
        self.assertTrue(isinstance(tr, TreeRow))
        self.assertEqual(expected_schema, tr.schema)
