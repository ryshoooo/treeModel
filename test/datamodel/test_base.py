from unittest import TestCase
import os
import numpy as np
import json
from copy import copy

import test.datamodel.testdata as td
from test.datamodel.testdata.data_repo import DataGenerator
from treemodel.datamodel.base import TreeRow, TreeDataSet
from treemodel.datamodel.datatypes import DateDataType, StringDataType, FloatDataType, ListDataType, ArrayDataType
from treemodel.datamodel.tree import ChildNode, TreeSchema, ForkNode, TreeDataType


class TestTreeRow(TestCase):
    """
    Test class for the TreeRow.
    """

    def test_print(self):
        input_row, expected_output = DataGenerator.simple_dict_for_print_v1()
        self.assertEqual(str(TreeRow(input_row).schema), expected_output)

        input_row, expected_output = DataGenerator.simple_dict_for_print_v2()
        self.assertEqual(str(TreeRow(input_row).schema), expected_output)

    def test_build_row(self):
        tr = TreeRow({'foo': 12})
        self.assertTrue(tr.row is None)
        tr.build_row({}, method='numpy')
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
        output_row = tr.build_tree(input_row, method='numpy')

        self.assertEqual(input_row, output_row)

        # Case 2
        input_row = {'foo': "2018-01-01", 'foo2': [1, 2, 3]}
        tr = TreeRow(input_row)
        output_row = tr.build_tree(input_row, method='numpy')

        self.assertEqual(input_row['foo'], output_row['foo'])
        self.assertTrue((input_row['foo2'] == output_row['foo2']).all())

        # Case 3
        input_row = {"level1-float": 12.2,
                     "level1-list": ["s", 2],
                     'level1-fork': {'level2-string': 'wrq2',
                                     'level2-array': [{"array_tree_0": 0, "array_tree_1": "sd"}, {"b": 1}]},
                     "level1": "OK"}
        tr = TreeRow(input_row)
        output_row = tr.build_tree(input_row, method='numpy')

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

        child_empty_list = tr._infer_element(value=[], name="empty_list", current_level=1, within_array=False)
        self.assertTrue(isinstance(child_empty_list, ChildNode))
        self.assertTrue(isinstance(child_empty_list.get_data_type(), ArrayDataType))
        self.assertTrue(isinstance(child_empty_list.get_data_type().element_data_type, StringDataType))

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
        input_dict, expected_output = DataGenerator.sample_dict_for_test_schema_v1()

        tr = TreeRow(input_dict)
        self.assertEqual(str(expected_output), str(tr.infer_schema(input_dict)))


class TreeDataSetTestCase(TestCase):
    """
    Configurable class for tree dataset testcases.
    """

    @staticmethod
    def get_test_data_path():
        return os.path.abspath(td.__file__).replace("__init__.py", "")

    @staticmethod
    def generate_json_data_same_schema(file_path, num=100):
        with open(file_path, "w") as fp:
            for num_line in range(num):
                to_dump = DataGenerator.base_dict_json_same_schema()
                if num_line == num - 1:
                    fp.write(json.dumps(to_dump))
                else:
                    fp.write(json.dumps(to_dump) + "\n")

    @staticmethod
    def load_json_file(file_path):
        with open(file_path, "r") as fp:
            res = [json.loads(line) for line in fp]
        return res

    def get_json_data_same_schema(self, overwrite=False):
        test_data_path = self.get_test_data_path() + "test_data_same_schema.json"

        if os.path.exists(test_data_path) and not overwrite:
            return self.load_json_file(test_data_path)
        else:
            self.generate_json_data_same_schema(test_data_path)
            return self.load_json_file(test_data_path)

    def _get_schema_from_dict(self, d, key, level):
        sorted_children_names = sorted(d.keys())
        children = []
        for name in sorted_children_names:
            if isinstance(d[name], dict):
                children.append(self._get_schema_from_dict(d[name], name, level + 1).base_fork_node)
            else:
                children.append(ChildNode(name=name, data_type=d[name]))

        return TreeSchema(base_fork_node=ForkNode(name=key, children=children, level=level))

    def get_schema_for_json_data_same_schema(self):
        d_data_types = DataGenerator.base_dict_json_same_schema_types()

        return self._get_schema_from_dict(d_data_types, "base", level=1)

    def _assert_arrays(self, arr1, arr2):
        if isinstance(arr1, (list, tuple)) and isinstance(arr2, (list, tuple)):
            return arr1 == arr2
        elif isinstance(arr1, (list, tuple)) and isinstance(arr2, np.ndarray):
            if arr2.dtype.type == np.void:
                return arr1 == list(arr2[0])
            else:
                return (arr1 == arr2).all()
        elif isinstance(arr1, np.ndarray):
            return self._assert_arrays(arr2, arr1)
        else:
            raise RuntimeError("Incompatible input types")

    def _assert_equal_dictionaries(self, d1, d2):
        entries_equal = []
        for key, value in d1.items():
            if key not in d2.keys():
                entries_equal.append(False)
            elif isinstance(value, dict):
                entries_equal.append(self._assert_equal_dictionaries(value, d2[key]))
            elif isinstance(value, (list, tuple, np.ndarray)):
                entries_equal.append(self._assert_arrays(value, d2[key]))
            else:
                entries_equal.append(value == d2[key])

        return all(entries_equal)


class TestTreeDataSet(TreeDataSetTestCase):
    """
    Test class for tree dataset.
    """

    def test___init__(self):
        # Same schema
        tds = TreeDataSet(input_rows=self.get_json_data_same_schema())
        first, *rest = [x.schema for x in tds.data]
        self.assertTrue(all([first == x for x in rest]))

    def test__get_tree_row(self):
        data = self.get_json_data_same_schema()[0]

        # Case 1: Dictionary + no schema
        expected_schema = self.get_schema_for_json_data_same_schema()
        tr = TreeDataSet._get_tree_row(input_row=data, schema=None, method='numpy')
        self.assertTrue(isinstance(tr, TreeRow))
        self.assertEqual(expected_schema, tr.schema)
        self._assert_equal_dictionaries(data, tr.row)

        # Case 2: Dictionary + single schema
        expected_schema = self.get_schema_for_json_data_same_schema()
        expected_schema = expected_schema.set_data_type('level1-date',
                                                        DateDataType(resolution='D', format_string='%Y-%m-%d'))
        expected_schema = expected_schema.set_data_type('level1-fork/level2-date',
                                                        DateDataType(resolution='D', format_string='%Y-%m-%d'))
        schema = tr.get_schema()
        schema = schema.set_data_type('level1-date', DateDataType(resolution='D', format_string='%Y-%m-%d'))
        schema = schema.set_data_type('level1-fork/level2-date', DateDataType(resolution='D', format_string='%Y-%m-%d'))

        tr = TreeDataSet._get_tree_row(input_row=data, schema=schema, method='numpy')

        self.assertTrue(isinstance(tr, TreeRow))
        self.assertEqual(expected_schema, tr.schema)
        self._assert_equal_dictionaries(data, tr.row)

        # Case 3: TreeRow + no schema
        tr = TreeRow(input_row=data).build_row(input_row=data, method='numpy')
        expected_schema = self.get_schema_for_json_data_same_schema()
        tr = TreeDataSet._get_tree_row(input_row=tr, schema=None, method='numpy')
        self.assertTrue(isinstance(tr, TreeRow))
        self.assertEqual(expected_schema, tr.schema)
        self._assert_equal_dictionaries(data, tr.row)

        # Case 4: TreeRow + schema
        tr = TreeRow(input_row=data).build_row(input_row=data, method='numpy')

        expected_schema = self.get_schema_for_json_data_same_schema()
        expected_schema = expected_schema.set_data_type('level1-date',
                                                        DateDataType(resolution='D', format_string='%Y-%m-%d'))
        expected_schema = expected_schema.set_data_type('level1-fork/level2-date',
                                                        DateDataType(resolution='D', format_string='%Y-%m-%d'))
        schema = tr.get_schema()
        schema = schema.set_data_type('level1-date', DateDataType(resolution='D', format_string='%Y-%m-%d'))
        schema = schema.set_data_type('level1-fork/level2-date', DateDataType(resolution='D', format_string='%Y-%m-%d'))

        tr = TreeDataSet._get_tree_row(input_row=tr, schema=schema, method='numpy')

        self.assertTrue(isinstance(tr, TreeRow))
        self.assertEqual(expected_schema, tr.schema)
        self._assert_equal_dictionaries(data, tr.row)

    def test_uniformize_intersection(self):
        tds = TreeDataSet(input_rows=self.get_json_data_same_schema())

        tds_after = copy(tds).uniformize_schema('intersection')
