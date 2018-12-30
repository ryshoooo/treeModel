from unittest import TestCase
import os
import numpy as np
import json
from copy import copy
from datetime import datetime

import tests.datamodel.testdata as td
from tests.datamodel.testdata.data_repo import DataGenerator

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
        tr.build_row({'foo': 1}, method='numpy')
        self.assertTrue(tr.row is not None)
        self.assertEqual(tr.row, {'foo': 1})

        tr = TreeRow({'foo': 12})
        self.assertTrue(tr.row is None)
        tr.build_row({'foo': 1}, method='python')
        self.assertTrue(tr.row is not None)
        self.assertEqual(tr.row, {'foo': 1})

        tr = TreeRow({'foo': 12})
        self.assertTrue(tr.row is None)
        with self.assertRaises(RuntimeError):
            tr.build_row({'foo': 1}, method='no')

    def test_get_schema(self):
        tr = TreeRow({'foo': "2018-01-01"})
        self.assertTrue(isinstance(tr.get_schema(), TreeSchema))
        self.assertTrue("foo" in tr.get_schema().base_fork_node.get_children_names())

        new_schema = TreeSchema(
            base_fork_node=ForkNode(name='base', children=[
                ChildNode(name='foo-new', data_type=DateDataType(resolution='D', format_string="%Y-%m-%d"))]))
        tr.set_schema(new_schema)
        self.assertTrue(isinstance(tr.get_schema(), TreeSchema))
        self.assertNotIn("foo", tr.get_schema().base_fork_node.get_children_names())
        self.assertIn("foo-new", tr.get_schema().base_fork_node.get_children_names())
        self.assertEqual(tr.get_schema(), new_schema)

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

        output_row = tr.build_tree(input_row, method='python')
        self.assertEqual(input_row, output_row)

        with self.assertRaises(RuntimeError):
            tr.build_tree(input_row, method='no')

        # Case 2
        input_row = {'foo': "2018-01-01", 'foo2': [1, 2, 3]}
        tr = TreeRow(input_row)

        output_row = tr.build_tree(input_row, method='numpy')
        self.assertEqual(input_row['foo'], output_row['foo'])
        self.assertTrue((input_row['foo2'] == output_row['foo2']).all())

        output_row = tr.build_tree(input_row, method='python')
        self.assertEqual(input_row['foo'], output_row['foo'])
        self.assertTrue((input_row['foo2'] == output_row['foo2']))

        with self.assertRaises(RuntimeError):
            tr.build_tree(input_row, method='no')

        output_row = tr.build_tree({'foo': "something"}, method='python')
        self.assertEqual("something", output_row['foo'])
        self.assertEqual(len(output_row['foo2']), 0)

        # Case 3
        input_row = {"level1-float": 12.2,
                     "level1-list": ["s", 2],
                     'level1-fork': {'level2-string': 'wrq2',
                                     'level2-array': [{"array_tree_0": 0, "array_tree_1": "sd"}, {"b": 1}]},
                     "level1": "OK",
                     "level1-array": [1, 2, 3, 4]}
        tr = TreeRow(input_row)

        output_row = tr.build_tree(input_row, method='numpy')
        self.assertEqual(input_row['level1-float'], output_row['level1-float'])
        self.assertEqual(input_row['level1'], output_row['level1'])
        self.assertEqual(input_row['level1-list'], list(output_row['level1-list'][0]))
        self.assertEqual(input_row['level1-fork']['level2-string'], output_row['level1-fork']['level2-string'])
        self.assertEqual(output_row['level1-fork']['level2-array']['0'][0], {"array_tree_0": 0, "array_tree_1": "sd"})
        self.assertEqual(output_row['level1-fork']['level2-array']['1'][0], {"b": 1})

        output_row = tr.build_tree(input_row, method='python')
        self.assertEqual(input_row['level1-float'], output_row['level1-float'])
        self.assertEqual(input_row['level1'], output_row['level1'])
        self.assertEqual(input_row['level1-list'], output_row['level1-list'])
        self.assertEqual(input_row['level1-fork']['level2-string'], output_row['level1-fork']['level2-string'])
        self.assertEqual(output_row['level1-fork']['level2-array'][0], {"array_tree_0": 0, "array_tree_1": "sd"})
        self.assertEqual(output_row['level1-fork']['level2-array'][1], {"b": 1})

        out_python = tr.build_tree({}, 'python')
        exp_out_python = {'level1': 'None', 'level1-array': [], 'level1-float': None,
                          'level1-fork': {'level2-array': [{'array_tree_0': None, 'array_tree_1': 'None'}, {'b': None}],
                                          'level2-string': 'None'}, 'level1-list': ['None', None]}
        self.assertEqual(out_python, exp_out_python)
        out_numpy = str(tr.build_tree({}, 'numpy'))
        exp_out_numpy = str({'level1': 'nan', 'level1-array': np.array([], dtype=np.float64), 'level1-float': np.nan,
                             'level1-fork': {
                                 'level2-array': np.array(
                                     [({'array_tree_0': np.nan, 'array_tree_1': 'nan'}, {'b': np.nan})],
                                     dtype=[('0', 'O'), ('1', 'O')]), 'level2-string': 'nan'},
                             'level1-list': np.array([('nan', np.nan)], dtype=[('0', '<U128'), ('1', '<f8')])})
        self.assertEqual(out_numpy, exp_out_numpy)

    def test__assert_transformation_possible(self):
        fork1 = ForkNode('base', [ChildNode('c1', StringDataType()), ChildNode('c2', FloatDataType()),
                                  ForkNode('f1', [ChildNode('c2', DateDataType())])])

        with self.assertRaises(RuntimeError):
            TreeRow._assert_transformation_possible(['c2'], fork1)
        with self.assertRaises(RuntimeError):
            TreeRow._assert_transformation_possible(['c1', 'c2'], fork1)
        with self.assertRaises(RuntimeError):
            TreeRow._assert_transformation_possible(['f1', 'c1', 'c2'], fork1)

        TreeRow._assert_transformation_possible(['c1'], fork1)
        TreeRow._assert_transformation_possible(['c1', 'f1'], fork1)

    def test__transform_child_value(self):
        # Case 1
        value1 = '120.28'
        leaf1 = ChildNode('case1', FloatDataType())

        self.assertEqual(float(value1), TreeRow._transform_child_value(value1, leaf1, 'numpy'))
        self.assertEqual(float(value1), TreeRow._transform_child_value(value1, leaf1, 'python'))
        with self.assertRaises(ValueError):
            TreeRow._transform_child_value(value1, leaf1, 'no')

        # Case 2
        value2 = 40
        leaf2 = ChildNode('case2', StringDataType())

        self.assertEqual(str(value2), TreeRow._transform_child_value(value2, leaf2, 'numpy'))
        self.assertEqual(str(value2), TreeRow._transform_child_value(value2, leaf2, 'python'))
        with self.assertRaises(ValueError):
            TreeRow._transform_child_value(value2, leaf2, 'no')

        # Case 3
        value3 = '2018-01-04'
        leaf3 = ChildNode('case3', DateDataType(resolution='D', format_string="%Y-%m-%d"))

        self.assertEqual(np.datetime64(value3), TreeRow._transform_child_value(value3, leaf3, 'numpy'))
        self.assertEqual(datetime.strptime(value3, "%Y-%m-%d"), TreeRow._transform_child_value(value3, leaf3, 'python'))
        with self.assertRaises(ValueError):
            TreeRow._transform_child_value(value3, leaf3, 'no')

        # Case 4
        value4 = None

        self.assertTrue(np.isnan(TreeRow._transform_child_value(value4, leaf1, 'numpy')))
        self.assertTrue(TreeRow._transform_child_value(value4, leaf1, 'python') is None)
        self.assertEqual(TreeRow._transform_child_value(value4, leaf2, 'numpy'), 'nan')
        self.assertEqual(TreeRow._transform_child_value(value4, leaf2, 'python'), 'None')
        self.assertTrue(np.isnat(TreeRow._transform_child_value(value4, leaf3, 'numpy')))
        self.assertEqual(TreeRow._transform_child_value(value4, leaf3, 'python'), '')

    def test_transform_tree(self):
        input_data_1 = {"l1-f": "120.9", "l1-s": 34, "l1-d": "2018-01-04",
                        "f": {"l2-f": "-120.9", "l2-s": 'YES', "l2-a": ["2018-01-04"]}}
        output_data_1_exp = {"l1-f": 120.9, "l1-s": "34", "l1-d": np.datetime64("2018-01-04"),
                             "f": {"l2-f": -120.9, "l2-s": 'YES', "l2-a": [np.datetime64("2018-01-04")],
                                   'l2-missing': 'nan'}}
        fork_1 = ForkNode('base', [
            ChildNode('l1-f', FloatDataType()),
            ChildNode('l1-s', StringDataType()),
            ChildNode('l1-d', DateDataType(resolution='D', format_string="%Y-%m-%d")),
            ForkNode('f', [
                ChildNode('l2-f', FloatDataType()),
                ChildNode('l2-s', StringDataType()),
                ChildNode('l2-a', ArrayDataType(DateDataType(resolution='D', format_string="%Y-%m-%d"))),
                ChildNode('l2-missing', StringDataType())
            ])
        ])

        tr = TreeRow(input_data_1)
        self.assertEqual(tr.transform_tree(input_data_1, fork_1, 'numpy'), output_data_1_exp)

        input_data_2 = {'f': {'float': 20}}
        fork_2 = ForkNode('base', [ChildNode('f', FloatDataType())])

        with self.assertRaises(RuntimeError):
            tr = TreeRow(input_data_2)
            tr.transform_tree(input_data_2, fork_2, 'numpy')

        input_data_3 = {'f': 20}
        fork_3 = ForkNode('base', [ForkNode('f', [ChildNode('float', FloatDataType())])])

        with self.assertRaises(RuntimeError):
            tr = TreeRow(input_data_3)
            tr.transform_tree(input_data_3, fork_3, 'numpy')

    def test_apply_schema(self):
        # Case 1
        input_data_1 = {"l1-f": "120.9", "l1-s": 34, "l1-d": "2018-01-04",
                        "f": {"l2-f": "-120.9", "l2-s": 'YES', "l2-a": ["2018-01-04"]}}
        output_data_1_exp = {"l1-f": 120.9, "l1-s": "34.0", "l1-d": np.datetime64("2018-01-04"),
                             "f": {"l2-f": -120.9, "l2-s": 'YES', "l2-a": [np.datetime64("2018-01-04")],
                                   'l2-missing': 'nan'}}
        fork_1 = ForkNode('base', [
            ChildNode('l1-f', FloatDataType()),
            ChildNode('l1-s', StringDataType()),
            ChildNode('l1-d', DateDataType(resolution='D', format_string="%Y-%m-%d")),
            ForkNode('f', [
                ChildNode('l2-f', FloatDataType()),
                ChildNode('l2-s', StringDataType()),
                ChildNode('l2-a', ArrayDataType(DateDataType(resolution='D', format_string="%Y-%m-%d"))),
                ChildNode('l2-missing', StringDataType())
            ])
        ])

        tr_1 = TreeRow(input_data_1)
        schema_1 = TreeSchema(base_fork_node=fork_1)

        assert tr_1.row is None
        tr_1 = tr_1.build_row(input_data_1, 'numpy')

        self.assertNotEqual(tr_1.row, output_data_1_exp)
        self.assertNotEqual(tr_1.get_schema(), schema_1)
        tr_1 = tr_1.set_schema(schema_1)
        tr_1 = tr_1.apply_schema('numpy')
        self.assertEqual(tr_1.row, output_data_1_exp)

        # Case 2
        input_data_2 = {'f': {'float': 20}}
        fork_2 = ForkNode('base', [ChildNode('f', FloatDataType())])

        tr_2 = TreeRow(input_data_2)
        schema_2 = TreeSchema(base_fork_node=fork_2)

        assert tr_2.row is None
        tr_2 = tr_2.build_row(input_data_2, 'numpy')

        self.assertNotEqual(tr_2.get_schema(), schema_2)

        tr_2 = tr_2.set_schema(schema_2)
        with self.assertRaises(RuntimeError):
            tr_2.apply_schema('numpy')

        # Case 3
        input_data_3 = {'f': 20}
        fork_3 = ForkNode('base', [ForkNode('f', [ChildNode('float', FloatDataType())])])

        tr_3 = TreeRow(input_data_3)
        schema_3 = TreeSchema(base_fork_node=fork_3)

        assert tr_3.row is None
        tr_3 = tr_3.build_row(input_data_3, 'numpy')

        self.assertNotEqual(tr_3.get_schema(), schema_3)

        tr_3 = tr_3.set_schema(schema_3)
        with self.assertRaises(RuntimeError):
            tr_3.apply_schema('numpy')

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

        # Case 3
        self.assertEqual(tr._infer_fork_type({}, 'base', 1), ForkNode('base', []))
        self.assertNotEqual(tr._infer_fork_type({}, 'base', 2), ForkNode('base', []))

    def test_infer_schema(self):
        input_dict, expected_output = DataGenerator.sample_dict_for_test_schema_v1()
        tr = TreeRow(input_dict)
        self.assertEqual(expected_output, tr.infer_schema(input_dict))


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
