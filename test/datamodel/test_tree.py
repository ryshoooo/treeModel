from unittest import TestCase
from datetime import datetime
import numpy as np
from copy import copy

from src.datamodel.tree import ChildNode, ForkNode, TreeSchema, TreeDataType
from src.datamodel.datatypes import StringDataType, FloatDataType, DateDataType, DataType, ArrayDataType, ListDataType


class TestNode(TestCase):
    """
    Test class for the Node class.
    """

    @staticmethod
    def get_single_string_leaf():
        return ChildNode(name="leaf-string", data_type=StringDataType())

    @staticmethod
    def get_single_float_leaf():
        return ChildNode(name="leaf-float", data_type=FloatDataType())

    @staticmethod
    def get_single_date_leaf(format_string=''):
        return ChildNode(name="leaf-date", data_type=DateDataType(resolution='D', format_string=format_string))

    @staticmethod
    def get_fork_node(format_string="%Y-%m-%d %H:%M:%S.%f"):
        leaf_string = TestNode.get_single_string_leaf()
        leaf_date = TestNode.get_single_date_leaf(format_string=format_string)
        leaf_float = TestNode.get_single_float_leaf()
        return ForkNode(name='test-fork', children=[leaf_string, leaf_date, leaf_float])

    @staticmethod
    def get_random_fork_values():
        return {'leaf-float': str(np.random.random()), 'leaf-date': "2018-01-01 19:45:33.123456",
                "leaf-string": np.random.choice(["a", "b", "c", "d"])}

    def test_is_child(self):
        leaf_string = self.get_single_string_leaf()
        self.assertTrue(leaf_string.is_child())
        leaf_date = self.get_single_date_leaf()
        self.assertTrue(leaf_date.is_child())
        leaf_float = self.get_single_float_leaf()
        self.assertTrue(leaf_float.is_child())

    def test_is_fork(self):
        single_fork = self.get_fork_node()
        self.assertTrue(single_fork.is_fork())

    def test_get_name(self):
        single_fork = self.get_fork_node()
        single_leaf = self.get_single_float_leaf()

        self.assertEqual(single_fork.name, single_fork.get_name())
        self.assertEqual(single_leaf.name, single_leaf.get_name())

    def test_set_name(self):
        single_fork = self.get_fork_node()
        single_leaf = self.get_single_float_leaf()

        self.assertEqual(single_fork.name, single_fork.get_name())
        single_fork.set_name("new_fork_name")
        self.assertEqual(single_fork.name, "new_fork_name")
        self.assertEqual(single_leaf.name, single_leaf.get_name())
        single_leaf.set_name("new_leaf_name")
        self.assertEqual(single_leaf.name, "new_leaf_name")

    def test_get_data_type(self):
        single_fork = self.get_fork_node()
        single_leaf = self.get_single_float_leaf()

        dtp = single_leaf.get_data_type()
        self.assertTrue(isinstance(dtp, FloatDataType))
        dtp = single_fork.get_data_type()
        self.assertTrue(isinstance(dtp, TreeDataType))

    def test_set_data_type(self):
        single_leaf = self.get_single_float_leaf()

        dtp = single_leaf.get_data_type()
        self.assertTrue(isinstance(dtp, FloatDataType))
        single_leaf.set_data_type(DateDataType())
        dtp = single_leaf.get_data_type()
        self.assertTrue(isinstance(dtp, DateDataType))


class TestChildNode(TestCase):
    """
    Test class for ChildNode
    """

    def test_overwrite_child(self):
        single_leaf = TestNode.get_single_float_leaf()
        self.assertTrue(single_leaf.is_child())
        self.assertTrue(single_leaf.children is None)
        self.assertEqual(single_leaf.name, 'leaf-float')
        self.assertTrue(isinstance(single_leaf.data_type, FloatDataType))

        single_leaf = single_leaf.set_name(name='new-leaf')
        single_leaf = single_leaf.set_data_type(data_type=DateDataType(resolution='M'))

        self.assertTrue(single_leaf.is_child())
        self.assertTrue(single_leaf.children is None)
        self.assertEqual(single_leaf.name, 'new-leaf')
        self.assertTrue(isinstance(single_leaf.data_type, DateDataType))

    def test_eq(self):
        single_leaf1 = TestNode.get_single_float_leaf()
        single_leaf2 = TestNode.get_single_float_leaf()
        self.assertEqual(single_leaf1, single_leaf2)
        single_leaf1 = TestNode.get_single_date_leaf()
        single_leaf2 = TestNode.get_single_date_leaf()
        self.assertEqual(single_leaf1, single_leaf2)
        single_leaf1 = TestNode.get_single_string_leaf()
        single_leaf2 = TestNode.get_single_string_leaf()
        self.assertEqual(single_leaf1, single_leaf2)


class TestForkNode(TestCase):
    """
    Test class for ForkNode
    """

    def test_overwrite_children(self):
        single_fork = TestNode.get_fork_node()
        self.assertTrue(single_fork.is_fork())
        self.assertEqual(len(single_fork.children), 3)
        self.assertEqual(single_fork.name, 'test-fork')
        for child in single_fork.children:
            self.assertTrue('leaf' in child.name)

        new_children = [ChildNode(name="new", data_type=DateDataType(resolution='M'))]
        single_fork = single_fork.set_children(children=new_children)

        self.assertTrue(single_fork.is_fork())
        self.assertEqual(len(single_fork.children), 1)
        self.assertEqual(single_fork.name, 'test-fork')
        for child in single_fork.children:
            self.assertTrue('new' in child.name)

        new_children_fail = new_children + [ChildNode(name='new', data_type=FloatDataType())]

        try:
            single_fork.set_children(children=new_children_fail)
        except AttributeError as e:
            self.assertEqual("Children nodes with the same name are not allowed! 'new'", str(e))

    def test_get_children(self):
        single_fork = TestNode.get_fork_node()

        single_fork_children = single_fork.get_children()

        self.assertTrue(len(single_fork_children))
        for ind in range(len(single_fork_children)):
            self.assertTrue(single_fork_children[ind] in single_fork.children)
            self.assertEqual(single_fork_children[ind].name, single_fork.children[ind].name)

    def test_get_children_names(self):
        single_fork = TestNode.get_fork_node()
        single_fork_children_names = single_fork.get_children_names()

        self.assertTrue('leaf-string' in single_fork_children_names)
        self.assertTrue('leaf-float' in single_fork_children_names)
        self.assertTrue('leaf-date' in single_fork_children_names)
        self.assertEqual(single_fork_children_names[0], 'leaf-string')
        self.assertEqual(single_fork_children_names[1], 'leaf-date')
        self.assertEqual(single_fork_children_names[2], 'leaf-float')
        self.assertEqual(len(single_fork_children_names), 3)

    def test_find_child(self):
        single_fork = TestNode.get_fork_node()
        children = single_fork.get_children()

        new_child_1 = ChildNode(name='leaf2-string', data_type=StringDataType())
        new_child_2 = ChildNode(name='leaf2-float', data_type=FloatDataType())
        new_fork = ForkNode(name='level2', children=[new_child_1, new_child_2])

        fork_for_test = ForkNode(name='test_find_child', children=children + [new_fork])
        children = fork_for_test.get_children()

        for ind, name in enumerate(fork_for_test.get_children_names()):
            found = fork_for_test.find_child(name)
            self.assertEqual(children[ind], found)

        try:
            fork_for_test.find_child('nonexistent')
        except RuntimeError as e:
            self.assertEqual("Child 'nonexistent' was not found in 'test_find_child'", str(e))

    def test_build_value_numpy(self):
        single_fork = TestNode.get_fork_node()
        single_fork_values = TestNode.get_random_fork_values()
        single_fork_built_values = single_fork.get_data_type().build_numpy_value(value=single_fork_values)

        self.assertEqual(float(single_fork_values['leaf-float']), single_fork_built_values['leaf-float'])
        self.assertEqual(np.datetime64(single_fork_values['leaf-date']).astype('<M8[D]'),
                         single_fork_built_values['leaf-date'])
        self.assertEqual(single_fork_values['leaf-string'], single_fork_built_values['leaf-string'])

        new_child_1 = ChildNode(name='leaf2-string', data_type=StringDataType())
        new_child_2 = ChildNode(name='leaf2-float', data_type=FloatDataType())
        new_fork = ForkNode(name='level2', children=[new_child_1, new_child_2])
        fork_for_test = ForkNode(name='test_find_child', children=single_fork.get_children() + [new_fork])

        single_fork_values['level2'] = {}
        single_fork_values['level2']['leaf2-string'] = np.random.choice(['q', 'w', 'e', 'r'])
        single_fork_values['level2']['leaf2-float'] = str(np.random.random())
        single_fork_built_values = fork_for_test.get_data_type().build_numpy_value(value=single_fork_values)

        self.assertEqual(float(single_fork_values['leaf-float']), single_fork_built_values['leaf-float'])
        self.assertEqual(np.datetime64(single_fork_values['leaf-date']).astype('<M8[D]'),
                         single_fork_built_values['leaf-date'])
        self.assertEqual(single_fork_values['leaf-string'], single_fork_built_values['leaf-string'])
        self.assertEqual(single_fork_built_values['level2']['leaf2-string'],
                         single_fork_values['level2']['leaf2-string'])
        self.assertEqual(single_fork_built_values['level2']['leaf2-float'],
                         float(single_fork_values['level2']['leaf2-float']))

    def test_build_value_python(self):
        single_fork = TestNode.get_fork_node()
        single_fork_values = TestNode.get_random_fork_values()
        single_fork_built_values = single_fork.get_data_type().build_python_value(value=single_fork_values)

        self.assertEqual(float(single_fork_values['leaf-float']), single_fork_built_values['leaf-float'])
        self.assertEqual(datetime.strptime(single_fork_values['leaf-date'], "%Y-%m-%d %H:%M:%S.%f"),
                         single_fork_built_values['leaf-date'])
        self.assertEqual(single_fork_values['leaf-string'], single_fork_built_values['leaf-string'])

        new_child_1 = ChildNode(name='leaf2-string', data_type=StringDataType())
        new_child_2 = ChildNode(name='leaf2-float', data_type=FloatDataType())
        new_fork = ForkNode(name='level2', children=[new_child_1, new_child_2])
        fork_for_test = ForkNode(name='test_find_child', children=single_fork.get_children() + [new_fork])

        single_fork_values['level2'] = {}
        single_fork_values['level2']['leaf2-string'] = np.random.choice(['q', 'w', 'e', 'r'])
        single_fork_values['level2']['leaf2-float'] = str(np.random.random())
        single_fork_built_values = fork_for_test.get_data_type().build_python_value(value=single_fork_values)

        self.assertEqual(float(single_fork_values['leaf-float']), single_fork_built_values['leaf-float'])
        self.assertEqual(datetime.strptime(single_fork_values['leaf-date'], "%Y-%m-%d %H:%M:%S.%f"),
                         single_fork_built_values['leaf-date'])
        self.assertEqual(single_fork_values['leaf-string'], single_fork_built_values['leaf-string'])
        self.assertEqual(single_fork_built_values['level2']['leaf2-string'],
                         single_fork_values['level2']['leaf2-string'])
        self.assertEqual(single_fork_built_values['level2']['leaf2-float'],
                         float(single_fork_values['level2']['leaf2-float']))

    def test_eq(self):
        single_fork1 = TestNode.get_fork_node()
        single_fork2 = TestNode.get_fork_node()
        self.assertEqual(single_fork1, single_fork2)

        new_child_1 = ChildNode(name='leaf2-string', data_type=StringDataType())
        new_child_2 = ChildNode(name='leaf2-float', data_type=FloatDataType())
        new_fork = ForkNode(name='level2', children=[new_child_1, new_child_2])
        fork_for_test1 = ForkNode(name='test_find_child', children=single_fork1.get_children() + [new_fork])
        new_child_1 = ChildNode(name='leaf2-string', data_type=StringDataType())
        new_child_2 = ChildNode(name='leaf2-float', data_type=FloatDataType())
        new_fork = ForkNode(name='level2', children=[new_child_1, new_child_2])
        fork_for_test2 = ForkNode(name='test_find_child', children=single_fork2.get_children() + [new_fork])
        self.assertEqual(fork_for_test1, fork_for_test2)


class TestTreeSchema(TestCase):
    """
    Test class for TreeSchema
    """

    def test_create_dummy_nan_tree(self):
        single_fork = TestNode.get_fork_node()
        schema = TreeSchema(base_fork_node=single_fork)

        single_fork_nan_dict = schema.create_dummy_nan_tree()

        self.assertTrue(isinstance(single_fork_nan_dict, dict))
        for key in single_fork.get_children_names():
            self.assertTrue(key in single_fork_nan_dict.keys())

        self.assertTrue(np.isnan(single_fork_nan_dict['leaf-float']))
        self.assertEqual(single_fork_nan_dict['leaf-string'], 'nan')
        self.assertTrue(np.isnat(single_fork_nan_dict['leaf-date']))

        new_child_1 = ChildNode(name='leaf2-string', data_type=StringDataType())
        new_child_2 = ChildNode(name='leaf2-float', data_type=FloatDataType())
        new_fork = ForkNode(name='level2', children=[new_child_1, new_child_2])
        fork_for_test = ForkNode(name='test_find_child', children=single_fork.get_children() + [new_fork])

        schema = TreeSchema(base_fork_node=fork_for_test)
        multi_fork_nan_dict = schema.create_dummy_nan_tree()

        self.assertTrue(isinstance(multi_fork_nan_dict, dict))
        for key in fork_for_test.get_children_names():
            self.assertTrue(key in multi_fork_nan_dict.keys())

        self.assertTrue(np.isnan(multi_fork_nan_dict['leaf-float']))
        self.assertEqual(multi_fork_nan_dict['leaf-string'], 'nan')
        self.assertTrue(np.isnat(multi_fork_nan_dict['leaf-date']))
        self.assertEqual(multi_fork_nan_dict['level2']['leaf2-string'], 'nan')
        self.assertTrue(np.isnan(multi_fork_nan_dict['level2']['leaf2-float']))

    def test_eq(self):
        single_fork = TestNode.get_fork_node()
        schema1 = TreeSchema(base_fork_node=single_fork)
        schema2 = TreeSchema(base_fork_node=single_fork)
        self.assertEqual(schema1, schema2)

    def test__traverse(self):
        single_fork = TestNode.get_fork_node()
        ts = TreeSchema(base_fork_node=single_fork)

        self.assertEqual(ts._traverse(['leaf-float']), single_fork.find_child('leaf-float'))

        new_child_1 = ChildNode(name='leaf2-string', data_type=StringDataType())
        new_child_2 = ChildNode(name='leaf2-float', data_type=FloatDataType())
        new_fork = ForkNode(name='level2', children=[new_child_1, new_child_2])
        fork_for_test = ForkNode(name='test_find_child', children=single_fork.get_children() + [new_fork])
        ts = TreeSchema(base_fork_node=fork_for_test)

        self.assertEqual(ts._traverse(['level2', 'leaf2-string']), new_child_1)
        self.assertEqual(ts._traverse(['level2', 'leaf2-float']), new_child_2)

    def test_find_data_type(self):
        single_fork = TestNode.get_fork_node()
        ts = TreeSchema(base_fork_node=single_fork)

        self.assertEqual(ts.find_data_type("leaf-float"), single_fork.find_child('leaf-float').get_data_type())

        new_child_1 = ChildNode(name='leaf2-string', data_type=StringDataType())
        new_child_2 = ChildNode(name='leaf2-float', data_type=FloatDataType())
        new_fork = ForkNode(name='level2', children=[new_child_1, new_child_2])
        fork_for_test = ForkNode(name='test_find_child', children=single_fork.get_children() + [new_fork])
        ts = TreeSchema(base_fork_node=fork_for_test)

        self.assertEqual(ts.find_data_type('level2/leaf2-string'), new_child_1.get_data_type())
        self.assertEqual(ts.find_data_type('level2/leaf2-float'), new_child_2.get_data_type())

    def test_set_data_type(self):
        single_fork = TestNode.get_fork_node()
        ts = TreeSchema(base_fork_node=single_fork)

        self.assertEqual(ts.find_data_type('leaf-date'), DateDataType(resolution='D'))
        self.assertEqual(ts.find_data_type('leaf-date'), single_fork.find_child('leaf-date').get_data_type())
        ts = ts.set_data_type('leaf-date', StringDataType())
        self.assertEqual(ts.find_data_type('leaf-date'), StringDataType())
        self.assertEqual(ts.find_data_type('leaf-date'), single_fork.find_child('leaf-date').get_data_type())

        new_child_1 = ChildNode(name='leaf2-string', data_type=StringDataType())
        new_child_2 = ChildNode(name='leaf2-float', data_type=FloatDataType())
        new_fork = ForkNode(name='level2', children=[new_child_1, new_child_2])
        fork_for_test = ForkNode(name='test_find_child', children=single_fork.get_children() + [new_fork])
        ts = TreeSchema(base_fork_node=fork_for_test)

        self.assertEqual(ts.find_data_type('level2/leaf2-float'), FloatDataType())
        self.assertEqual(ts.find_data_type('level2/leaf2-float'), new_child_2.get_data_type())
        ts = ts.set_data_type('level2/leaf2-float', StringDataType())
        self.assertEqual(ts.find_data_type('level2/leaf2-float'), StringDataType())

    def test_mul(self):
        fork_1_string = ForkNode(name="base", children=[TestNode.get_single_string_leaf()])
        fork_1_float = ForkNode(name="base2", children=[TestNode.get_single_float_leaf()])
        ts_string = TreeSchema(base_fork_node=fork_1_string)
        ts_float = TreeSchema(base_fork_node=fork_1_float)

        ts_mul = ts_string * ts_float
        self.assertEqual(ts_mul.base_fork_node.get_name(), "empty")
        self.assertEqual(len(ts_mul.base_fork_node.get_children()), 0)

        fork_1_string = ForkNode(name="base", children=[TestNode.get_single_string_leaf()])
        fork_1_float = ForkNode(name="base", children=[TestNode.get_single_float_leaf()])
        ts_string = TreeSchema(base_fork_node=fork_1_string)
        ts_float = TreeSchema(base_fork_node=fork_1_float)

        ts_mul = ts_string * ts_float
        self.assertEqual(ts_mul.base_fork_node.get_name(), "base")
        self.assertEqual(len(ts_mul.base_fork_node.get_children()), 0)

        fork_1_string = ForkNode(name="base", children=[TestNode.get_single_string_leaf().set_name("foo")])
        fork_1_float = ForkNode(name="base", children=[TestNode.get_single_float_leaf().set_name("foo")])
        ts_string = TreeSchema(base_fork_node=fork_1_string)
        ts_float = TreeSchema(base_fork_node=fork_1_float)

        ts_mul = ts_string * ts_float
        self.assertEqual(ts_mul.base_fork_node.get_name(), "base")
        self.assertEqual(len(ts_mul.base_fork_node.get_children()), 1)
        self.assertTrue(
            ts_mul.base_fork_node.get_children()[0] == TestNode.get_single_string_leaf().set_name(
                "foo"))


class TestTreeDataType(TestCase):
    """
    Test class for ListDataType.
    """

    @staticmethod
    def get_schema_v1():
        fork = TestNode.get_fork_node(format_string="%Y-%m-%d")
        return TreeSchema(base_fork_node=fork)

    @staticmethod
    def get_schema_v2():
        fork = TestNode.get_fork_node(format_string="%Y-%m-%d")
        return TreeSchema(
            base_fork_node=ForkNode(name='base',
                                    children=fork.get_children() + [ChildNode('new_child', StringDataType())])
        )

    @staticmethod
    def get_schema_v3():
        fork = TestNode.get_fork_node(format_string="%Y-%m-%d")
        fork2 = ForkNode(name='level2',
                         children=[ChildNode('leaf2-string', StringDataType()),
                                   ChildNode('leaf2-float', FloatDataType())]
                         )
        return TreeSchema(
            base_fork_node=ForkNode('base', fork.get_children() + [ChildNode('new_child', StringDataType()), fork2])
        )

    def test_is_nullable(self):
        dtp = TreeDataType(base_fork=self.get_schema_v1().base_fork_node, nullable=False)
        self.assertFalse(dtp.is_nullable())
        dtp = TreeDataType(base_fork=self.get_schema_v1().base_fork_node, nullable=True)
        self.assertTrue(dtp.is_nullable())

    def test_get_numpy_type(self):
        dtp = TreeDataType(base_fork=self.get_schema_v1().base_fork_node)
        self.assertEqual(dtp.get_numpy_type(), dict)
        dtp = TreeDataType(base_fork=self.get_schema_v2().base_fork_node)
        self.assertEqual(dtp.get_numpy_type(), dict)
        dtp = TreeDataType(base_fork=self.get_schema_v3().base_fork_node)
        self.assertEqual(dtp.get_numpy_type(), dict)

    def test_get_python_type(self):
        dtp = TreeDataType(base_fork=self.get_schema_v1().base_fork_node)
        self.assertEqual(dtp.get_python_type(), dict)
        dtp = TreeDataType(base_fork=self.get_schema_v2().base_fork_node)
        self.assertEqual(dtp.get_python_type(), dict)
        dtp = TreeDataType(base_fork=self.get_schema_v3().base_fork_node)
        self.assertEqual(dtp.get_python_type(), dict)

    def test_build_numpy_value(self):
        # Case number 1
        dtp = TreeDataType(base_fork=self.get_schema_v1().base_fork_node)

        built_empty = dtp.build_numpy_value({})
        for name in dtp.base_fork.get_children_names():
            self.assertTrue(name in built_empty.keys())

        self.assertEqual(built_empty['leaf-string'], 'nan')
        self.assertTrue(np.isnan(built_empty['leaf-float']))
        self.assertTrue(np.isnat(built_empty['leaf-date']))

        built_non_empty = dtp.build_numpy_value({
            'leaf-string': 'tralala', 'leaf-float': 29.23, 'leaf-date': '1993-04-01'})
        for name in dtp.base_fork.get_children_names():
            self.assertTrue(name in built_non_empty.keys())

        self.assertEqual(built_non_empty['leaf-string'], 'tralala')
        self.assertEqual(built_non_empty['leaf-float'], float(29.23))
        self.assertEqual(built_non_empty['leaf-date'], np.datetime64('1993-04-01'))

        try:
            dtp.build_numpy_value({'non-existent': 29.23})
        except RuntimeError as e:
            self.assertEqual(str(e), "Unknown node of name 'non-existent' not specified in the Node 'test-fork'")

        # Case number 2
        dtp = TreeDataType(base_fork=self.get_schema_v2().base_fork_node)

        built_empty = dtp.build_numpy_value({})
        for name in dtp.base_fork.get_children_names():
            self.assertTrue(name in built_empty.keys())

        self.assertEqual(built_empty['leaf-string'], 'nan')
        self.assertTrue(np.isnan(built_empty['leaf-float']))
        self.assertTrue(np.isnat(built_empty['leaf-date']))
        self.assertEqual(built_empty['new_child'], 'nan')

        built_non_empty = dtp.build_numpy_value({
            'leaf-string': 'tralala', 'leaf-float': 29.23, 'leaf-date': '1993-04-01'})
        for name in dtp.base_fork.get_children_names():
            self.assertTrue(name in built_non_empty.keys())

        self.assertEqual(built_non_empty['leaf-string'], 'tralala')
        self.assertEqual(built_non_empty['leaf-float'], float(29.23))
        self.assertEqual(built_non_empty['leaf-date'], np.datetime64('1993-04-01'))
        self.assertEqual(built_non_empty['new_child'], 'nan')

        try:
            dtp.build_numpy_value({'non-existent': 29.23})
        except RuntimeError as e:
            self.assertEqual(str(e), "Unknown node of name 'non-existent' not specified in the Node 'base'")

        # Case number 3
        dtp = TreeDataType(base_fork=self.get_schema_v3().base_fork_node)

        built_empty = dtp.build_numpy_value({})
        for name in dtp.base_fork.get_children_names():
            self.assertTrue(name in built_empty.keys())

        self.assertEqual(built_empty['leaf-string'], 'nan')
        self.assertTrue(np.isnan(built_empty['leaf-float']))
        self.assertTrue(np.isnat(built_empty['leaf-date']))
        self.assertEqual(built_empty['new_child'], 'nan')
        self.assertEqual(built_empty['level2']['leaf2-string'], 'nan')
        self.assertTrue(np.isnan(built_empty['level2']['leaf2-float']))

        built_non_empty = dtp.build_numpy_value({
            'leaf-string': 'tralala', 'leaf-float': 29.23, 'leaf-date': '1993-04-01',
            'level2': {'leaf2-float': -10.99}})
        for name in dtp.base_fork.get_children_names():
            self.assertTrue(name in built_non_empty.keys())

        self.assertEqual(built_non_empty['leaf-string'], 'tralala')
        self.assertEqual(built_non_empty['leaf-float'], float(29.23))
        self.assertEqual(built_non_empty['leaf-date'], np.datetime64('1993-04-01'))
        self.assertEqual(built_non_empty['new_child'], 'nan')
        self.assertEqual(built_non_empty['level2']['leaf2-string'], 'nan')
        self.assertEqual(built_non_empty['level2']['leaf2-float'], float(-10.99))

        try:
            dtp.build_numpy_value({'level2': {'non-existent': 29.23}})
        except RuntimeError as e:
            self.assertEqual(str(e), "Unknown node of name 'non-existent' not specified in the Node 'level2'")

    def test_build_python_value(self):
        # Case number 1
        dtp = TreeDataType(base_fork=self.get_schema_v1().base_fork_node)

        built_empty = dtp.build_python_value({})
        for name in dtp.base_fork.get_children_names():
            self.assertTrue(name in built_empty.keys())

        self.assertEqual(built_empty['leaf-string'], "None")
        self.assertTrue(built_empty['leaf-float'] is None)
        self.assertEqual(built_empty['leaf-date'], '')

        built_non_empty = dtp.build_python_value({
            'leaf-string': 'tralala', 'leaf-float': 29.23, 'leaf-date': '1993-04-01'})
        for name in dtp.base_fork.get_children_names():
            self.assertTrue(name in built_non_empty.keys())

        self.assertEqual(built_non_empty['leaf-string'], 'tralala')
        self.assertEqual(built_non_empty['leaf-float'], float(29.23))
        self.assertEqual(built_non_empty['leaf-date'], datetime(1993, 4, 1))

        try:
            dtp.build_python_value({'non-existent': 29.23})
        except RuntimeError as e:
            self.assertEqual(str(e), "Unknown node of name 'non-existent' not specified in the Node 'test-fork'")

        # Case number 2
        dtp = TreeDataType(base_fork=self.get_schema_v2().base_fork_node)

        built_empty = dtp.build_python_value({})
        for name in dtp.base_fork.get_children_names():
            self.assertTrue(name in built_empty.keys())

        self.assertEqual(built_empty['leaf-string'], "None")
        self.assertTrue(built_empty['leaf-float'] is None)
        self.assertEqual(built_empty['leaf-date'], '')
        self.assertEqual(built_empty['new_child'], 'None')

        built_non_empty = dtp.build_python_value({
            'leaf-string': 'tralala', 'leaf-float': 29.23, 'leaf-date': '1993-04-01'})
        for name in dtp.base_fork.get_children_names():
            self.assertTrue(name in built_non_empty.keys())

        self.assertEqual(built_non_empty['leaf-string'], 'tralala')
        self.assertEqual(built_non_empty['leaf-float'], float(29.23))
        self.assertEqual(built_non_empty['leaf-date'], datetime(1993, 4, 1))
        self.assertEqual(built_non_empty['new_child'], 'None')

        try:
            dtp.build_python_value({'non-existent': 29.23})
        except RuntimeError as e:
            self.assertEqual(str(e), "Unknown node of name 'non-existent' not specified in the Node 'base'")

        # Case number 3
        dtp = TreeDataType(base_fork=self.get_schema_v3().base_fork_node)

        built_empty = dtp.build_python_value({})
        for name in dtp.base_fork.get_children_names():
            self.assertTrue(name in built_empty.keys())

        self.assertEqual(built_empty['leaf-string'], "None")
        self.assertTrue(built_empty['leaf-float'] is None)
        self.assertEqual(built_empty['leaf-date'], '')
        self.assertEqual(built_empty['new_child'], 'None')
        self.assertEqual(built_empty['level2']['leaf2-string'], 'None')
        self.assertTrue(built_empty['level2']['leaf2-float'] is None)

        built_non_empty = dtp.build_python_value({
            'leaf-string': 'tralala', 'leaf-float': 29.23, 'leaf-date': '1993-04-01',
            'level2': {'leaf2-float': -10.99}})
        for name in dtp.base_fork.get_children_names():
            self.assertTrue(name in built_non_empty.keys())

        self.assertEqual(built_non_empty['leaf-string'], 'tralala')
        self.assertEqual(built_non_empty['leaf-float'], float(29.23))
        self.assertEqual(built_non_empty['leaf-date'], datetime(1993, 4, 1))
        self.assertEqual(built_non_empty['new_child'], 'None')
        self.assertEqual(built_non_empty['level2']['leaf2-string'], 'None')
        self.assertEqual(built_non_empty['level2']['leaf2-float'], float(-10.99))

        try:
            dtp.build_python_value({'level2': {'non-existent': 29.23}})
        except RuntimeError as e:
            self.assertEqual(str(e), "Unknown node of name 'non-existent' not specified in the Node 'level2'")

    def test_eq(self):
        dtp1 = TreeDataType(base_fork=self.get_schema_v1().base_fork_node)
        dtp2 = TreeDataType(base_fork=self.get_schema_v1().base_fork_node)
        self.assertEqual(dtp1, dtp2)
        dtp1 = TreeDataType(base_fork=self.get_schema_v2().base_fork_node)
        dtp2 = TreeDataType(base_fork=self.get_schema_v2().base_fork_node)
        self.assertEqual(dtp1, dtp2)
        dtp1 = TreeDataType(base_fork=self.get_schema_v3().base_fork_node)
        dtp2 = TreeDataType(base_fork=self.get_schema_v3().base_fork_node)
        self.assertEqual(dtp1, dtp2)


class TestComparisonsChild(TestCase):
    """
    Test class for comparisons
    """

    @staticmethod
    def get_data_types():
        dt = ChildNode('foo',
                       DataType(numpy_dtype='<i8', python_dtype=int, numpy_na_value=np.nan, python_na_value=None))
        sdt = ChildNode('foo', StringDataType())
        fdt = ChildNode('foo', FloatDataType())
        ddt_d = ChildNode('foo', DateDataType(resolution='D'))
        ddt_s = ChildNode('foo', DateDataType(resolution='s'))
        adt_f = ChildNode('foo', ArrayDataType(element_data_type=FloatDataType()))
        adt_s = ChildNode('foo', ArrayDataType(element_data_type=StringDataType()))
        ldt_fsd = ChildNode('foo', ListDataType(element_data_types=[FloatDataType(), StringDataType(), DateDataType()]))
        ldt_ssd = ChildNode('foo',
                            ListDataType(element_data_types=[StringDataType(), StringDataType(), DateDataType()]))
        wc = ChildNode('foo2', StringDataType())

        f_sdt = copy(sdt).set_name('sdt')
        f_fdt = copy(fdt).set_name('fdt')
        f_ddt_s = copy(ddt_s).set_name('ddt_s')
        f = ForkNode(name='foo_fork', children=[f_sdt, f_fdt, f_ddt_s])

        return dt, sdt, fdt, ddt_d, ddt_s, adt_f, adt_s, ldt_fsd, ldt_ssd, wc, f_sdt, f_fdt, f_ddt_s, f

    def test_comparisons_data_type(self):
        dt, sdt, fdt, ddt_d, ddt_s, adt_f, adt_s, ldt_fsd, ldt_ssd, wc, *f = self.get_data_types()
        self.assertTrue(dt <= dt)
        self.assertFalse(fdt <= dt)
        self.assertFalse(ddt_d <= dt)
        self.assertFalse(ddt_s <= dt)
        self.assertFalse(adt_f <= dt)
        self.assertFalse(adt_s <= dt)
        self.assertFalse(ldt_fsd <= dt)
        self.assertFalse(ldt_ssd <= dt)
        self.assertFalse(sdt <= dt)
        self.assertFalse(wc <= dt)
        self.assertFalse(f[-1] <= dt)

        self.assertFalse(dt > dt)
        self.assertFalse(fdt > dt)
        self.assertFalse(ddt_d > dt)
        self.assertFalse(ddt_s > dt)
        self.assertFalse(adt_f > dt)
        self.assertFalse(adt_s > dt)
        self.assertFalse(ldt_fsd > dt)
        self.assertFalse(ldt_ssd > dt)
        self.assertTrue(sdt > dt)
        self.assertFalse(wc > dt)
        self.assertFalse(f[-1] > dt)

        self.assertTrue(dt >= dt)
        self.assertFalse(dt >= fdt)
        self.assertFalse(dt >= ddt_d)
        self.assertFalse(dt >= ddt_s)
        self.assertFalse(dt >= adt_f)
        self.assertFalse(dt >= adt_s)
        self.assertFalse(dt >= ldt_fsd)
        self.assertFalse(dt >= ldt_ssd)
        self.assertFalse(dt >= sdt)
        self.assertFalse(dt >= wc)
        self.assertFalse(dt >= f[-1])

        self.assertFalse(dt < dt)
        self.assertFalse(dt < fdt)
        self.assertFalse(dt < ddt_d)
        self.assertFalse(dt < ddt_s)
        self.assertFalse(dt < adt_f)
        self.assertFalse(dt < adt_s)
        self.assertFalse(dt < ldt_fsd)
        self.assertFalse(dt < ldt_ssd)
        self.assertTrue(dt < sdt)
        self.assertFalse(dt < wc)
        self.assertFalse(dt < f[-1])

    def test_comparisons_string(self):
        dt, sdt, fdt, ddt_d, ddt_s, adt_f, adt_s, ldt_fsd, ldt_ssd, wc, *f = self.get_data_types()

        self.assertTrue(dt <= sdt)
        self.assertTrue(fdt <= sdt)
        self.assertTrue(ddt_d <= sdt)
        self.assertTrue(ddt_s <= sdt)
        self.assertTrue(adt_f <= sdt)
        self.assertTrue(adt_s <= sdt)
        self.assertTrue(ldt_fsd <= sdt)
        self.assertTrue(ldt_ssd <= sdt)
        self.assertTrue(sdt <= sdt)
        self.assertFalse(wc <= sdt)
        self.assertFalse(f[-1] <= sdt)

        self.assertFalse(dt > sdt)
        self.assertFalse(fdt > sdt)
        self.assertFalse(ddt_d > sdt)
        self.assertFalse(ddt_s > sdt)
        self.assertFalse(adt_f > sdt)
        self.assertFalse(adt_s > sdt)
        self.assertFalse(ldt_fsd > sdt)
        self.assertFalse(ldt_ssd > sdt)
        self.assertFalse(sdt > sdt)
        self.assertFalse(wc > sdt)
        self.assertFalse(f[-1] > sdt)
        self.assertTrue(f[-1] > f[0])

        self.assertTrue(sdt >= dt)
        self.assertTrue(sdt >= fdt)
        self.assertTrue(sdt >= ddt_d)
        self.assertTrue(sdt >= ddt_s)
        self.assertTrue(sdt >= adt_f)
        self.assertTrue(sdt >= adt_s)
        self.assertTrue(sdt >= ldt_fsd)
        self.assertTrue(sdt >= ldt_ssd)
        self.assertTrue(sdt >= sdt)
        self.assertFalse(sdt <= wc)
        self.assertFalse(sdt <= f[-1])
        self.assertTrue(f[0] <= f[-1])

        self.assertFalse(sdt < dt)
        self.assertFalse(sdt < fdt)
        self.assertFalse(sdt < ddt_d)
        self.assertFalse(sdt < ddt_s)
        self.assertFalse(sdt < adt_f)
        self.assertFalse(sdt < adt_s)
        self.assertFalse(sdt < ldt_fsd)
        self.assertFalse(sdt < ldt_ssd)
        self.assertFalse(sdt < sdt)
        self.assertFalse(sdt < wc)
        self.assertFalse(sdt < f[-1])
        self.assertTrue(f[0] < f[-1])

    def test_comparisons_float(self):
        dt, sdt, fdt, ddt_d, ddt_s, adt_f, adt_s, ldt_fsd, ldt_ssd, wc = self.get_data_types()

        self.assertFalse(dt <= fdt)
        self.assertTrue(fdt <= fdt)
        self.assertFalse(ddt_d <= fdt)
        self.assertFalse(ddt_s <= fdt)
        self.assertFalse(adt_f <= fdt)
        self.assertFalse(adt_s <= fdt)
        self.assertFalse(ldt_fsd <= fdt)
        self.assertFalse(ldt_ssd <= fdt)
        self.assertFalse(sdt <= fdt)
        self.assertFalse(wc <= fdt)

        self.assertFalse(dt > fdt)
        self.assertFalse(fdt > fdt)
        self.assertFalse(ddt_d > fdt)
        self.assertFalse(ddt_s > fdt)
        self.assertFalse(adt_f > fdt)
        self.assertFalse(adt_s > fdt)
        self.assertFalse(ldt_fsd > fdt)
        self.assertFalse(ldt_ssd > fdt)
        self.assertTrue(sdt > fdt)
        self.assertFalse(wc > fdt)

        self.assertFalse(fdt >= dt)
        self.assertTrue(fdt >= fdt)
        self.assertFalse(fdt >= ddt_d)
        self.assertFalse(fdt >= ddt_s)
        self.assertFalse(fdt >= adt_f)
        self.assertFalse(fdt >= adt_s)
        self.assertFalse(fdt >= ldt_fsd)
        self.assertFalse(fdt >= ldt_ssd)
        self.assertFalse(fdt >= sdt)
        self.assertFalse(fdt <= wc)

        self.assertFalse(fdt < dt)
        self.assertFalse(fdt < fdt)
        self.assertFalse(fdt < ddt_d)
        self.assertFalse(fdt < ddt_s)
        self.assertFalse(fdt < adt_f)
        self.assertFalse(fdt < adt_s)
        self.assertFalse(fdt < ldt_fsd)
        self.assertFalse(fdt < ldt_ssd)
        self.assertTrue(fdt < sdt)
        self.assertFalse(fdt < wc)

    def test_comparisons_date_1(self):
        dt, sdt, fdt, ddt_d, ddt_s, adt_f, adt_s, ldt_fsd, ldt_ssd, wc = self.get_data_types()

        self.assertFalse(dt <= ddt_d)
        self.assertFalse(fdt <= ddt_d)
        self.assertTrue(ddt_d <= ddt_d)
        self.assertFalse(ddt_s <= ddt_d)
        self.assertFalse(adt_f <= ddt_d)
        self.assertFalse(adt_s <= ddt_d)
        self.assertFalse(ldt_fsd <= ddt_d)
        self.assertFalse(ldt_ssd <= ddt_d)
        self.assertFalse(sdt <= ddt_d)
        self.assertFalse(wc <= ddt_d)

        self.assertFalse(dt > ddt_d)
        self.assertFalse(fdt > ddt_d)
        self.assertFalse(ddt_d > ddt_d)
        self.assertTrue(ddt_s > ddt_d)
        self.assertFalse(adt_f > ddt_d)
        self.assertFalse(adt_s > ddt_d)
        self.assertFalse(ldt_fsd > ddt_d)
        self.assertFalse(ldt_ssd > ddt_d)
        self.assertTrue(sdt > ddt_d)
        self.assertFalse(wc > ddt_d)

        self.assertFalse(ddt_d >= dt)
        self.assertFalse(ddt_d >= fdt)
        self.assertTrue(ddt_d >= ddt_d)
        self.assertFalse(ddt_d >= ddt_s)
        self.assertFalse(ddt_d >= adt_f)
        self.assertFalse(ddt_d >= adt_s)
        self.assertFalse(ddt_d >= ldt_fsd)
        self.assertFalse(ddt_d >= ldt_ssd)
        self.assertFalse(ddt_d >= sdt)
        self.assertFalse(ddt_d >= wc)

        self.assertFalse(ddt_d < dt)
        self.assertFalse(ddt_d < fdt)
        self.assertFalse(ddt_d < ddt_d)
        self.assertTrue(ddt_d < ddt_s)
        self.assertFalse(ddt_d < adt_f)
        self.assertFalse(ddt_d < adt_s)
        self.assertFalse(ddt_d < ldt_fsd)
        self.assertFalse(ddt_d < ldt_ssd)
        self.assertTrue(ddt_d < sdt)
        self.assertFalse(ddt_d < wc)

    def test_comparisons_date_2(self):
        dt, sdt, fdt, ddt_d, ddt_s, adt_f, adt_s, ldt_fsd, ldt_ssd, wc = self.get_data_types()

        self.assertFalse(dt <= ddt_s)
        self.assertFalse(fdt <= ddt_s)
        self.assertTrue(ddt_d <= ddt_s)
        self.assertTrue(ddt_s <= ddt_s)
        self.assertFalse(adt_f <= ddt_s)
        self.assertFalse(adt_s <= ddt_s)
        self.assertFalse(ldt_fsd <= ddt_s)
        self.assertFalse(ldt_ssd <= ddt_s)
        self.assertFalse(sdt <= ddt_s)
        self.assertFalse(wc <= ddt_s)

        self.assertFalse(dt > ddt_s)
        self.assertFalse(fdt > ddt_s)
        self.assertFalse(ddt_d > ddt_s)
        self.assertFalse(ddt_s > ddt_s)
        self.assertFalse(adt_f > ddt_s)
        self.assertFalse(adt_s > ddt_s)
        self.assertFalse(ldt_fsd > ddt_s)
        self.assertFalse(ldt_ssd > ddt_s)
        self.assertTrue(sdt > ddt_s)
        self.assertFalse(wc > ddt_s)

        self.assertFalse(ddt_s >= dt)
        self.assertFalse(ddt_s >= fdt)
        self.assertTrue(ddt_s >= ddt_d)
        self.assertTrue(ddt_s >= ddt_s)
        self.assertFalse(ddt_s >= adt_f)
        self.assertFalse(ddt_s >= adt_s)
        self.assertFalse(ddt_s >= ldt_fsd)
        self.assertFalse(ddt_s >= ldt_ssd)
        self.assertFalse(ddt_s >= sdt)
        self.assertFalse(ddt_s >= wc)

        self.assertFalse(ddt_s < dt)
        self.assertFalse(ddt_s < fdt)
        self.assertFalse(ddt_s < ddt_d)
        self.assertFalse(ddt_s < ddt_s)
        self.assertFalse(ddt_s < adt_f)
        self.assertFalse(ddt_s < adt_s)
        self.assertFalse(ddt_s < ldt_fsd)
        self.assertFalse(ddt_s < ldt_ssd)
        self.assertTrue(ddt_s < sdt)
        self.assertFalse(ddt_s < wc)

    def test_comparisons_array_1(self):
        dt, sdt, fdt, ddt_d, ddt_s, adt_f, adt_s, ldt_fsd, ldt_ssd, wc = self.get_data_types()

        self.assertFalse(dt <= adt_f)
        self.assertFalse(fdt <= adt_f)
        self.assertFalse(ddt_d <= adt_f)
        self.assertFalse(ddt_s <= adt_f)
        self.assertTrue(adt_f <= adt_f)
        self.assertFalse(adt_s <= adt_f)
        self.assertFalse(ldt_fsd <= adt_f)
        self.assertFalse(ldt_ssd <= adt_f)
        self.assertFalse(sdt <= adt_f)
        self.assertFalse(wc <= adt_f)

        self.assertFalse(dt > adt_f)
        self.assertFalse(fdt > adt_f)
        self.assertFalse(ddt_d > adt_f)
        self.assertFalse(ddt_s > adt_f)
        self.assertFalse(adt_f > adt_f)
        self.assertTrue(adt_s > adt_f)
        self.assertTrue(ldt_fsd > adt_f)
        self.assertTrue(ldt_ssd > adt_f)
        self.assertTrue(sdt > adt_f)
        self.assertFalse(wc > adt_f)

        self.assertFalse(adt_f >= dt)
        self.assertFalse(adt_f >= fdt)
        self.assertFalse(adt_f >= ddt_d)
        self.assertFalse(adt_f >= ddt_s)
        self.assertTrue(adt_f >= adt_f)
        self.assertFalse(adt_f >= adt_s)
        self.assertFalse(adt_f >= ldt_fsd)
        self.assertFalse(adt_f >= ldt_ssd)
        self.assertFalse(adt_f >= sdt)
        self.assertFalse(adt_f >= wc)

        self.assertFalse(adt_f < dt)
        self.assertFalse(adt_f < fdt)
        self.assertFalse(adt_f < ddt_d)
        self.assertFalse(adt_f < ddt_s)
        self.assertFalse(adt_f < adt_f)
        self.assertTrue(adt_f < adt_s)
        self.assertTrue(adt_f < ldt_fsd)
        self.assertTrue(adt_f < ldt_ssd)
        self.assertTrue(adt_f < sdt)
        self.assertFalse(adt_f < wc)

    def test_comparisons_array_2(self):
        dt, sdt, fdt, ddt_d, ddt_s, adt_f, adt_s, ldt_fsd, ldt_ssd, wc = self.get_data_types()

        self.assertFalse(dt <= adt_s)
        self.assertFalse(fdt <= adt_s)
        self.assertFalse(ddt_d <= adt_s)
        self.assertFalse(ddt_s <= adt_s)
        self.assertTrue(adt_f <= adt_s)
        self.assertTrue(adt_s <= adt_s)
        self.assertFalse(ldt_fsd <= adt_s)
        self.assertFalse(ldt_ssd <= adt_s)
        self.assertFalse(sdt <= adt_s)
        self.assertFalse(wc <= adt_s)

        self.assertFalse(dt > adt_s)
        self.assertFalse(fdt > adt_s)
        self.assertFalse(ddt_d > adt_s)
        self.assertFalse(ddt_s > adt_s)
        self.assertFalse(adt_f > adt_s)
        self.assertFalse(adt_s > adt_s)
        self.assertTrue(ldt_fsd > adt_s)
        self.assertTrue(ldt_ssd > adt_s)
        self.assertTrue(sdt > adt_s)
        self.assertFalse(wc > adt_s)

        self.assertFalse(adt_s >= dt)
        self.assertFalse(adt_s >= fdt)
        self.assertFalse(adt_s >= ddt_d)
        self.assertFalse(adt_s >= ddt_s)
        self.assertTrue(adt_s >= adt_f)
        self.assertTrue(adt_s >= adt_s)
        self.assertFalse(adt_s >= ldt_fsd)
        self.assertFalse(adt_s >= ldt_ssd)
        self.assertFalse(adt_s >= sdt)
        self.assertFalse(adt_s >= wc)

        self.assertFalse(adt_s < dt)
        self.assertFalse(adt_s < fdt)
        self.assertFalse(adt_s < ddt_d)
        self.assertFalse(adt_s < ddt_s)
        self.assertFalse(adt_s < adt_f)
        self.assertFalse(adt_s < adt_s)
        self.assertTrue(adt_s < ldt_fsd)
        self.assertTrue(adt_s < ldt_ssd)
        self.assertTrue(adt_s < sdt)
        self.assertFalse(adt_s < wc)

    def test_comparisons_list_1(self):
        dt, sdt, fdt, ddt_d, ddt_s, adt_f, adt_s, ldt_fsd, ldt_ssd, wc = self.get_data_types()

        self.assertFalse(dt <= ldt_fsd)
        self.assertFalse(fdt <= ldt_fsd)
        self.assertFalse(ddt_d <= ldt_fsd)
        self.assertFalse(ddt_s <= ldt_fsd)
        self.assertTrue(adt_f <= ldt_fsd)
        self.assertTrue(adt_s <= ldt_fsd)
        self.assertTrue(ldt_fsd <= ldt_fsd)
        self.assertFalse(ldt_ssd <= ldt_fsd)
        self.assertFalse(sdt <= ldt_fsd)
        self.assertFalse(wc <= ldt_fsd)

        self.assertFalse(dt > ldt_fsd)
        self.assertFalse(fdt > ldt_fsd)
        self.assertFalse(ddt_d > ldt_fsd)
        self.assertFalse(ddt_s > ldt_fsd)
        self.assertFalse(adt_f > ldt_fsd)
        self.assertFalse(adt_s > ldt_fsd)
        self.assertFalse(ldt_fsd > ldt_fsd)
        self.assertFalse(ldt_ssd > ldt_fsd)
        self.assertTrue(sdt > ldt_fsd)
        self.assertFalse(wc > ldt_fsd)

        self.assertFalse(ldt_fsd >= dt)
        self.assertFalse(ldt_fsd >= fdt)
        self.assertFalse(ldt_fsd >= ddt_d)
        self.assertFalse(ldt_fsd >= ddt_s)
        self.assertTrue(ldt_fsd >= adt_f)
        self.assertTrue(ldt_fsd >= adt_s)
        self.assertTrue(ldt_fsd >= ldt_fsd)
        self.assertFalse(ldt_fsd >= ldt_ssd)
        self.assertFalse(ldt_fsd >= sdt)
        self.assertFalse(ldt_fsd >= wc)

        self.assertFalse(ldt_fsd < dt)
        self.assertFalse(ldt_fsd < fdt)
        self.assertFalse(ldt_fsd < ddt_d)
        self.assertFalse(ldt_fsd < ddt_s)
        self.assertFalse(ldt_fsd < adt_f)
        self.assertFalse(ldt_fsd < adt_s)
        self.assertFalse(ldt_fsd < ldt_fsd)
        self.assertFalse(ldt_fsd < ldt_ssd)
        self.assertTrue(ldt_fsd < sdt)
        self.assertFalse(ldt_fsd < wc)

    def test_comparisons_list_2(self):
        dt, sdt, fdt, ddt_d, ddt_s, adt_f, adt_s, ldt_fsd, ldt_ssd, wc = self.get_data_types()

        self.assertFalse(dt <= ldt_ssd)
        self.assertFalse(fdt <= ldt_ssd)
        self.assertFalse(ddt_d <= ldt_ssd)
        self.assertFalse(ddt_s <= ldt_ssd)
        self.assertTrue(adt_f <= ldt_ssd)
        self.assertTrue(adt_s <= ldt_ssd)
        self.assertTrue(ldt_fsd <= ldt_ssd)
        self.assertTrue(ldt_ssd <= ldt_ssd)
        self.assertFalse(sdt <= ldt_ssd)
        self.assertFalse(sdt <= wc)

        self.assertFalse(dt > ldt_ssd)
        self.assertFalse(fdt > ldt_ssd)
        self.assertFalse(ddt_d > ldt_ssd)
        self.assertFalse(ddt_s > ldt_ssd)
        self.assertFalse(adt_f > ldt_ssd)
        self.assertFalse(adt_s > ldt_ssd)
        self.assertFalse(ldt_fsd > ldt_ssd)
        self.assertFalse(ldt_ssd > ldt_ssd)
        self.assertTrue(sdt > ldt_ssd)
        self.assertFalse(sdt > wc)

        self.assertFalse(ldt_ssd >= dt)
        self.assertFalse(ldt_ssd >= fdt)
        self.assertFalse(ldt_ssd >= ddt_d)
        self.assertFalse(ldt_ssd >= ddt_s)
        self.assertTrue(ldt_ssd >= adt_f)
        self.assertTrue(ldt_ssd >= adt_s)
        self.assertTrue(ldt_ssd >= ldt_fsd)
        self.assertTrue(ldt_ssd >= ldt_ssd)
        self.assertFalse(ldt_ssd >= sdt)
        self.assertFalse(ldt_ssd >= wc)

        self.assertFalse(ldt_ssd < dt)
        self.assertFalse(ldt_ssd < fdt)
        self.assertFalse(ldt_ssd < ddt_d)
        self.assertFalse(ldt_ssd < ddt_s)
        self.assertFalse(ldt_ssd < adt_f)
        self.assertFalse(ldt_ssd < adt_s)
        self.assertFalse(ldt_ssd < ldt_fsd)
        self.assertFalse(ldt_ssd < ldt_ssd)
        self.assertTrue(ldt_ssd < sdt)
        self.assertFalse(ldt_ssd < wc)
