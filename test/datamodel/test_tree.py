from unittest import TestCase
from datetime import datetime
import numpy as np

from src.datamodel.datatypes import ChildNode, ForkNode, TreeSchema
from src.datamodel.datatypes import StringDataType, FloatDataType, DateDataType, TreeDataType


class TestNode(TestCase):
    """
    Test class for the Node class.
    """

    @staticmethod
    def get_single_string_leaf():
        return ChildNode(name="leaf-string", data_type=StringDataType(longest_string=10))

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

        single_leaf = single_leaf.overwrite_child(name='new-leaf', data_type=DateDataType(resolution='M'))

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
        single_fork = single_fork.overwrite_children(children=new_children, name='new-fork')

        self.assertTrue(single_fork.is_fork())
        self.assertEqual(len(single_fork.children), 1)
        self.assertEqual(single_fork.name, 'new-fork')
        for child in single_fork.children:
            self.assertTrue('new' in child.name)

        new_children_fail = new_children + [ChildNode(name='new', data_type=FloatDataType())]

        try:
            single_fork.overwrite_children(children=new_children_fail, name='new-fork')
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
        single_fork_built_values = single_fork.build_value(value=single_fork_values, method='numpy')

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
        single_fork_built_values = fork_for_test.build_value(value=single_fork_values, method='numpy')

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
        single_fork_built_values = single_fork.build_value(value=single_fork_values, method='python')

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
        single_fork_built_values = fork_for_test.build_value(value=single_fork_values, method='numpy')

        self.assertEqual(float(single_fork_values['leaf-float']), single_fork_built_values['leaf-float'])
        self.assertEqual(np.datetime64(single_fork_values['leaf-date']).astype('<M8[D]'),
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
        self.assertEqual(single_fork1, single_fork2)


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

        self.assertEqual(ts._traverse(ts.base_fork_node, ['leaf-float']), single_fork.find_child('leaf-float'))

        new_child_1 = ChildNode(name='leaf2-string', data_type=StringDataType())
        new_child_2 = ChildNode(name='leaf2-float', data_type=FloatDataType())
        new_fork = ForkNode(name='level2', children=[new_child_1, new_child_2])
        fork_for_test = ForkNode(name='test_find_child', children=single_fork.get_children() + [new_fork])
        ts = TreeSchema(base_fork_node=fork_for_test)

        self.assertEqual(ts._traverse(ts.base_fork_node, ['leaf2-string', 'level2']), new_child_1)
        self.assertEqual(ts._traverse(ts.base_fork_node, ['leaf2-float', 'level2']), new_child_2)

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
        self.assertEqual(ts.find_data_type('level2/leaf2-float'), new_child_2.get_data_type())


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
        dtp = TreeDataType(schema=self.get_schema_v1(), nullable=False)
        self.assertFalse(dtp.is_nullable())
        dtp = TreeDataType(schema=self.get_schema_v1(), nullable=True)
        self.assertTrue(dtp.is_nullable())

    def test_get_numpy_type(self):
        dtp = TreeDataType(schema=self.get_schema_v1())
        self.assertEqual(dtp.get_numpy_type(), dict)
        dtp = TreeDataType(schema=self.get_schema_v2())
        self.assertEqual(dtp.get_numpy_type(), dict)
        dtp = TreeDataType(schema=self.get_schema_v3())
        self.assertEqual(dtp.get_numpy_type(), dict)

    def test_get_python_type(self):
        dtp = TreeDataType(schema=self.get_schema_v1())
        self.assertEqual(dtp.get_python_type(), dict)
        dtp = TreeDataType(schema=self.get_schema_v2())
        self.assertEqual(dtp.get_python_type(), dict)
        dtp = TreeDataType(schema=self.get_schema_v3())
        self.assertEqual(dtp.get_python_type(), dict)

    def test_build_numpy_value(self):
        # Case number 1
        dtp = TreeDataType(schema=self.get_schema_v1())

        built_empty = dtp.build_numpy_value({})
        for name in dtp.schema.base_fork_node.get_children_names():
            self.assertTrue(name in built_empty.keys())

        self.assertEqual(built_empty['leaf-string'], 'nan')
        self.assertTrue(np.isnan(built_empty['leaf-float']))
        self.assertTrue(np.isnat(built_empty['leaf-date']))

        built_non_empty = dtp.build_numpy_value({
            'leaf-string': 'tralala', 'leaf-float': 29.23, 'leaf-date': '1993-04-01'})
        for name in dtp.schema.base_fork_node.get_children_names():
            self.assertTrue(name in built_non_empty.keys())

        self.assertEqual(built_non_empty['leaf-string'], 'tralala')
        self.assertEqual(built_non_empty['leaf-float'], float(29.23))
        self.assertEqual(built_non_empty['leaf-date'], np.datetime64('1993-04-01'))

        try:
            dtp.build_numpy_value({'non-existent': 29.23})
        except RuntimeError as e:
            self.assertEqual(str(e), "Unknown node of name 'non-existent' not specified in the Node 'test-fork'")

        # Case number 2
        dtp = TreeDataType(schema=self.get_schema_v2())

        built_empty = dtp.build_numpy_value({})
        for name in dtp.schema.base_fork_node.get_children_names():
            self.assertTrue(name in built_empty.keys())

        self.assertEqual(built_empty['leaf-string'], 'nan')
        self.assertTrue(np.isnan(built_empty['leaf-float']))
        self.assertTrue(np.isnat(built_empty['leaf-date']))
        self.assertEqual(built_empty['new_child'], 'nan')

        built_non_empty = dtp.build_numpy_value({
            'leaf-string': 'tralala', 'leaf-float': 29.23, 'leaf-date': '1993-04-01'})
        for name in dtp.schema.base_fork_node.get_children_names():
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
        dtp = TreeDataType(schema=self.get_schema_v3())

        built_empty = dtp.build_numpy_value({})
        for name in dtp.schema.base_fork_node.get_children_names():
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
        for name in dtp.schema.base_fork_node.get_children_names():
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
        dtp = TreeDataType(schema=self.get_schema_v1())

        built_empty = dtp.build_python_value({})
        for name in dtp.schema.base_fork_node.get_children_names():
            self.assertTrue(name in built_empty.keys())

        self.assertEqual(built_empty['leaf-string'], "None")
        self.assertTrue(built_empty['leaf-float'] is None)
        self.assertEqual(built_empty['leaf-date'], '')

        built_non_empty = dtp.build_python_value({
            'leaf-string': 'tralala', 'leaf-float': 29.23, 'leaf-date': '1993-04-01'})
        for name in dtp.schema.base_fork_node.get_children_names():
            self.assertTrue(name in built_non_empty.keys())

        self.assertEqual(built_non_empty['leaf-string'], 'tralala')
        self.assertEqual(built_non_empty['leaf-float'], float(29.23))
        self.assertEqual(built_non_empty['leaf-date'], datetime(1993, 4, 1))

        try:
            dtp.build_python_value({'non-existent': 29.23})
        except RuntimeError as e:
            self.assertEqual(str(e), "Unknown node of name 'non-existent' not specified in the Node 'test-fork'")

        # Case number 2
        dtp = TreeDataType(schema=self.get_schema_v2())

        built_empty = dtp.build_python_value({})
        for name in dtp.schema.base_fork_node.get_children_names():
            self.assertTrue(name in built_empty.keys())

        self.assertEqual(built_empty['leaf-string'], "None")
        self.assertTrue(built_empty['leaf-float'] is None)
        self.assertEqual(built_empty['leaf-date'], '')
        self.assertEqual(built_empty['new_child'], 'None')

        built_non_empty = dtp.build_python_value({
            'leaf-string': 'tralala', 'leaf-float': 29.23, 'leaf-date': '1993-04-01'})
        for name in dtp.schema.base_fork_node.get_children_names():
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
        dtp = TreeDataType(schema=self.get_schema_v3())

        built_empty = dtp.build_python_value({})
        for name in dtp.schema.base_fork_node.get_children_names():
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
        for name in dtp.schema.base_fork_node.get_children_names():
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
        dtp1 = TreeDataType(schema=self.get_schema_v1())
        dtp2 = TreeDataType(schema=self.get_schema_v1())
        self.assertEqual(dtp1, dtp2)
        dtp1 = TreeDataType(schema=self.get_schema_v2())
        dtp2 = TreeDataType(schema=self.get_schema_v2())
        self.assertEqual(dtp1, dtp2)
        dtp1 = TreeDataType(schema=self.get_schema_v3())
        dtp2 = TreeDataType(schema=self.get_schema_v3())
        self.assertEqual(dtp1, dtp2)
