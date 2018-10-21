from unittest import TestCase

from src.datamodel.tree import ChildNode, ForkNode
from src.datamodel.datatypes import StringDataType, FloatDataType, DateDataType


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
    def get_single_date_leaf():
        return ChildNode(name="leaf-date", data_type=DateDataType(resolution='D'))

    def get_fork_node(self):
        leaf_string = self.get_single_string_leaf()
        leaf_date = self.get_single_date_leaf()
        leaf_float = self.get_single_float_leaf()
        return ForkNode(name='test-fork', children=[leaf_string, leaf_date, leaf_float])

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

    def test_overwrite_children(self):
        single_fork = self.get_fork_node()
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

    def test_overwrite_child(self):
        single_leaf = self.get_single_float_leaf()
        self.assertTrue(single_leaf.is_child())
        self.assertTrue(single_leaf.children is None)
        self.assertEqual(single_leaf.name, 'leaf-float')
        self.assertTrue(isinstance(single_leaf.data_type, FloatDataType))

        single_leaf = single_leaf.overwrite_child(name='new-leaf', data_type=DateDataType(resolution='M'))

        self.assertTrue(single_leaf.is_child())
        self.assertTrue(single_leaf.children is None)
        self.assertEqual(single_leaf.name, 'new-leaf')
        self.assertTrue(isinstance(single_leaf.data_type, DateDataType))

    def test_get_children(self):
        single_fork = self.get_fork_node()
        single_leaf = self.get_single_float_leaf()

        single_fork_children = single_fork.get_children()
        try:
            single_leaf.get_children()
        except Exception as e:
            self.assertTrue(isinstance(e, AttributeError))
            self.assertEqual(str(e), 'Cannot get children from a leaf!')

        self.assertTrue(len(single_fork_children))
        for ind in range(len(single_fork_children)):
            self.assertTrue(single_fork_children[ind] in single_fork.children)
            self.assertEqual(single_fork_children[ind].name, single_fork.children[ind].name)

    def test_get_name(self):
        single_fork = self.get_fork_node()
        single_leaf = self.get_single_float_leaf()

        self.assertEqual(single_fork.name, single_fork.get_name())
        self.assertEqual(single_leaf.name, single_leaf.get_name())

    def test_get_data_type(self):
        single_fork = self.get_fork_node()
        single_leaf = self.get_single_float_leaf()

        try:
            single_fork.get_data_type()
        except Exception as e:
            self.assertTrue(isinstance(e, AttributeError))
            self.assertEqual(str(e), "Cannot get data type from a fork!")

        dtp = single_leaf.get_data_type()
        self.assertTrue(isinstance(dtp, FloatDataType))
