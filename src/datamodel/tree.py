from .datatypes import DataType
from copy import deepcopy
from functools import reduce
import collections
import numpy as np
from warnings import warn


#####################################################
#              TREE FUNCTIONALITY                   #
#####################################################

class TreeDataType(DataType):
    """
    DataType for trees (python dictionaries).

    :param schema: TreeSchema specifying the input tree.
    :param nullable: Boolean specifying whether the data type can contain missing values.
    """

    def __init__(self, base_fork, nullable=True):
        """
        Initialize the data type.
        """

        if not isinstance(base_fork, ForkNode):
            raise AttributeError("Input base fork has to be an instance of ForkNode!")

        self.base_fork = base_fork

        if nullable:
            super(TreeDataType, self).__init__(dict, dict, {}, {})
        else:
            super(TreeDataType, self).__init__(dict, dict, None, None)

    def build_numpy_value(self, value):
        """
        Method which converts the input value into the numpy type.
        :param value: Value to be converted.
        :return: Converted value of the specific data type.
        """
        value_safe = self.get_numpy_type().type(value).copy()

        if not isinstance(value_safe, dict):
            raise RuntimeError("Incorrect input format of the value!")

        for name in value_safe.keys():
            if name not in self.base_fork.get_children_names():
                raise RuntimeError(
                    "Unknown node of name '{}' not specified in the Node '{}'".format(name, self.base_fork.name))

        for name in self.base_fork.get_children_names():
            try:
                value_safe[name] = self.base_fork.find_child(name).get_data_type().build_numpy_value(value_safe[name])
            except KeyError:
                child_data_type = self.base_fork.find_child(name).get_data_type()
                value_safe[name] = child_data_type.build_numpy_value(child_data_type.numpy_na_value)

        return value_safe

    def build_python_value(self, value):
        """
        Method which converts the input value into the python type value.
        :param value: Value to be converted.
        :return: Converted value of the specific data type.
        """
        value_safe = self.get_python_type()(value).copy()
        if not isinstance(value_safe, dict):
            raise RuntimeError("Incorrect input format of the value!")

        for name in value_safe.keys():
            if name not in self.base_fork.get_children_names():
                raise RuntimeError(
                    "Unknown node of name '{}' not specified in the Node '{}'".format(name, self.base_fork.name))

        for name in self.base_fork.get_children_names():
            try:
                value_safe[name] = self.base_fork.find_child(name).get_data_type().build_python_value(value_safe[name])
            except KeyError:
                child_data_type = self.base_fork.find_child(name).get_data_type()
                value_safe[name] = child_data_type.build_python_value(child_data_type.python_na_value)

        return value_safe

    def _compare(self, other, method):
        if not isinstance(other, DataType):
            warn("Cannot compare TreeDataType to '{}'".format(type(other)), UserWarning)
            return False

        if isinstance(other, TreeDataType):
            return self.base_fork.__getattribute__(method)(other.base_fork)

        return any([x.get_data_type().__getattribute__(method.replace("t", "e"))(other)
                    for x in self.base_fork.get_children()])

    def __str__(self):
        return """TreeDataType({})""".format(str(self.base_fork))

    def __eq__(self, other):
        if not isinstance(other, DataType):
            raise AttributeError("Cannot compare TreeDataType to '{}'".format(type(other)))
        elif not isinstance(other, TreeDataType):
            warn("TreeDataType is not a {}".format(type(other)), UserWarning)
            return False
        else:
            return self.base_fork == other.base_fork


class Node(object):
    """
    Main node object, can be considered as a data point or a collection of data points.
    """

    def __init__(self):
        """
        Instantiation of the main node object.
        """
        # Set children nodes, name, value and the data_type of the node to None.
        self.children = None
        self.name = None
        self.data_type = None
        self.value = None

    def is_child(self):
        """
        Simple method to determine whether the node is a leaf.
        :return: Boolean
        """
        return self.children is None and self.name is not None and self.data_type is not None and not isinstance(
            self.data_type, TreeDataType)

    def is_fork(self):
        """
        Simple method to determine whether the node is forking.
        :return: Boolean
        """
        return self.children is not None and self.name is not None and self.data_type is not None and isinstance(
            self.data_type, TreeDataType)

    def get_name(self):
        """
        Get the name of the node.
        :return: String.
        """
        if self.name is None:
            raise RuntimeError("The name of the node is missing!")

        return self.name

    def set_name(self, name):
        """
        Set the name of the node.
        :param name: String
        :return: Node with name set.
        """
        if not isinstance(name, str):
            raise AttributeError("Parameter name has to be a string!")
        self.name = name

        return self

    def get_data_type(self):
        """
        Get the DataType of the node.
        :return: DataType.
        """
        if self.data_type is None:
            raise RuntimeError("The data type is missing!")

        return self.data_type

    def set_data_type(self, data_type):
        """
        Set the data type of the node.
        :param data_type: DataType object.
        :return: Instance of self with updated data type.
        """
        if not isinstance(data_type, DataType):
            raise AttributeError("Parameter data_type has to be an instance of DataType object!")
        self.data_type = deepcopy(data_type)

        return self

    def _compare(self, other, method):
        """
        Generic method to compare Node with other nodes.
        :param other: Node
        :param method: String
        :return: Boolean
        """
        raise NotImplementedError("Cannot compare generic Node!")

    def __le__(self, other):
        return self._compare(other, self.__le__.__name__)

    def __ge__(self, other):
        return self._compare(other, self.__ge__.__name__)

    def __lt__(self, other):
        return self._compare(other, self.__lt__.__name__)

    def __gt__(self, other):
        return self._compare(other, self.__gt__.__name__)


class ForkNode(Node):
    """
    Fork node.

    :param name: Name for the fork.
    :param children: List of Node objects.
    """

    def __init__(self, name, children, level=1):
        """
        Initialize the ForkNode object.
        """
        super(ForkNode, self).__init__()
        self.level = level

        self.set_name(name=name)
        self.set_children(children=children)
        self.data_type = TreeDataType(base_fork=self)

    def set_children(self, children):
        """
        Force method which sets the name and the children leaves to the node.
        :param children: Array-like of Nodes.
        :param name: String specifying the name of the node.
        :return: Instance of the object itself with children and name set.
        """
        if not isinstance(children, (collections.Sequence, np.ndarray)) or isinstance(children, str):
            raise AttributeError("Incorrect format of input children nodes!")

        for child in children:
            if not isinstance(child, Node):
                raise AttributeError("Nodes have to be of Node instance!")

        self.children = deepcopy(children)

        values, counts = np.unique(ar=[x.get_name() for x in children], return_counts=True)

        if len(counts) != 0 and np.max(counts) > 1:
            raise AttributeError(
                "Children nodes with the same name are not allowed! '{}'".format(values[np.argmax(counts)]))

        return self

    def get_children(self):
        """
        Get the list of the children nodes.
        :return: List of Nodes.
        """
        if not self.is_fork():
            raise AttributeError("Cannot get children from a leaf!")

        if self.children is None:
            raise RuntimeError("Empty children leaves!")

        return self.children

    def get_children_names(self):
        """
        Get the list of children names.
        :return: List of strings representing the children names.
        """
        if not self.is_fork():
            raise AttributeError("Cannot get children from a leaf!")

        if self.children is None:
            raise RuntimeError("Empty children leaves!")

        return [x.get_name() for x in self.get_children()]

    def find_child(self, name):
        """
        Find specific child by name
        :param name: String specifying the child's name
        :return: Node
        """
        if not isinstance(name, str):
            raise AttributeError("Input parameter 'name' has to be a string!")

        child_list = [x for x in self.get_children() if x.get_name() == name]

        if not len(child_list):
            raise RuntimeError("Child '{}' was not found in '{}'".format(name, self.name))
        elif not len(child_list) - 1:
            return child_list[0]
        else:
            raise RuntimeError(
                "Impossible error achieved! More than 1 child found with the same "
                "name '{}' in Node '{}'".format(name, self.name))

    def find_child_in_any_branch(self, name, as_fork=False):
        """
        Find specific child by name in all nodes of the fork.
        :param name: String
        :param as_fork: Boolean specifying whether the return value should be fork or not.
        :return: List of Nodes
        """
        if not isinstance(name, str):
            raise ValueError("Input parameter 'name' has to be a string")

        res = []
        for child in self.get_children():
            if child.get_name() == name:
                res.append(deepcopy(child))

            if isinstance(child, ForkNode):
                found_children = child.find_child_in_any_branch(name, as_fork)

                if as_fork and found_children is not None:
                    found_children = [found_children]
                if as_fork and found_children is None:
                    found_children = []

                res += found_children

        if as_fork and not res:
            return None
        if as_fork:
            return ForkNode(self.get_name(), res, self.level)
        else:
            return res

    def is_subtree(self, other, direct=False):
        """
        Method which determines whether an other Fork is a sub-fork of the Fork.
        :param other: ForkNode
        :param direct: Boolean specifying whether the other has to be a direct sub-fork, i.e. not equal
        :return: Boolean
        """
        if not isinstance(other, ForkNode):
            raise ValueError("Parameter other has to be an instance of ForkNode! '{}'".format(type(other)))

        if self.get_name() == other.get_name():
            other_children = other.get_children()
            try:
                res = all([other_child <= self.find_child(other_child.get_name()) for other_child in other_children])
                if res and not direct:
                    return True
                elif res and direct:
                    return not self == other
                else:
                    return res
            except RuntimeError:
                return False

        return any([x.is_subtree(other) for x in self.get_children() if isinstance(x, ForkNode)])

    def _compare(self, other, method):
        if not isinstance(other, (ForkNode, ChildNode)):
            warn("Cannot compare ForkNode to {}".format(type(other)), UserWarning)
            return False

        if isinstance(other, ChildNode):
            if 'g' in method:
                method = method.replace("g", "l")
            else:
                method = method.replace("l", "g")
            return other.__getattribute__(method)(self)

        if 'g' in method:
            return self.is_subtree(other, 'e' not in method)
        else:
            return other.is_subtree(self, 'e' not in method)

    def __str__(self):
        """
        Format the ForkNode into a string, add tab for each child based on the level of the fork.
        :return: String
        """
        children_str = ("\n" + "\t" * self.level).join([str(x) for x in self.children])
        return str("{}(\n" + "\t" * self.level + "{}\n" + "\t" * (self.level - 1) + " " * len(self.get_name()) + ")") \
            .format(self.get_name(), children_str)

    def __eq__(self, other):
        if not isinstance(other, Node):
            raise AttributeError("Cannot compare ForkNode with '{}'".format(type(other)))
        elif not isinstance(other, ForkNode):
            warn("{} is a Fork while {} is a child!".format(self.get_name(), other.get_name()), UserWarning)
            return False

        if self.get_name() != other.get_name():
            warn("{} does not equal in name to {} in a fork!".format(self.get_name(), other.get_name()), UserWarning)
            return False
        else:
            return all(
                [child == other.find_child(child.get_name()) if child.get_name() in other.get_children_names() else
                 False for child in self.get_children()])

    def __mul__(self, other):
        if not isinstance(other, (ChildNode, ForkNode)):
            raise ValueError("Intersection is not defined for type '{}'".format(type(other)))

        if isinstance(other, ChildNode):
            return other * self

        found_branches = other.find_child_in_any_branch(self.get_name())

        if len(found_branches) > 1 or (len(found_branches) > 0 and self.get_name() == other.get_name()):
            raise RuntimeError("Cannot merge 2 forks by the same name! '{}'".format(self.get_name()))

        if self.get_name() == other.get_name():
            children = []
            for child in self.get_children():
                found_children = other.find_child_in_any_branch(child.get_name(), as_fork=False)

                if len(found_children) > 1:
                    raise RuntimeError(
                        "Cannot merge 2 forks with children of the same name! '{}'".format(child.get_name()))

                if not found_children:
                    continue

                merged_children = [child * other_child for other_child in other.get_children()]
                merged_children = [x for x in merged_children if x is not None]

                if len(merged_children) > 1:
                    raise RuntimeError("Impossible error achieved, congratulations!")

                children += merged_children

            if not children:
                return None
            else:
                return ForkNode(name=self.get_name(), children=children, level=self.level)

        if not found_branches:
            return None

        larger_fork = other.find_child_in_any_branch(self.get_name(), as_fork=True)
        larger_forks_merged_children = [x * self for x in larger_fork.get_children()]
        larger_forks_merged_children = [x for x in larger_forks_merged_children if x is not None]

        if not larger_forks_merged_children:
            return None
        else:
            return larger_fork.set_children(larger_forks_merged_children)


class ChildNode(Node):
    """
    Leaf.

    :param name: Name for the child node.
    :param data_type: DataType object specifying the data type for the child.
    """

    def __init__(self, name, data_type):
        """
        Initialize the ChildNode object.
        """
        super(ChildNode, self).__init__()

        self.set_name(name=name)
        self.set_data_type(data_type=data_type)

    def _compare(self, other, method):
        if not isinstance(other, (ForkNode, ChildNode)):
            warn("Cannot compare ChildNode to {}".format(type(other)), UserWarning)
            return False

        if isinstance(other, ForkNode) and method in ('__le__', '__lt__'):
            found_children = other.find_child_in_any_branch(self.get_name())
            return any([self.__getattribute__(method.replace("t", "e"))(x) for x in found_children])
        elif isinstance(other, ForkNode) and method not in ('__le__', '__lt__'):
            return False

        if self.get_name() != other.get_name():
            return False

        return self.get_data_type().__getattribute__(method)(other.get_data_type())

    def __str__(self):
        return """{}({})""".format(self.get_name(), str(self.get_data_type()))

    def __eq__(self, other):
        if not isinstance(other, Node):
            raise AttributeError("Cannot compare ChildNode to '{}'".format(type(other)))
        elif not isinstance(other, ChildNode):
            warn("{} is a child, while {} is a fork".format(self.get_name(), other.get_name()), UserWarning)
            return False

        if self.get_name() != other.get_name():
            warn("{} does not equal in name to {} in a child!".format(self.get_name(), other.get_name()), UserWarning)
            return False
        elif self.get_data_type() != other.get_data_type():
            warn("{}'s data types are different".format(self.get_name()), UserWarning)
            return False
        else:
            return True

    def __mul__(self, other):
        if not isinstance(other, (ChildNode, ForkNode)):
            raise ValueError("Cannot perform intersection on object of type '{}'!".format(tyoe(other)))

        if isinstance(other, ChildNode):
            if self <= other:
                return deepcopy(other)
            elif self >= other:
                return deepcopy(self)
            else:
                return None

        if self.get_name() == other.get_name():
            return None

        found_children = other.find_child_in_any_branch(self.get_name(), as_fork=False)

        if not found_children:
            return None
        elif len(found_children) == 1:
            if isinstance(found_children[0], ChildNode):
                return other.find_child_in_any_branch(self.get_name(), as_fork=True)
            else:
                return None
        elif len(found_children) > 1:
            raise RuntimeError(
                "Cannot perform intersection where there are multiple children of the same name: '{}'".format(
                    self.get_name()))


class TreeSchema(object):
    """
    Base class for input schema for a particular dataset.
    NB: Not a big difference between ForkNode and TreeSchema, it is important to distinguish between them though,
    since ForkNode's functionality is more tree-like, while the schema only gives more metadata about the object.

    :param base_fork_node: ForkNode containing the full tree
    """

    def __init__(self, base_fork_node):
        """
        Initialize the TreeSchema object (basically works as a ForkNode)
        """
        if not isinstance(base_fork_node, ForkNode):
            raise AttributeError("Incorrect format of input base node!")

        self.base_fork_node = base_fork_node

    def _traverse(self, arr_keys):
        """
        Helper method which traverses through the tree.
        :param fork_node: ForkNode in which we currently are.
        :param arr_keys: List of strings representing the keys in order to traverse through.
        :return: Node
        """
        return reduce(lambda x, y: x.find_child(y), arr_keys, self.base_fork_node)

    def find_data_type(self, name):
        """
        Method which finds the data type for the specific node. The name has to be of format
        'level1-name/level2-name/...', i.e. a slash denotes forking.
        :param name: String
        :return: DataType
        """
        if not isinstance(name, str):
            raise ValueError("Parameter 'name' has to be a string!")

        arr_keys = name.split("/")
        return self._traverse(arr_keys).get_data_type()

    def set_data_type(self, name, data_type):
        """
        Method which sets the data type for the specific ndoe. The name has to be of format
        'level1-name/level2-name'...', i.e. a slash denotes forking.
        :param name: String
        :param data_type: DataType
        :return: TreeSchema
        """
        if not isinstance(name, str):
            raise ValueError("Parameter 'name' has to be a string!")
        if not isinstance(data_type, DataType):
            raise ValueError("Parameter 'data_type' has to be a DataType!")

        arr_keys = name.split("/")
        self._traverse(arr_keys).set_data_type(data_type)

        return self

    def create_dummy_nan_tree(self, method='numpy'):
        """
        Create dummy tree with NaN values.
        :param method: String speciyfing the build method
        :return: Dictionary
        """
        if method == 'numpy':
            return self.base_fork_node.get_data_type().build_numpy_value(value={})
        elif method == 'python':
            return self.base_fork_node.get_data_type().build_python_value(value={})
        else:
            raise ValueError("Unknown method: '{}'".format(method))

    def __str__(self):
        """
        String method on schema.
        :return: String
        """
        return str(self.base_fork_node)

    def __eq__(self, other):
        """
        Equality method on schema.
        :param other: TreeSchema
        :return: Boolean
        """
        if not isinstance(other, TreeSchema):
            raise AttributeError("Cannot compare TreeSchema with '{}'".format(type(other)))

        return self.base_fork_node == other.base_fork_node

    def __mul__(self, other):
        """
        Multiplication method on schema, i.e. intersection of 2 schemas.
        :param other: TreeSchema
        :return: TreeSchema
        """
        if not isinstance(other, TreeSchema):
            raise ValueError("Intersection of schemas can be performed only on TreeSchema objects!")

        merged_base_forks = self.base_fork_node * other.base_fork_node

        if merged_base_forks is None:
            return TreeSchema(base_fork_node=ForkNode('empty', []))
        else:
            return TreeSchema(base_fork_node=merged_base_forks)

    def __add__(self, other):
        """
        Addition method on schema, i.e. union of 2 schemas.
        :param other: TreeSchema
        :return: TreeSchema
        """
        if not isinstance(other, TreeSchema):
            raise ValueError("Intersection of schemas can be performed only on TreeSchema objects!")

        return TreeSchema(base_fork_node=self.base_fork_node + other.base_fork_node)
