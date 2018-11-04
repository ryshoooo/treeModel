from .datatypes import DataType, StringDataType, ListDataType
from copy import deepcopy, copy
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
        if not isinstance(value, dict):
            raise AttributeError("Cannot build non-dictionary-like input in TreeDataType!")
        return self.base_fork.build_value(self.get_numpy_type().type(value), 'numpy')

    def build_python_value(self, value):
        """
        Method which converts the input value into the python type value.
        :param value: Value to be converted.
        :return: Converted value of the specific data type.
        """
        if not isinstance(value, dict):
            raise AttributeError("Cannot build non-dictionary-like input in TreeDataType!")

        return self.base_fork.build_value(self.get_python_type()(value), 'python')

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

    def build_value(self, value, method='numpy'):
        """
        Method which builds tree to the specific data type of the tree.
        :param value: Dictionary
        :param method: String specifying the building method (numpy or python)
        :return: Dictionary with its values casted to the correct type.
        """
        value_safe = value.copy()
        if not isinstance(value_safe, dict):
            raise RuntimeError("Incorrect input format of the value!")

        for name in value_safe.keys():
            if name not in self.get_children_names():
                raise RuntimeError("Unknown node of name '{}' not specified in the Node '{}'".format(name, self.name))

        for name in self.get_children_names():
            if method == 'numpy':
                try:
                    value_safe[name] = self.find_child(name).get_data_type().build_numpy_value(value_safe[name])
                except KeyError:
                    child_data_type = self.find_child(name).get_data_type()
                    value_safe[name] = child_data_type.build_numpy_value(child_data_type.numpy_na_value)
            elif method == 'python':
                try:
                    value_safe[name] = self.find_child(name).get_data_type().build_python_value(value_safe[name])
                except KeyError:
                    child_data_type = self.find_child(name).get_data_type()
                    value_safe[name] = child_data_type.build_python_value(child_data_type.python_na_value)
            else:
                raise AttributeError("Method '{}' is not supported!".format(method))

        return value_safe

    def __str__(self):
        return str("{}(\n" + "\t" * self.level + "{}\n" + "\t" * (self.level - 1) + " " * len(self.get_name()) + ")") \
            .format(self.get_name(), ("\n" + "\t" * self.level).join([str(x) for x in self.children]))

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
        if not isinstance(other, ForkNode):
            raise ValueError("Intersection of forks can be performed only on ForkNode objects!")

        if self.get_name() != other.get_name():
            return ForkNode(name="empty", children=[])

        other_children = other.get_children()
        common_children = []
        for child in self.get_children():
            possibles = [x for x in other_children if x.get_name() == child.get_name() and type(x) == type(child)]
            if not len(possibles):
                continue
            elif not len(possibles) - 1:
                if isinstance(child, ForkNode):
                    common_children.append(child * possibles[0])
                elif isinstance(child, ChildNode):
                    if child == possibles[0]:
                        common_children.append(deepcopy(child))
                    elif child.get_data_type() <= possibles[0].get_data_type():
                        common_children.append(deepcopy(possibles[0]))
                    elif child.get_data_type() >= possibles[0].get_data_type():
                        common_children.append(deepcopy(child))
                    else:
                        warn("Data types of child '{}' are incomparable: {}, {}".format(child.get_name(),
                                                                                        child.get_data_type(),
                                                                                        possibles[0].get_data_type()),
                             UserWarning)
                        continue
                else:
                    raise ValueError("Incompatible type of a child: '{}'".format(type(child)))
            else:
                raise RuntimeError(
                    "Impossible tree construction, 2 children cannot have the same name! '{}'".format(child.get_name()))

        return ForkNode(name=self.get_name(), children=common_children, level=self.level)


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

        return TreeSchema(base_fork_node=self.base_fork_node * other.base_fork_node)

    def __add__(self, other):
        """
        Addition method on schema, i.e. union of 2 schemas.
        :param other: TreeSchema
        :return: TreeSchema
        """
        if not isinstance(other, TreeSchema):
            raise ValueError("Intersection of schemas can be performed only on TreeSchema objects!")

        return TreeSchema(base_fork_node=self.base_fork_node + other.base_fork_node)

    @staticmethod
    def _traverse(fork_node, arr_keys):
        """
        Helper method which traverses through the tree.
        :param fork_node: ForkNode in which we currently are.
        :param arr_keys: List of strings representing the keys in order to traverse through.
        :return: Node
        """
        return reduce(lambda x, y: x.find_child(y), arr_keys, fork_node)

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
        return self._traverse(self.base_fork_node, arr_keys).get_data_type()

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
        self._traverse(self.base_fork_node, arr_keys).set_data_type(data_type)

        return self

    def create_dummy_nan_tree(self):
        """
        Create dummy tree with NaN values.
        :return: Dictionary
        """
        return self.base_fork_node.build_value(value={}, method='numpy')
